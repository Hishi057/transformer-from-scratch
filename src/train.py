from model import *
from torch.optim.lr_scheduler import LambdaLR
from utils import execute_example, DummyOptimizer, DummyScheduler

class Batch:
    def __init__(self, src, tgt=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask

class TrainState:
    step: int = 0
    accum_step: int = 0
    samples: int = 0
    tokens: int = 0

def run_epoch(
        data_iter,
        model,
        loss_compute,
        optimizer,
        scheduler,
        mode="train",
        accum_iter=1,
        train_state=TrainState()
):
    start = time.time()
    total_loss = 0
    total_tokens = 0
    tokens = 0
    n_accum = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)

        if mode == "train" or mode == "train+log":
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()
        
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 20 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.4f "
                    + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        del loss
        del loss_node
    return total_loss / total_tokens, train_state

def rate(step, model_size, factor, warmup):
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )

class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing = 0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
    
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())

#
# First Exmaple
#
def data_gen(V, batch_size, nbatches):
    # yield from data_gen_copy(V, batch_size, nbatches)
    yield from data_gen_sort(V, batch_size, nbatches)

# コピータスク
def data_gen_copy(V, batch_size, nbatches):
    for i in range(nbatches):
        data = torch.randint(2, V, size=(batch_size, 10))
        data[:, 0] = 1
        src = data.requires_grad_(False).clone().detach()
        tgt = data.requires_grad_(False).clone().detach()
        yield Batch(src, tgt, pad=0)

# ソートタスク
def data_gen_sort(V, batch_size, nbatches):
    for i in range(nbatches):
        current_len = torch.randint(5, 15, (1,)).item()
        body = torch.randint(2, V, size=(batch_size, current_len))
        src = torch.cat([torch.ones(batch_size, 1).long(), body], dim=1)
        body_sorted = torch.sort(body, dim=1)[0]
        tgt = torch.cat([torch.ones(batch_size, 1).long(), body_sorted], dim=1)
        yield Batch(src, tgt, pad=0)

class SimpleLossCompute:
    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion
    
    def __call__(self, x, y, norm):
        x = self.generator(x)
        sloss = (
            self.criterion(
                x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)) / norm
            )
        return sloss.data * norm, sloss

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([
            ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )

    return ys

def example_simple_model(epoch=50):
    V = 11
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = make_model(V, V, N=2, d_model=128, d_ff=512, h=8)

    optimizer = torch.optim.Adam(
        model.parameters(), lr = 1.0, betas = (0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, model_size=model.src_embed[0].d_model, factor=1.0, warmup=100
        ),
    )

    # 学習
    batch_size = 64
    for e in range(epoch):
        print("Epoch : ", e)
        model.train()
        run_epoch(
            data_gen(V, batch_size, 50),
            model,
            SimpleLossCompute(model.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train",
        )
        model.eval()
        run_epoch(
            data_gen(V, batch_size, 5),
            model,
            SimpleLossCompute(model.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval"
        )[0]
    
    model.eval()

    srcs = [torch.LongTensor([[2,3,4,5]]), 
            torch.LongTensor([[5,3,4,2]]), 
            torch.LongTensor([[9,8,7,6,5]]), 
            torch.LongTensor([[2,5,4,3,6,7,8]]), 
            torch.LongTensor([[2,5,4,3,6,7,8]]), 
            torch.LongTensor([[3,5,7,2,4,8,9,6]]), 
            torch.LongTensor([[5,4,3,3,4,2,2,5]]), 
            torch.LongTensor([[2,3,5,4,8,7,6,9,7,8,6,5,2,3,4]]), 
            ]
    
    counter = 0
    for src_body in srcs:
        src =  torch.cat([torch.ones(1, 1).long(), src_body], dim=1)
        max_len = src.shape[1]
        src_mask = torch.ones(1, 1, max_len)
        print("\n入力: ", src.tolist())
        print("出力: ", greedy_decode(model, src, src_mask, max_len=max_len, start_symbol=1).tolist())

if __name__ == "__main__":
    # execute_example(example_simple_model)
    print("学習を開始します...") 
    example_simple_model(epoch=20)
    print("学習が終了しました。")