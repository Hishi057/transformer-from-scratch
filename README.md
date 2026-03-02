# The Annotated Transformer

https://nlp.seas.harvard.edu/annotated-transformer/

この資料をもとに、transformerをスクラッチ実装してみる。

## テンソルの形状

このtransformerの実装およびpytorch(自然言語処理？)において、データは以下の形式でやりとりされる。
このような形式であれば、GPUでの計算が効率的に行えるためである。

> (Batch(文章の数), Length(文章の長さ), Dimension(単語ベクトルの長さ))

また、マルチヘッドで入力を分割する際には以下のようになる

> (Batch, Head, Length, Dimension)

Dimensionが分割されて、何番目の分割かをHeadで見る感じ。

**具体例** 
(B, L, D) = (32, 100, 400)

入力が32つの文章ごとに区切られる。一文章あたりの単語数は100つで、400次元の単語ベクトルからそれぞれ構成される。
余った分は無意味な要素を埋める(padding)。

マルチヘッドで8分割すると、
(32, 8, 100, 50)
となる。