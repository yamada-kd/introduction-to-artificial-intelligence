#!/usr/bin/env python
# coding: utf-8

# # アテンションネットワーク

# ## 基本的な事柄

# この節ではアテンションがどのようなものなのか，また，アテンションの計算方法について紹介します．

# ### アテンションとは

# アテンション機構とは単にアテンションとも呼ばれる，ニューラルネットワークが入力データのどの部分に注目するかを明示することで予測の性能向上をさせるため技術です．例えば，何らかの風景が写っている画像を人工知能に処理させ，その画像に関する何らかの説明を自然言語によってさせるタスクを考えます．そのような場合において，人工知能が空の色に関する話題に触れる場合には，空が写っている部分の色を重点的に処理するかもしれないし，海の大きさを話題にする場合は海が写っている部分の面積を重点的に処理するかもしれません．少なくとも人は何かの意思決定をする際に，意識を対象に集中させることで複雑な情報から自身が処理すべき情報のみを抽出する能力を有しています．人工知能は人とは全く違う情報処理をしているかもしれないので，このような直接的な対応があるかどうかは定かではないですが，人の情報処理方法にヒントを得て，このような注意を払うべき部分を人工知能に教えるための技術がアテンション機構です．アテンションを有するニューラルネットワークの性能は凄まじく，アテンションネットワークの登場は CNN や RNN 等のアテンション以前に用いられていたニューラルネットワークを過去のものとしました．

# ```{note}
# 以前の章で紹介した CNN や RNN は簡潔にまとめました．CNN や RNN よりアテンションを利用すれば良いのではないかという考えです．
# ```

# ### アテンションの計算方法

# アテンション機構の処理は非常に単純で簡単な式によって定義されます．アテンションの基本的な計算は以下に示す通りです．アテンション機構とは，$[\boldsymbol{h}_1,\boldsymbol{h}_2,\dots,\boldsymbol{h}_I]$ のようなデータに対して以下のような計算をして，$\boldsymbol{c}$ を得るものです．
# 
# $
# \displaystyle \boldsymbol{c}=\sum_{i=1}^I{\phi}_i\boldsymbol{h}_i
# $
# 
# このとき，スカラである $\phi_i$ は $[0,1]$ の範囲にあり，また，以下の式を満たす値です．
# 
# $
# \displaystyle \sum_{i=1}^I{\phi}_i=1
# $
# 
# すなわち，${\phi}_i$ は何らかの入力に対して得られるソフトマックス関数の出力の $\boldsymbol{\phi}$ の $i$ 番目の要素です．よって，この $\boldsymbol{c}$ は単に $[\boldsymbol{h}_1,\boldsymbol{h}_2,\dots,\boldsymbol{h}_I]$ の加重平均です．人工知能にこのベクトル $\boldsymbol{c}$ を入力値として処理させる場合，$\boldsymbol{h}_i$ に対する ${\phi}_i$ の値を大きな値とすることで $\boldsymbol{h}_i$ が注目すべき要素として認識されるという仕組みです．

# ```{note}
# ソフトマックス関数とはベクトルを入力として，各要素が0から1の範囲内にあり，各要素の総和が1となるベクトルを出力する関数のひとつでした．出力データを確率として解釈できるようになるというものです．入力ベクトルと出力ベクトルの長さは等しいです．
# ```

# ### セルフアテンション

# あるベクトル（ソースベクトル）の各要素が別のベクトル（ターゲットベクトル）のどの要素に関連性が強いかということを明らかにしたい際にはソース・ターゲットアテンションというアテンションを計算します．これに対してこの章ではセルフアテンションを主に扱うのでこれを紹介します．セルフアテンションとは配列データの各トークン間の関連性を計算するためのものです．
# 
# セルフアテンションを視覚的に紹介します．ここでは，$x_1$，$x_2$，$x_3$ という 3 個のトークン（要素）からなるベクトルを入力として $y_1$，$y_2$，$y_3$ という 3 個のトークンからなるベクトルを得ようとします．各々のトークンもベクトルであるとします．
# 
# <img src="https://github.com/yamada-kd/introduction-to-artificial-intelligence/blob/main/image/selfAttention.svg?raw=1" width="100%" />
# 
# アテンションの計算では最初に，$x_1$ に 3 個の別々の重みパラメータを掛け，キー，バリュー，クエリという値を得ます．図ではぞれぞれ，$k_1$，$v_1$，$q_1$ と表されています．同様の計算を $x_2$ と $x_3$ についても行います．この図は既に $y_1$ の計算は終了して $y_2$ の計算が行われているところなので，$x_2$ を例に計算を進めます．$x_2$ に対して計算したクエリの値，$q_2$ とすべてのキーの値との内積を計算します．図中の $\odot$ は内積を計算する記号です．これによって得られた 3 個のスカラ値をその合計値が 1 となるように標準化します．これによって，0.7，0.2，0.1 という値が得られています．次にこれらの値をバリューに掛けます．これによって得られた 3 個のベクトルを足し合わせた値が $y_2$ です．これがセルフアテンションの計算です．クエリとキーの内積を計算することで，元のトークンと別のトークンの関係性を計算し，それを元のトークンに由来するバリューに掛けることで最終的に得られるベクトル（ここでは $y_2$）は別のすべてのトークンの情報を保持しているということになります．文脈情報を下流のネットワークに伝えることができるということです．
# 
# 以下では数式を利用してセルフアテンションを紹介します．入力配列データを $x$ とします．これは，$N$ 行 $m$ 列の行列であるとします．つまり，$N$ 個のトークンからなる配列データです．各トークンの特徴量は $m$ の長さからなるということです．セルフアテンションの計算では最初に，クエリ，キー，バリューの値を計算するのでした．$W_q$，$W_k$，$W_v$ という 3 個の重みパラメータを用いて以下のように計算します．この重みパラメータはそれぞれ $m$ 行 $d$ 列であるとします．
# 
# $
# Q=  xW_q
# $
# 
# $
# K=  xW_k
# $
# 
# $
# V=  xW_v
# $
# 
# これらを用いてアテンション $A$ は以下のように計算します．
# 
# $
# \displaystyle A(x)=\sigma\left(\frac{QK^\mathsf{T}}{\sqrt{d}}\right)V
# $
# 
# この式の $\sigma$ は $QK^\mathsf{T}$ の行方向に対して計算するソフトマックス関数です．これでアテンションの計算は終わりなのですが，アテンションの計算に出てくる項の掛け算の順番を変更するという操作でアテンションの計算量を改良した線形アテンションというものがあるのでそれもあわせて紹介します．線形アテンション $L$ は以下のように計算します．
# 
# $
# L(x)=\tau(Q)(\tau(K)^\mathsf{T}V)
# $
# 
# ここで，$\tau$ は以下の関数です．
# 
# $
# \displaystyle \tau(x)=\begin{cases}
# x+1 & (x > 0) \\
# e^x & (x \leq 0)
# \end{cases}
# $
# 
# アテンションの時間計算量は配列の長さ $N$ に対して $O(N^2)$ なのですが，線形アテンションの時間計算量は線形 $O(N)$ です．とても簡単な工夫なのに恐ろしくスピードアップしています．

# ```{note}
# この項ではベクトルを英小文字の太字ではなくて普通のフォントで記述していることに注意してください．
# ```

# ## トランスフォーマー

# トランスフォーマーはアテンションの最も有用な応用先のひとつです．トランスフォーマーの構造やその構成要素を紹介します．

# ### 基本構造

# トランスフォーマーとはアテンションと位置エンコード（ポジショナルエンコード）といわれる技術を用いて，再帰型ニューラルネットワークとは異なる方法で文字列を処理することができるニューラルネットワークの構造です．機械翻訳や質問応答に利用することができます．
# 
# 例えば，機械翻訳の場合，翻訳したい文字列を入力データ，翻訳結果の文字列を教師データとして利用します．構築した人工知能は翻訳したい文字列を入力値として受け取り，配列を出力します．配列の各要素は文字の個数と同じサイズのベクトル（その要素が何の文字なのかを示す確率ベクトル）です．
# 
# トランスフォーマーはエンコーダーとデコーダーという構造からなります．エンコーダーは配列（機械翻訳の場合，翻訳したい配列）を入力にして，同じ長さの配列を出力します．デコーダーも配列（機械翻訳の場合，翻訳で得たい配列）とエンコーダーが出力した配列を入力にして同じ長さの配列（各要素は確率ベクトル）を出力します．エンコーダーが出力した配列情報をデコーダーで処理する際にアテンションが利用されます．
# 
# <img src="https://github.com/yamada-kd/introduction-to-artificial-intelligence/blob/main/image/transformer.svg?raw=1" width="100%" />
# 
# エンコーダーとデコーダー間のアテンション以外にも，エンコーダーとデコーダーの内部でもそれぞれアテンション（セルフアテンション）が計算されます．アテンションは文字列内における文字の関連性を計算します．
# 
# トランスフォーマーは再帰型ニューラルネットワークで行うような文字の逐次的な処理が不要です．よって，計算機の並列化性能をより引き出せます．扱える文脈の長さも無限です．
# 
# このトランスフォーマーはものすごい性能を発揮しており，これまでに作られてきた様々な構造を過去のものとしました．最初に応用分野で脚光が当たったのはトランスフォーマーのエンコーダーの部分です．BERT と呼ばれる方法を筆頭に自然言語からなる配列を入力にして何らかの分散表現を出力する方法として自然言語処理に関わる様々な研究開発に利用されています．その後，デコーダーの部分を利用して，入力単語の次に出現する単語を予測するモデルのひとつである GPT を基にした ChatGPT をはじめとする大規模言語モデルが注目を集めました．

# ```{note}
# 再帰型ニューラルネットワークでも扱える配列の長さは理論上無限です．
# ```

# ```{note}
# 会話でトランスフォーマーという場合は，トランスフォーマーのエンコーダーまたはデコーダーのことを言っている場合があります．エンコーダー・デコーダー，エンコーダー，デコーダー，この3個でそれぞれできることが異なります．
# ```

# ### 構成要素

# トランスフォーマーは大きく分けると 3 個の要素から構成されています．アテンション（セルフアテンション），位置エンコード，ポイントワイズ MLP です．アテンションは上の節で紹介した通り，入力データのトークン間の関係性を下流のネットワークに伝えるための機構です．
# 
# **位置エンコード**は入力データの各トークンの位置関係を下流のネットワークに伝える仕組みです．RNN では入力された配列情報を最初，または，最後から順番に処理します．よって，人工知能は読み込まれた要素の順序，位置情報を明確に認識できています．これに対して，セルフアテンションを導入して文脈情報を扱えるようにした MLP はその認識ができません．アテンションで計算しているものは要素間の関連性であり，位置関係は考慮されていないのです．例えば，「これ　は　私　の　ラーメン　です」という文を考えた場合，アテンション機構では「これ」に対して「は」，「私」，「の」，「ラーメン」，「です」の関連度を計算していますが，「2 番目のタイムステップのは」や「3 番目のタイムステップの私」を認識しているわけではありません．よって，この文章の要素を並び替え，「は　です　の　私　これ　ラーメン」としても同じ計算結果が得られます．この例においては，そのような処理をしたとしても最終的な結果に影響はないかもしれませんが，例えば，「ラーメン　より　牛丼　が　好き　です　か」のような文章を考えた場合には問題が生じます．この文章では「ラーメンより牛丼が好き」かということを問うているのであって，「牛丼よりラーメンが好き」かということは問うていないのです．「より」の前後にある文字列の登場順が重要な意味を持ちます．このような情報を処理する際には各要素の位置情報を人工知能に認識させる必要があります．これは様々な方法で実現できますが，トランスフォーマーで用いている方法を紹介します．ここでは，以下のような配列情報を処理するものとします．
# 
# $
# \boldsymbol{x}=[[x_{11},x_{12},\dots,x_{1d}],[x_{21},x_{22},\dots,x_{2d}],\dots,[x_{l1},x_{l2},\dots,x_{ld}]]
# $
# 
# すなわち，配列 $\boldsymbol{x}$ は $l$ 個の長さからなり，その $l$ 個の各要素は $d$ 個の要素からなるベクトルです．この $[x_{11},x_{12},\dots,x_{1d}]$ のような要素は例えば自然言語においては単語のことを示します．単語が $d$ 個の要素からなる何らかのベクトルとしてエンコードされているとします．このような配列情報に位置情報を持たせるには以下のような位置情報を含んだ配列を用意します．
# 
# $
# \boldsymbol{p}=[[p_{11},p_{12},\dots,p_{1d}],[p_{21},p_{22},\dots,p_{2d}],\dots,[p_{l1},p_{l2},\dots,p_{ld}]]
# $
# 
# この配列は入力配列 $\boldsymbol{x}$ と同じ形をしています．よって $\boldsymbol{x}$ に加算することができます．$\boldsymbol{x}$ に $\boldsymbol{p}$ を加算することで位置情報を保持した配列を生成することができるのです．トランスフォーマーでは位置情報を含んだ配列の要素 $p_{ik}$ を $p_{i(2j)}$ と $p_{i(2j+1)}$ によって，トークンを示すベクトルの要素の偶数番目と奇数番目で場合分けし，それぞれ正弦関数と余弦関数で定義します．偶数番目は以下のように表されます．
# 
# $
# \displaystyle p_{i(2j)}=\sin\left(\frac{i}{10000^{\frac{2j}{d}}}\right)
# $
# 
# 奇数番目は以下のように表されます．
# 
# $
# \displaystyle  p_{i(2j+1)}=\cos\left(\frac{i}{10000^{\frac{2j}{d}}}\right)
# $
# 
# 
# **ポイントワイズ MLP** は可変長の配列データを処理するための MLP です．機械学習モデルを学習させる際には様々なデータをモデルに読み込ませますが，配列データにおいては，それらのデータの長さが不揃いであることがあります．そのような場合において固定長のデータを処理するための方法である MLP を使おうとするとどのような長さのデータにも対応させるためにはかなり長い入力層のサイズを用意しなければなりません．かなり長い入力層を利用したとしても，その入力層よりさらに長い入力データを処理しなければならない可能性は排除できません．これに対してポイントワイズ MLP は入力配列の各トークンに対してたったひとつの MLP の計算をする方法です．入力配列のどのトークンに対しても同じパラメータを持った MLP の計算が行われます．10 の長さの入力配列に対しては同じ MLP による計算が 10 回行われ，10 の長さの配列が出力されます．100 の長さの入力配列に対しては同じ MLP による計算が 100 回行われ，100 の長さの配列が出力されます．

# ### 現実世界での利用方法

# トランスフォーマーの現実世界での応用先としては，感情分析，特徴抽出，穴埋め，固有表現抽出（文章中の固有表現を見つける），質問応答，要約，文章生成，翻訳等があります．これらの問題を解決しようとする際には，実際には，事前学習モデルを活用する場合が多いです．事前学習モデルとは解きたい問題ではなくて，あらかじめ別の何らかの問題に対して（大量の）データを利用して学習した学習済みのモデルです．学習済みモデルにはそのドメインにおける知識を獲得していることを期待しています．
# 
# 事前学習モデルとして有名なものには BERT（bidirectional encoder representations from transformers）があります．上述のようにトランスフォーマーはエンコーダー・デコーダー構造を有しますが，BERT はトランスフォーマーのエンコーダー構造を利用して構築されるものです．BERT は自然言語からなる配列データを入力として何らかの配列データを出力する汎用言語表現モデルです．利用方法は多岐にわたりますが，自然言語の分散表現を生成するモデルと言えます．事前学習モデルとして公開されており，自然言語からなる配列を入力として得られる BERT の出力を別の何らかの問題を解くための人工知能アルゴリズムの入力として用いることで，様々な問題を解決するための基礎として利用することができます．
# 
# また，GPT（generative pre-trained transformer）も有名な事前学習モデルです．GPT は何らかの文字列が入力されるとその次に出力されるべき文字（単語）を出力する人工知能です．質問文を入力することでそれに対する回答文を生成できますが，それを対話エージェントのサービスとして提供したものが ChatGPT です．

# ## エンコーダーの実装

# この節ではトランスフォーマーのエンコーダーに相当する部分を実装します．

# ### 扱うデータ

# この節では以下のようなデータを利用して，3 個の値の分類問題を解きます．入力データの長さは一定ではありません．このデータには特徴があります．`0` に分類される 3 個の入力データの最初の要素はそれぞれ `1`，`3`，`7` です．また，`1` に分類される 3 個の入力データの最初の要素もそれぞれ `1`，`3`，`7` です．さらに，`2` に分類される 3 個の入力データの最初の要素も同じく `1`，`3`，`7` です．
# 
# 入力ベクトル | ターゲットベクトル
# :---: | :---:
# [1, 2, 3, 4, 5, 6, 7, 8, 9, 1] | [0]
# [3, 9, 3, 4, 7] | [0]
# [7, 5, 8] | [0]
# [1, 5, 8] | [1]
# [3, 9, 3, 4, 6] | [1]
# [7, 3, 4, 1] | [1]
# [1, 3] | [2]
# [3, 9, 3, 4, 1] | [2]
# [7, 5, 5, 7, 7, 5] | [2]
# 
# このような入力データに対してアテンションの計算を行い，その後ポイントワイズ MLP の計算を行った後に，その出力ベクトルの最初の要素（元データの最初のトークンに対応する値）のみに対して MLP の計算を行い，ターゲットベクトルの予測をします．図で示すと以下のようなネットワークです．つまり，この問題はアテンションの出力（の最初の要素）が入力データの全部の文脈情報を保持できていないと解けない問題です．
# 
# <img src="https://github.com/yamada-kd/introduction-to-artificial-intelligence/blob/main/image/contextNetwork.svg?raw=1" width="100%" />
# 

# ### アテンションクラス

# PyTorch に実装されているアテンションのクラスを利用してこの問題を解くには以下のようなコードを書きます．

# In[ ]:


#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
torch.manual_seed(0)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # データの生成．
    trainx = [
        torch.tensor([1,2,3,4,5,6,7,8,9,1], dtype=torch.long),
        torch.tensor([3,9,3,4,7], dtype=torch.long),
        torch.tensor([7,5,8], dtype=torch.long),
        torch.tensor([1,5,8], dtype=torch.long),
        torch.tensor([3,9,3,4,6], dtype=torch.long),
        torch.tensor([7,3,4,1], dtype=torch.long),
        torch.tensor([1,3], dtype=torch.long),
        torch.tensor([3,9,3,4,1], dtype=torch.long),
        torch.tensor([7,5,5,7,7,5], dtype=torch.long)
    ]
    traint = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int64)
    vocab_size = max(max(seq) for seq in trainx) + 1
    trainx = pad_sequence(trainx, batch_first=True, padding_value=0).to(device)
    traint = torch.tensor(traint, dtype=torch.long).to(device)

    # ハイパーパラメータ．
    embed_size = 16
    attention_unit_size = 32
    output_size = 3
    minibatch_size = 3

    # モデルの生成．
    model = Network(vocab_size, embed_size, attention_unit_size, output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    dataset = TensorDataset(trainx, traint)
    dataloader = DataLoader(dataset, batch_size=minibatch_size, shuffle=True)

    # 学習ループ．
    for epoch in range(1000 + 1):
        totalcost, totalacc = 0, 0
        for tx, tt in dataloader:
            optimizer.zero_grad()
            ty = model(tx)
            cost = criterion(ty, tt)
            totalcost += cost
            tp = ty.argmax(dim=1)
            totalacc += (tp == tt).sum().item() / len(trainx)
            cost.backward()
            optimizer.step()
        totalcost /= len(dataloader)
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Cost {totalcost:.4f}, Acc {totalacc:.4f}")
            with torch.no_grad():
                model.eval()
                trainp = model(trainx)
                print("Prediction:", torch.argmax(trainp, axis=1))

class Network(nn.Module):
    def __init__(self, vocab_size, embed_size, attention_unit_size, output_size):
        super(Network, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size, padding_idx=0)
        self.positional_encoder = PositionalEncoder(embed_size)
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=1, batch_first=True) # アテンションの計算．num_heads=1のときはシンプルなアテンションの計算．
        self.fc1 = nn.Linear(embed_size, attention_unit_size)
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(attention_unit_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoder(x) # 位置情報を入力データに追加．
        x, _ = self.attention(x, x, x)
        x = self.fc1(x[:, 0, :]) # 配列の最初のタイムステップだけを以降の計算に利用．
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class PositionalEncoder(nn.Module):
    def __init__(self, input_size, max_seq_len=500):
        super(PositionalEncoder, self).__init__()
        self.input_size = input_size
        pe = torch.zeros(max_seq_len, input_size) # 位置エンコーディング用のゼロ行列の生成．
        for pos in range(max_seq_len):
            for i in range(0, input_size, 2): # サインとコサインの計算のため2個ごとにスキップ．
                pos_tensor = torch.tensor(pos, dtype=torch.float32)
                base = torch.tensor(10000.0, dtype=torch.float32) # 位置エンコードの計算．
                denominator = torch.pow(base, (2 * i) / input_size) # 位置エンコードの計算．
                pe[pos, i] = torch.sin(pos_tensor / denominator) # 位置エンコードの計算．
                pe[pos, i + 1] = torch.cos(pos_tensor / denominator) # 位置エンコードの計算．
        pe = pe.unsqueeze(0) # unsqueeze()は指定した位置に次元数を追加するもの．これにてミニバッジのための次元を追加．
        self.register_buffer("pe", pe) # register_buffer()はモデルの一部として保存されるべきデータを登録するためのもの．これで位置エンコードをモデルに組み込む．

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].detach()
        return x

if __name__ == '__main__':
    main()


# プログラムを実行すると以下のような結果が得られたと思います．この `Prediction` が入力データに対する予測結果です．正解を導けたことがわかります．
# 
# ```shell
# .
# .
# .
# Epoch 1000: Cost 0.0000, Acc 1.0000
# Prediction: tensor([0, 0, 0, 1, 1, 1, 2, 2, 2], device='cuda:0')
# ```

# 以降でプログラムの説明をします．以下の部分はデータを生成するための記述です．入力配列の長さを揃えるためにゼロパディングをしています．
# 
# ```python
#     # データの生成．
#     trainx = [
#         torch.tensor([1,2,3,4,5,6,7,8,9,1], dtype=torch.long),
#         torch.tensor([3,9,3,4,7], dtype=torch.long),
#         torch.tensor([7,5,8], dtype=torch.long),
#         torch.tensor([1,5,8], dtype=torch.long),
#         torch.tensor([3,9,3,4,6], dtype=torch.long),
#         torch.tensor([7,3,4,1], dtype=torch.long),
#         torch.tensor([1,3], dtype=torch.long),
#         torch.tensor([3,9,3,4,1], dtype=torch.long),
#         torch.tensor([7,5,5,7,7,5], dtype=torch.long)
#     ]
#     traint = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int64)
#     vocab_size = max(max(seq) for seq in trainx) + 1
#     trainx = pad_sequence(trainx, batch_first=True, padding_value=0).to(device)
#     traint = torch.tensor(traint, dtype=torch.long).to(device)
# ```

# ハイパーパラメータの設定部分やデータサイズ等の取得の部分の説明は省略して，以下の部分ですが，これはモデルを生成するための記述です．この問題は分類問題なのでクロスエントロピーをコスト関数にします．
# 
# ```python
#     # モデルの生成．
#     model = Network(vocab_size, embed_size, attention_unit_size, output_size).to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters())
# ```

# 学習ループの記述は以下の部分です．データローダーでデータを読み込み，勾配の初期化，モデルによる予測値の出力，コストの計算，微分，パラメータ更新を行います．
# 
# ```python
#     # 学習ループ．
#     for epoch in range(1000 + 1):
#         totalcost, totalacc = 0, 0
#         for tx, tt in dataloader:
#             optimizer.zero_grad()
#             ty = model(tx)
#             cost = criterion(ty, tt)
#             totalcost += cost
#             tp = ty.argmax(dim=1)
#             totalacc += (tp == tt).sum().item() / len(trainx)
#             cost.backward()
#             optimizer.step()
#         totalcost /= len(dataloader)
#         if epoch % 50 == 0:
#             print(f"Epoch {epoch}: Cost {totalcost:.4f}, Acc {totalacc:.4f}")
#             with torch.no_grad():
#                 trainp = model(trainx)
#                 print("Prediction:", torch.argmax(trainp, axis=1))
# ```

# 次に，ネットワークの全体構造の説明をします．コンストラクタの部分ですが，エンベッド，位置エンコード，アテンションのクラスを利用する点が他の章と異なります．エンベッドとは質的変数を指定したサイズのベクトルに包埋するためのものです．計算機で処理させるために行います．ここで行っているのはワンホットエンコーディングではなくて，浮動小数点数からなるベクトルへの包埋です．位置エンコードのためのクラスは新たに生成しました．後で説明します．次のアテンションのクラスではエンベッドのサイズと `num_heads=1` を指定しています．アテンションは複数個用いることでデータに内包される様々な特徴を別々に処理することが可能なのですが，この値によって，利用するアテンションの個数を決定します．その他の変数はよくある構成要素です．これらを用いて，このネットワークでは入力データの各トークをエンベッドし，位置エンコードし，アテンションの計算を行い，ポイントワイズ MLP（`self.fc1`）の計算をアテンションの計算結果の最初の要素（`x[:, 0, :]`）のみに行い，その後に MLP（`self.fc2`）の計算をします．
# 
# ```python
# class Network(nn.Module):
#     def __init__(self, vocab_size, embed_size, attention_unit_size, output_size):
#         super(Network, self).__init__()
#         self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size, padding_idx=0)
#         self.positional_encoder = PositionalEncoder(embed_size)
#         self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=1, batch_first=True) # アテンションの計算．num_heads=1のときはシンプルなアテンションの計算．
#         self.fc1 = nn.Linear(embed_size, attention_unit_size)
#         self.dropout = nn.Dropout()
#         self.fc2 = nn.Linear(attention_unit_size, output_size)
#         self.relu = nn.ReLU()
# 
#     def forward(self, x):
#         x = self.embedding(x)
#         x = self.positional_encoder(x) # 位置情報を入力データに追加．
#         x, _ = self.attention(x, x, x)
#         x = self.fc1(x[:, 0, :]) # 配列の最初のタイムステップだけを以降の計算に利用．
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return x
# ```

# ```{note}
# アテンションの計算結果として得られた配列の最初の要素の値だけをそれ以降の計算に利用しました．つまり，アテンションが入力配列の全部の文脈を理解できていないなら，正解は導き出せようにもありません．
# ```

# 位置エンコードのためのクラスは以下に示す通りです．位置エンコードの式に沿った記述です．最初に $2j$（`j`）を計算し，$10000^{(2j/d)}$（`denominator`）を求めます．最終的に得られた位置エンコードの値を入力データに加えることで位置エンコード情報の付加が終わります．
# 
# ```python
# class PositionalEncoder(nn.Module):
#     def __init__(self, input_size, max_seq_len=500):
#         super(PositionalEncoder, self).__init__()
#         self.input_size = input_size
#         pe = torch.zeros(max_seq_len, input_size) # 位置エンコーディング用のゼロ行列の生成．
#         for pos in range(max_seq_len):
#             for i in range(0, input_size, 2): # サインとコサインの計算のため2個ごとにスキップ．
#                 pos_tensor = torch.tensor(pos, dtype=torch.float32)
#                 base = torch.tensor(10000.0, dtype=torch.float32) # 位置エンコードの計算．
#                 denominator = torch.pow(base, (2 * i) / input_size) # 位置エンコードの計算．
#                 pe[pos, i] = torch.sin(pos_tensor / denominator) # 位置エンコードの計算．
#                 pe[pos, i + 1] = torch.cos(pos_tensor / denominator) # 位置エンコードの計算．
#         pe = pe.unsqueeze(0) # unsqueeze()は指定した位置に次元数を追加するもの．これにてミニバッジのための次元を追加．
#         self.register_buffer("pe", pe) # register_buffer()はモデルの一部として保存されるべきデータを登録するためのもの．これで位置エンコードをモデルに組み込む．
# 
#     def forward(self, x):
#         x = x + self.pe[:, :x.size(1)].detach()
#         return x
# ```

# ```{note}
# とっても簡単なコードでアテンションが実装できますね．
# ```

# アテンションの計算をこの問題に利用することで正解が導き出されたことから，アテンションを利用するとしっかり入力配列データの文脈情報を下流のネットワークに伝えることが出来たと思いますが，もうひとつ比較のための実験を行います．以下ではアテンションを利用せず，つまり，単なるポイントワイズ MLP を利用したときにこの問題を解くことができるかどうかを調べます．できないはずです．以下のようなコードを書いて実行します．

# In[ ]:


#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
torch.manual_seed(0)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # データの生成．
    trainx = [
        torch.tensor([1,2,3,4,5,6,7,8,9,1], dtype=torch.long),
        torch.tensor([3,9,3,4,7], dtype=torch.long),
        torch.tensor([7,5,8], dtype=torch.long),
        torch.tensor([1,5,8], dtype=torch.long),
        torch.tensor([3,9,3,4,6], dtype=torch.long),
        torch.tensor([7,3,4,1], dtype=torch.long),
        torch.tensor([1,3], dtype=torch.long),
        torch.tensor([3,9,3,4,1], dtype=torch.long),
        torch.tensor([7,5,5,7,7,5], dtype=torch.long)
    ]
    traint = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int64)
    vocab_size = max(max(seq) for seq in trainx) + 1
    trainx = pad_sequence(trainx, batch_first=True, padding_value=0).to(device)
    traint = torch.tensor(traint, dtype=torch.long).to(device)

    # ハイパーパラメータ．
    embed_size = 16
    attention_unit_size = 32
    output_size = 3
    minibatch_size = 3

    # モデルの生成．
    model = Network(vocab_size, embed_size, attention_unit_size, output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    dataset = TensorDataset(trainx, traint)
    dataloader = DataLoader(dataset, batch_size=minibatch_size, shuffle=True)

    # 学習ループ．
    for epoch in range(1000 + 1):
        totalcost, totalacc = 0, 0
        for tx, tt in dataloader:
            optimizer.zero_grad()
            ty = model(tx)
            cost = criterion(ty, tt)
            totalcost += cost
            tp = ty.argmax(dim=1)
            totalacc += (tp == tt).sum().item() / len(trainx)
            cost.backward()
            optimizer.step()
        totalcost /= len(dataloader)
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Cost {totalcost:.4f}, Acc {totalacc:.4f}")
            with torch.no_grad():
                model.eval()
                trainp = model(trainx)
                print("Prediction:", torch.argmax(trainp, axis=1))

class Network(nn.Module):
    def __init__(self, vocab_size, embed_size, attention_unit_size, output_size):
        super(Network, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size, padding_idx=0)
        self.positional_encoder = PositionalEncoder(embed_size)
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=1, batch_first=True) # アテンションの計算．num_heads=1のときはシンプルなアテンションの計算．
        self.fc1 = nn.Linear(embed_size, attention_unit_size)
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(attention_unit_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoder(x) # 位置情報を入力データに追加．
#        x, _ = self.attention(x, x, x)
        x = self.fc1(x[:, 0, :]) # 配列の最初のタイムステップだけを以降の計算に利用．
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class PositionalEncoder(nn.Module):
    def __init__(self, input_size, max_seq_len=500):
        super(PositionalEncoder, self).__init__()
        self.input_size = input_size
        pe = torch.zeros(max_seq_len, input_size) # 位置エンコーディング用のゼロ行列の生成．
        for pos in range(max_seq_len):
            for i in range(0, input_size, 2): # サインとコサインの計算のため2個ごとにスキップ．
                pos_tensor = torch.tensor(pos, dtype=torch.float32)
                base = torch.tensor(10000.0, dtype=torch.float32) # 位置エンコードの計算．
                denominator = torch.pow(base, (2 * i) / input_size) # 位置エンコードの計算．
                pe[pos, i] = torch.sin(pos_tensor / denominator) # 位置エンコードの計算．
                pe[pos, i + 1] = torch.cos(pos_tensor / denominator) # 位置エンコードの計算．
        pe = pe.unsqueeze(0) # unsqueeze()は指定した位置に次元数を追加するもの．これにてミニバッジのための次元を追加．
        self.register_buffer("pe", pe) # register_buffer()はモデルの一部として保存されるべきデータを登録するためのもの．これで位置エンコードをモデルに組み込む．

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].detach()
        return x

if __name__ == '__main__':
    main()


# ```{note}
# アテンションの計算をコメントアウトしただけです．
# ```

# 実行した結果，以下のような出力が得られました．学習がうまく進んでいません．3 個の値の分類問題なのでランダムに予測すると大体正確度は 0.33 になると思いますが，そのようになっています．
# 
# ```shell
# .
# .
# .
# Epoch 1000: Cost 1.0992, Acc 0.2222
# Prediction: tensor([0, 1, 0, 0, 1, 0, 0, 1, 0], device='cuda:0')
# ```

# ```{note}
# このように構成要素の重要な部分をあえて除いて計算機実験をし，その重要性を確認する方法をアブレーションスタディと言います．
# ```

# ### トランスフォーマークラス

# PyTorch にはトランスフォーマーのクラスも実装されています．ここではエンコーダーのクラスを利用しますが，トランスフォーマー全体も利用することはできます．前項のコードとほぼ同様の内容のコードは以下のように書きます．

# In[ ]:


#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
torch.manual_seed(0)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # データの生成．
    trainx = [
        torch.tensor([1,2,3,4,5,6,7,8,9,1], dtype=torch.long),
        torch.tensor([3,9,3,4,7], dtype=torch.long),
        torch.tensor([7,5,8], dtype=torch.long),
        torch.tensor([1,5,8], dtype=torch.long),
        torch.tensor([3,9,3,4,6], dtype=torch.long),
        torch.tensor([7,3,4,1], dtype=torch.long),
        torch.tensor([1,3], dtype=torch.long),
        torch.tensor([3,9,3,4,1], dtype=torch.long),
        torch.tensor([7,5,5,7,7,5], dtype=torch.long)
    ]
    traint = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int64)
    vocab_size = max(max(seq) for seq in trainx) + 1
    trainx = pad_sequence(trainx, batch_first=True, padding_value=0).to(device)
    traint = torch.tensor(traint, dtype=torch.long).to(device)

    # ハイパーパラメータ．
    embed_size = 16
    attention_unit_size = 32
    output_size = 3
    minibatch_size = 3
    attention_head = 2
    attention_layer = 1

    # モデルの生成．
    model = Network(vocab_size, embed_size, attention_head, attention_layer, output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    dataset = TensorDataset(trainx, traint)
    dataloader = DataLoader(dataset, batch_size=minibatch_size, shuffle=True)

    # 学習ループ．
    for epoch in range(1000 + 1):
        totalcost, totalacc = 0, 0
        for tx, tt in dataloader:
            optimizer.zero_grad()
            ty = model(tx)
            cost = criterion(ty, tt)
            totalcost += cost
            tp = ty.argmax(dim=1)
            totalacc += (tp == tt).sum().item() / len(trainx)
            cost.backward()
            optimizer.step()
        totalcost /= len(dataloader)
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Cost {totalcost:.4f}, Acc {totalacc:.4f}")
            with torch.no_grad():
                model.eval()
                trainp = model(trainx)
                print("Prediction:", torch.argmax(trainp, axis=1))

class Network(nn.Module):
    def __init__(self, vocab_size, embed_size, nhead, num_encoder_layers, output_size):
        super(Network, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.positional_encoder = PositionalEncoder(embed_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=nhead, dim_feedforward=embed_size, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.fc = nn.Linear(embed_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoder(x)
        x = self.transformer_encoder(x)
        x = x[:,0,:]
        x = self.fc(x)
        return x

class PositionalEncoder(nn.Module):
    def __init__(self, input_size, max_seq_len=500):
        super(PositionalEncoder, self).__init__()
        self.input_size = input_size
        pe = torch.zeros(max_seq_len, input_size) # 位置エンコーディング用のゼロ行列の生成．
        for pos in range(max_seq_len):
            for i in range(0, input_size, 2): # サインとコサインの計算のため2個ごとにスキップ．
                pos_tensor = torch.tensor(pos, dtype=torch.float32)
                base = torch.tensor(10000.0, dtype=torch.float32) # 位置エンコードの計算．
                denominator = torch.pow(base, (2 * i) / input_size) # 位置エンコードの計算．
                pe[pos, i] = torch.sin(pos_tensor / denominator) # 位置エンコードの計算．
                pe[pos, i + 1] = torch.cos(pos_tensor / denominator) # 位置エンコードの計算．
        pe = pe.unsqueeze(0) # unsqueeze()は指定した位置に次元数を追加するもの．これにてミニバッジのための次元を追加．
        self.register_buffer("pe", pe) # register_buffer()はモデルの一部として保存されるべきデータを登録するためのもの．これで位置エンコードをモデルに組み込む．

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].detach()
        return x

if __name__ == '__main__':
    main()


# モデルを生成する以下の部分が前項のコードと異なります．マルチヘッドアテンションのヘッド数とアテンション層の個数を指定する必要があります．
# 
# ```python
#     model = Network(vocab_size, embed_size, attention_head, attention_layer, output_size).to(device)
# ```

# 実際のクラスは以下のように書きます．位置エンコードは以前のコード同様に利用する必要があります．
# 
# ```python
# class Network(nn.Module):
#     def __init__(self, vocab_size, embed_size, nhead, num_encoder_layers, output_size):
#         super(Network, self).__init__()
#         self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
#         self.positional_encoder = PositionalEncoder(embed_size)
#         encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=nhead, dim_feedforward=embed_size, batch_first=True)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
#         self.fc = nn.Linear(embed_size, output_size)
# 
#     def forward(self, x):
#         x = self.embedding(x)
#         x = self.positional_encoder(x)
#         x = self.transformer_encoder(x)
#         x = x[:,0,:]
#         x = self.fc(x)
#         return x
# ```

# ```{note}
# 終わりです．
# ```
