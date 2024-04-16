#!/usr/bin/env python
# coding: utf-8

# # 再帰型ニューラルネットワーク

# ## 基本的な事柄

# 配列データを処理することが得意なニューラルネットワークである再帰型ニューラルネットワーク（recurrent neural network（RNN））とそれに関する基本的な事柄をまとめます．

# ### RNN とは

# RNN とは配列データを処理することが得意なニューラルネットワークです．配列データの各要素（タイムステップと呼びます）を逐次的に処理して，各タイムステップに対応する何らかの出力をすることができます．配列データを入力として配列データを出力させることができるし，配列データを入力にしてスカラを出力させることもできます．

# ```{note}
# 配列データを処理させて配列データのある一部を利用してスカラを出力することができます．
# ```

# 世の中には配列データがたくさんあります．例えば，自然言語，生物学的文字列（核酸とかアミノ酸配列），音楽，人の動き，株価とかです．RNN を利用するとこのようなデータを上手に扱うことができます．

# ```{note}
# 上手に扱うことができるのであって，配列処理に必要不可欠な構造ではないです．
# ```

# RNN を用いて何らかのデータを解析する際には，普通，超短期記憶（Long Short-term Memory（LSTM））とケート付き再帰型ユニット（Gated Recurrent Unit（GRU））を利用すれば十分でしょう．TensorFlow や PyTorch には標準的に実装されているので簡単に利用することができます．

# ```{note}
# その他にも単純 RNN が実装されていますが，これの性能は多くの場合良くありません．
# ```

# LSTM は最も簡単には以下のように実行することができます．ここでは，`[7, 5, 8]` と `[3, 9, 3]` というふたつの 3 個の要素からなる配列データを入力にして 4 個の要素からなる配列データを LSTM で出力させています．

# In[ ]:


#!/usr/bin/env python3
import torch
import torch.nn as nn
import numpy as np
torch.manual_seed(0)

def main():
    # GPUの使用の設定．
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # データセットの生成．
    tx = np.asarray([[[7, 5, 8],[3, 9, 3]]], dtype=np.float32)
    tx = torch.tensor(tx).to(device)
    print(tx)
    # モデルの定義．
    rnn = nn.LSTM(input_size=3, hidden_size=4, batch_first=True).to(device)
    output, (hidden, cell) = rnn(tx)
    print(output)

if __name__ == "__main__":
    main()


# 以下の部分が LSTM を利用するための記述です．`batch_first=True` は、PyTorchにおいてRNNの層を使用する際に，入力データの形状が `[batch_size, seq_len, features]` の順序であることを指定するオプションです．このオプションが指定されていない場合（デフォルトは `batch_first=False`），入力データは `[seq_len, batch_size, features]` の形状を持つと期待されます．
# ```
# rnn = nn.LSTM(input_size=3, hidden_size=4, batch_first=True).to(device)
# ```

# この LSTM で行った計算を GRU に行わせるようにすることはとても簡単です．以下のように `LSTM` の部分を `GRU` に書き換えて，出力を受け取る変数を変えるだけです．

# In[ ]:


#!/usr/bin/env python3
import torch
import torch.nn as nn
import numpy as np
torch.manual_seed(0)

def main():
    # GPUの使用の設定．
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # データセットの生成．
    tx = np.asarray([[[7, 5, 8],[3, 9, 3]]], dtype=np.float32)
    tx = torch.tensor(tx).to(device)
    print(tx)
    # モデルの定義．
    rnn = nn.GRU(input_size=3, hidden_size=4, batch_first=True).to(device)
    output, hidden = rnn(tx)
    print(output)

if __name__ == "__main__":
    main()


# ```{note}
# `LSTM` と `GRU` の内部構造は少し異なります．`GRU` はセル状態を持たないので出力値は2個だけです．
# ```

# ### 文字エンコード

# 人工知能に何かの情報を処理させる際には，機械がそれを理解できるように何らかの方法でデータを数字に変換しなければなりませんが，これをエンコードと言います．例えば，自然言語に含まれる単語を，処理したいデータセット中の登場順に数字に変換する，みたいなことです．機械に処理させるためには自然言語では都合が悪く，数字データが都合が良いからです．そのような単語を数字に割り振って人工知能に入力しても，もちろん計算自体は可能です．しかし，この整数エンコーディングにはふたつの欠点があると，TensorFlow の公式ウェブサイトに書いてありました．
# 
# 
# *   整数エンコーディングは単語間のいかなる関係性を含まない．
# *   整数エンコーディングは人工知能にとっては解釈が難しい．例えば，線形分類器はそれぞれの特徴量について単一の重みしか学習しないため，ふたつの単語が似ていることとそれらのエンコーディングが似ていることの間には何の関係もなく，特徴と重みの組み合わせに意味がない．
# 
# ただし，最初のひとつは学習全体を通して考えると人工知能のパラメータの方で担えば解決できるし，ふたつ目も非線形な関数を近似できる人工知能を使えば解決できるため，回避できない問題とは思えません．
# 
# これとは別のエンコード方法として，ワンホットエンコーディングがあります．ワンホットエンコーディングはワンホットベクトルに単語を対応させます．ワンホットベクトルとは，ベクトルのある要素が1で他の全要素が0であるベクトルです．以下のような感じです．
# 
# <img src="https://github.com/yamada-kd/introduction-to-artificial-intelligence/blob/main/image/oneHotEmbedding.svg?raw=1" width="100%" />
# 
# 
# この方法は非効率的です．ワンホットベクトルはとても疎です．ボキャブラリに10000個の単語がある場合，各ベクトルはその要素の99.99%が0からなります．学習を効率よく進めるためには多くの重みパラメータを設定しなければならないことになるでしょう．また，入力ベクトルが場合によってはものすごく長くなるでしょう．
# 
# 現在のところ最も良い方法と考えられるのは単語埋め込み（word embedding）です．単語埋め込みを利用すると似たような単語が似たようなベクトルへとエンコードされます（学習の過程で）．埋め込みは浮動小数点数で行い，密なベクトルができます．以下のような感じです．
# 
# <img src="https://github.com/yamada-kd/introduction-to-artificial-intelligence/blob/main/image/floatEmbedding.svg?raw=1" width="100%" />
# 

# ### 入出力データ形式

# 次の節では，実際の LSTM の実装方法を紹介します．RNN を PyTorch（や TensorFlow）で利用する場合には，その入力データと出力データで少しずつ書き方が変わってしまい，これをしっかり書かないとエラーが起こってしまいます．また，RNN は入力ベクトルの最初から順々にデータを処理する単方向でのデータ処理と，最後から最初に遡りながらデータを処理する逆方向の処理を順方向の処理と同時に行う双方向のデータ処理があり，これを利用するかどうかでも出力が変化します．
# 
# RNN を活用したい問題としては0か1という分類をする問題があると思います．また，これとは異なり，配列を入力に配列を出力させたいときもあるはずです．さらに，分類ではなく回帰をしたい場合もあるはずです．単語（整数）の多次元のベクトルへの埋め込み（embedding）が必要な場合もあります．例えば `1` という文字は人工知能に入力するためには `[0.2, 0.1, 0.3, 0.4]` みたいなベクトルに変換しなければなりません．何次元のベクトルでも良いです．また，`[1, 0, 0, 0]` みたいなワンホットエンコーディングと呼ばれる埋め込み方法でも良いです．これらの種類をまとめると以下の表のようになります．実装方法がそれぞれにおいてほんの少しずつ異なります．ちょっと厄介ですね．
# 
# 入力 | 出力 | 問題 | 埋め込み | 向き
# :---: | :---: | :---: | :---: | :---:
# <font color="Crimson">配列</font> | <font color="Crimson">スカラ</font> | <font color="Crimson">分類</font> | <font color="Crimson">不要</font> | <font color="Crimson">単方向</font>
# 配列 | スカラ | 分類 | 不要 | 双方向
# 配列 | スカラ | 分類 | 必要 | 単方向
# 配列 | スカラ | 分類 | 必要 | 双方向
# 配列 | スカラ | 回帰 | 不要 | 単方向
# 配列 | スカラ | 回帰 | 不要 | 双方向
# 配列 | スカラ | 回帰 | 必要 | 単方向
# 配列 | スカラ | 回帰 | 必要 | 双方向
# 配列 | 配列 | 分類 | 不要 | 単方向
# 配列 | 配列 | 分類 | 不要 | 双方向
# 配列 | 配列 | 分類 | 必要 | 単方向
# 配列 | 配列 | 分類 | 必要 | 双方向
# 配列 | 配列 | 回帰 | 不要 | 単方向
# 配列 | 配列 | 回帰 | 不要 | 双方向
# 配列 | 配列 | 回帰 | 必要 | 単方向
# <font color="Crimson">配列</font> | <font color="Crimson">配列</font> | <font color="Crimson">回帰</font> | <font color="Crimson">必要</font>  | <font color="Crimson">双方向</font>
# 
# ここでは，「単方向で入力ベクトルが配列で出力ベクトルがスカラ（1要素からなるベクトル）で問題が分類で文字の埋め込みが不要なパターン」と「双方向で入力ベクトルと出力ベクトルがともに配列で問題が回帰で埋め込みが必要なパターン」のふたつの実装を紹介します．
# 

# ```{note}
# 全部紹介しても良いのですけど紙面の都合上避けます．
# ```

# ## RNN の実装

# この節では RNN の使い方を紹介します．上述のように RNN を利用する際にはいくつかの書き方がありますが，そのうちのいくつかを紹介します．

# ### LSTM とは

# 次に，簡単なデータを生成して RNN の使い方を紹介します．ここでは，LSTM を利用します．LSTM は様々な長さからなる配列情報を入力データとして受け取り，配列情報を出力することができるニューラルネットワークです．「長さの異なる入力ベクトルを同じ構造で扱うことができる」ことと「配列情報を逐次的に出力することができる」という点において MLP とは異なります．LSTM のアルゴリズムは以下の式で定義されます．
# 
# $
# \mathbf{v}_1=\sigma(\mathbf{W}_{1a}\mathbf{u}_t+\mathbf{W}_{1b}\mathbf{h}_{t-1}+\mathbf{b}_{1}),
# $
# 
# $
# \mathbf{v}_2=\sigma(\mathbf{W}_{2a}\mathbf{u}_t+\mathbf{W}_{2b}\mathbf{h}_{t-1}+\mathbf{b}_{2}),
# $
# 
# $
# \mathbf{v}_3=\sigma(\mathbf{W}_{3a}\mathbf{u}_t+\mathbf{W}_{3b}\mathbf{h}_{t-1}+\mathbf{b}_{3}),
# $
# 
# $
# \mathbf{v}_4=\tau(\mathbf{W}_{4a}\mathbf{u}_t+\mathbf{W}_{4b}\mathbf{h}_{t-1}+\mathbf{b}_{4}),
# $
# 
# $
# \mathbf{s}_t=\mathbf{v}_1\odot\mathbf{v}_4+\mathbf{v}_2\odot\mathbf{s}_{t-1},
# $
# 
# $
# \mathbf{h}_t=\mathbf{v}_3\odot\tau(\mathbf{s}_t)．
# $
# 
# ここで，太字の小文字はベクトル（1 列の行列）で，太字の大文字は行列です．$t$ は入力配列の要素の位置を示します．時系列なら $t$ 時間目の要素です．文字列なら $t$ 番目の要素です．例えば，ピリオドを含めて 5 文字の長さからなる `I have a pen .` のような文字列において，$t=1$ の値は `I` で，$t=5$ の値は `.` です．$\mathbf{W}$ は LSTM で学習すべき重みパラメータです．$\mathbf{b}$ はバイアスパラメータです．$\mathbf{\sigma}$ と $\mathbf{\tau}$ はそれぞれシグモイド関数とハイパボリックタンジェント関数です．それぞれの値域は，[0, 1] および [-1, 1] です．$\mathbf{h}_t$ は時間 $t$ における出力ベクトルです．$\mathbf{u}_t$ は時間 $t$ における入力ベクトルです．$\odot$ はアダマール積を示し，$+$ は行列の足し算を示します．$\mathbf{W}\mathbf{u}$ のような変数が結合している部分はその変数間で行列の掛け算を行う表記です．
# 
# LSTM では入力ベクトルに対して最初の類似した 4 つの式にて中間ベクトル $\mathbf{v}$ を計算します．シグモイド関数を含む式の最小値は 0 で最大値は 1 です．これに対してハイパボリックタンジェント関数を含む 4 番目の式の最小値は-1で最大値は1です．この 4 番目の式は入力された値に LSTM のパラメータを作用させ -1 から 1 の値に規格化する効果を持ちます．1 から 3 番目の式はゲート構造と呼ばれる LSTM の仕組みです．この 3 つの式の出力値を 5 番目と 6 番目で $\mathbf{v}_4$ に作用させます．
# 
# 例えば，$\mathbf{v}_1$ は入力ゲートと呼ばれるゲートです．この値が $\mathbf{0}$ であった場合，入力値である $\mathbf{v}_4$ と $\mathbf{v}_1$ の（アダマール）積である $\mathbf{v}_1\odot\mathbf{v}_4$ の値（5 番目の式の第 1 項）は $\mathbf{0}$ であり，その入力値は以降の計算に影響しなくなります．これはゲートが閉じているという状況です．また，2番目の式は忘却ゲートです．$\mathbf{s}_{t-1}$ は現在（$t$）のひとつ前の時間における情報を保持した以前の記憶を保持したベクトルです．これに対してのゲートとしての機能を持つことから忘却ゲートと呼ばれます．また，3番目の式は出力ゲートです．$\mathbf{h}_t$ は $t$ における出力ベクトルですが，これを計算するために用いられるため出力ゲートと呼ばれます．LSTM はこのようなゲート構造を有することで高性能化を達成した RNN であると考えられています．元々は動物の脳の機能をモデルにして開発されました．
# 
# この出力の値 $\mathbf{h}$ は $\mathbf{s}$ と同様に保存され，次の時間（timestep）での計算に用いられます．このように，LSTM は以前の情報（ひとつ前の timestep の情報）を使って出力をする人工知能です．ひとつ前の情報にはさらにひとつ前の情報が記憶されており，またそのひとつ前の情報にはさらにひとつ前の情報が記憶されています．よって，LSTM ではどんな長い配列情報であっても（パラメータサイズが十分なら），すべての文脈に関わる情報を記憶することが可能です．

# ```{note}
# 決して MLP が配列情報を扱えないというわけではありません．
# ```

# ### 実装方法①

# ここでは，「単方向で入力ベクトルが配列で出力ベクトルがスカラ（1要素からなるベクトル）で問題が分類で文字の埋め込みが不要なパターン」の実装方法を紹介します．ここでは以下のようなデータを生成して利用します．このデータセットは3つのインスタンスからなります．最初のインスタンスは4文脈からなる配列です．入力ベクトルの各 timestep は2要素からなるベクトルで構成されています．$t=1$ のときの値は `[1.1, 0.1]` で，$t=2$ のときの値は `[2.2, 0.3]` で，$t=3$ のときの値は `[3.0, 0.3]` で，$t=4$ のときの値は `[4.0, 1.0]` です．これに紐づいているターゲットベクトルは `[1]` です．これは分類問題なのでこのターゲットベクトルの `0` は `1` という値より `1` ほど小さい値を意味しているのでなく，単にクラスを意味する数字です．
# 
# 入力ベクトル | ターゲットベクトル
# :--- | :---:
# [ [1.1, 0.1], [2.2, 0.3], [3.0, 0.3], [4.0, 1.0] ] | [ 1 ]
# [ [2.0 ,0.9], [0.1, 0.8], [3.0, 0.7], [4.0, 0.1], [1.0, 0.3] ] | [ 2 ]
# [ [2.0, 1.0], [3.0, 0.6], [4.0, 0.6] ] | [ 0 ]
# 
# プログラムは以下のように書きます．

# In[ ]:


#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
torch.manual_seed(0)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # データセットの生成．
    tx = [
        torch.tensor([[1.1, 0.1], [2.2, 0.3], [3.0, 0.3], [4.0, 1.0]]),
        torch.tensor([[2.0, 0.9], [0.1, 0.8], [3.0, 0.7], [4.0, 0.1], [1.0, 0.3]]),
        torch.tensor([[2.0, 1.0], [3.0, 0.6], [4.0, 0.6]])
    ]
    tx = torch.nn.utils.rnn.pad_sequence(tx, batch_first=True, padding_value=0.0).to(device)
    tt = torch.tensor([1, 2, 0], dtype=torch.long).to(device)

    # ネットワークの定義
    model = Network().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    dataset = TensorDataset(tx, tt)
    loader = DataLoader(dataset, batch_size=3) # 全データを一括で読み込みたいためミニバッチサイズは3に設定．

    # 学習前の人工知能がどのような出力をしているのかを確認．
    with torch.no_grad():
        ty = model(tx)
        print("Output vector:", ty)
        print("Target vector:", tt)

    # 学習ループ．
    for epoch in range(1, 1001):
        for data, target in loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        if epoch % 50 == 0:
            print(f'Epoch {epoch}: Training Loss= {loss.item()}')

    # 学習後の人工知能がどのような出力をしているのかを確認．
    with torch.no_grad():
        ty = model(tx)
        print("Output vector:", ty)
        print("Target vector:", tt)

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=50, batch_first=True)
        self.fc = nn.Linear(50, 3)

    def forward(self, tx):
        lstm_out, _ = self.lstm(tx)
        ty = self.fc(lstm_out[:, -1, :])  # 最後のタイムステップのみを使用．
        return ty

if __name__ == "__main__":
    main()


# 以下の部分はデータセットを生成するための記述です．`tx` が入力ベクトルを格納するための変数です．入力ベクトルは二次元配列なので，`tx` は三次元配列の構造をとります．
# ```python
#     # データセットの生成．
#     tx = [
#         torch.tensor([[1.1, 0.1], [2.2, 0.3], [3.0, 0.3], [4.0, 1.0]]),
#         torch.tensor([[2.0, 0.9], [0.1, 0.8], [3.0, 0.7], [4.0, 0.1], [1.0, 0.3]]),
#         torch.tensor([[2.0, 1.0], [3.0, 0.6], [4.0, 0.6]])
#     ]
#     tx = pad_sequence(tx, batch_first=True, padding_value=0.0).to(device)
#     tt = torch.tensor([1, 2, 0], dtype=torch.long).to(device)
# ```
# `tx = torch.nn.utils.rnn.pad_sequence(tx, batch_first=True, padding_value=0.0).to(device)` はゼロパディングのための記述です．これによって以下のようにデータは変換されます．
# 
# 入力ベクトル | ターゲットベクトル
# :--- | :---:
# [ [1.1, 0.1], [2.2, 0.3], [3.0, 0.3], [4.0, 1.0], [0.0, 0.0] ] | [ 1 ]
# [ [2.0 ,0.9], [0.1, 0.8], [3.0, 0.7], [4.0, 0.1], [1.0, 0.3] ] | [ 2 ]
# [ [2.0, 1.0], [3.0, 0.6], [4.0, 0.6], [0.0, 0.0], [0.0, 0.0] ] | [ 0 ]

# その他の部分は MLP のときと変わりません．ネットワークの定義は異なっており，以下ように行います．MLP の場合は `nn.Linear()` だけを用いました．これに対して LSTM を実装したい場合は `nn.LSTM()` を用います．ここでは，LSTM のユニットサイズを50にしています．また，`nn.Linear(50, 3)` とありますが，これは全結合層（最も基本的な層）でニューロンのサイズは3個であることを意味しています．また，この層は出力層です．以前の章の MNIST を扱った MLP が入力層784，中間層500，出力層10のサイズでしたが，これを `(784, 500, 10)`と表現するとします．これに対して，このネットワークの1 timestep では `(2, 50, 3)` という MLP の計算がされます．この `2` というのは各タイムステップのベクトルの要素数です．最後の `3` というのはターゲットベクトルの要素数です．これが LSTM の仕組みによって，timestep 分（5回）繰り返されます．この場合，出力は配列でないため，最後の `(2, 50, 3)` の計算の出力のみを最終的な予測に用います．
# 
# ```python
# class Network(nn.Module):
#     def __init__(self):
#         super(Network, self).__init__()
#         self.lstm = nn.LSTM(input_size=2, hidden_size=50, batch_first=True)
#         self.fc = nn.Linear(50, 3)
# 
#     def forward(self, tx):
#         lstm_out, _ = self.lstm(tx)
#         ty = self.fc(lstm_out[:, -1, :])  # 最後のタイムステップのみを使用．
#         return ty
# ```

# 結果を確認すると，しっかりコストが下がっており，同時に正確度も上昇している様子が判ります．また，未学習の際には正解を導けていなかったところ，学習済みのモデルでは正解をしっかり出力できていることが確認できます．

# ### 実装方法②

# 次に，「双方向で入力ベクトルと出力ベクトルがともに配列で問題が回帰で埋め込みが必要なパターン」の実装方法を紹介します．ここでは以下のようなデータを生成して利用します．このデータセットは3つのインスタンスからなります．最初のインスタンスは3文脈からなる配列です．入力ベクトルの各 timestep は整数です．ここでの7という数字は5よりも2ほど大きい値を意味しているのはなく，単なるダミー変数です．ターゲットベクトルの各 timestep は2要素からなるベクトルで構成されています．最初のインスタンスの $t=1$ のときの値は `[6.2, 1.1]` です．この6.2は1.1よりも5.1ほど大きい数値です．このインスタンスでは，最初に入力された7というダミー変数に対して，`[6.2, 1.1]` を予測しなければなりません．これは，分類問題ではなく回帰問題です．
# 
# 入力ベクトル | ターゲットベクトル
# :--- | :---
# [ 7, 5, 8 ] | [ [6.2, 1.1], [3.5, 2.1], [2.0, 1.1] ]
# [ 3, 9, 3, 4, 6 ] | [ [4.5, 3.8], [4.1, 4.9], [3.4, 4.6], [2.7, 1.7], [2.1, 2.5] ]
# [ 2, 3, 4, 1 ] | [ [1.2, 1.0], [4.4, 3.3], [3.1, 2.8], [2.7, 1.6] ]
# 
# プログラムは以下のように書きます．

# In[ ]:


#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
torch.manual_seed(0)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # データセットの生成．
    tx = [[7, 5, 8], [3, 9, 3, 4, 6], [2, 3, 4, 1]]
    tt = [[[6.2, 1.1], [3.5, 2.1], [2.0, 1.1]], [[4.5, 3.8], [4.1, 4.9], [3.4, 4.6], [2.7, 1.7], [2.1, 2.5]], [[1.2, 1.0], [4.4, 3.3], [3.1, 2.8], [2.7, 1.6]]]

    tx = torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in tx], batch_first=True, padding_value=0).to(device)
    tt = torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in tt], batch_first=True, padding_value=-1).to(device)

    model = Network().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    # 学習前の出力を確認．
    with torch.no_grad():
        ty = model(tx)
        print("Output vector:", ty)
        print("Target vector:", tt)

    # 学習ループ．
    for epoch in range(1, 1001):
        optimizer.zero_grad()
        output = model(tx)
        loss = criterion(output, tt)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print("Epoch={:5d} Cost={:7.4f}".format(epoch, loss.item()))

    # 学習後の出力を確認．
    with torch.no_grad():
        ty = model(tx)
        print("Output vector:", ty)
        print("Target vector:", tt)

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.embed = nn.Embedding(num_embeddings=10, embedding_dim=3, padding_idx=0)
        self.lstm = nn.LSTM(input_size=3, hidden_size=50, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(in_features=100, out_features=2)

    def forward(self, tx):
        ty = self.embed(tx)
        ty, _ = self.lstm(ty)
        ty = self.fc(ty)
        return ty

if __name__ == "__main__":
    main()


# このプログラムではネットワークを定義するクラス内で `nn.Embedding()` を利用しています．これは `7` とか `5` とか `8` というようなダミー変数をそれぞれ，`[5.1234, 0.4516, 1.4631]` とか `[1.5462, 0.4641, 0.9798]` とか `[3.7486, 0.7672, 4.423]` みたいなベクトルに変換するための関数です．ダミー変数を何らかのベクトル空間に埋め込むという作業をします．
# 

# 次に，以下の部分ですが，ダミー変数の場合は0でパディングを行いました．それに対してこのターゲットベクトルに関しては，`-1` というターゲットベクトルに出現しそうにない値でパディングを行っています．
# ```python
# tt = torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in tt], batch_first=True, padding_value=-1).to(device)
# ```

# 実際のネットワークは以下で定義しています．`nn.Embedding(num_embeddings=9+1, embedding_dim=3, padding_idx=0)` における `9+1` は入力ベクトルのダミー変数の種類が `1` から `9` の9種類あることに加えて，配列をパディングするために `0` を利用するためです．この `0` から `9` の10種類の値を3要素からなるベクトルデータに変換する作業をこれで行います．`nn.LSTM(input_size=3, hidden_size=50, batch_first=True, bidirectional=True)` の部分は前述の例と異なります．`bidirectional=True` は RNN を双方向で計算させるためのものです．LSTM 自体は50個のニューロンで定義されていますが，双方向なので正方向の出力と負方向の出力が連結された100の大きさのベクトルが返ります．最後の `nn.Linear()` は全結合層であり，出力データの各時刻における値が2要素からなるベクトルなのでニューロンは2個に設定します．プログラムを実行すると学習済みの人工知能の出力は教師ベクトルと類似していることが確認できます．
# 
# ```python
# class Network(nn.Module):
#     def __init__(self):
#         super(Network, self).__init__()
#         self.embed = nn.Embedding(num_embeddings=9+1, embedding_dim=3, padding_idx=0)
#         self.lstm = nn.LSTM(input_size=3, hidden_size=50, batch_first=True, bidirectional=True)
#         self.fc = nn.Linear(in_features=100, out_features=2)
# 
#     def forward(self, tx):
#         ty = self.embed(tx)
#         ty, _ = self.lstm(ty)
#         ty = self.fc(ty)
#         return ty
# ```

# 以上の基本的な書き方を習得したら，それらに合わせて解析したいデータを整形することで RNN による解析ができるようになります．
# 

# ```{note}
# 終わりです．
# ```
