#!/usr/bin/env python
# coding: utf-8

# # 多層パーセプトロン

# ## 扱うデータの紹介

# このコンテンツでは最も基本的な深層学習アルゴリズムである多層パーセプトロンを実装する方法を紹介します．多層パーセプトロンは英語では multilayer perceptron（MLP）と言います．ニューラルネットワークの一種です．層という概念があり，この層を幾重にも重ねることで深層ニューラルネットワークを構築することができます．MLP を実装するためにとても有名なデータセットを利用しますが，この節ではそのデータセットの紹介をします．

# ### MNIST について

# MLP に処理させるデータセットとして，機械学習界隈で最も有名なデータセットである MNIST（Mixed National Institute of Standards and Technology database）を解析対象に用います．MNIST は縦横28ピクセル，合計784ピクセルよりなる画像データです．画像には手書きの一桁の数字（0から9）が含まれています．公式ウェブサイトでは，学習データセット6万個とテストデータセット1万個，全部で7万個の画像からなるデータセットが無償で提供されています．

# ```{note}
# MNIST はエムニストと読みます．
# ```

# ### ダウンロードと可視化

# 公式サイトよりダウンロードしてきても良いのですが，PyTorch がダウンロードするためのユーティリティを準備してくれているため，それを用います．MNIST は合計7万インスタンスからなるデータセットです．8行目は学習データセット，9行目はテストデータセットのための記述です．

# In[ ]:


#!/usr/bin/env python3
import torch
from torchvision import datasets, transforms

def main():
    # MNISTデータセットの読み込み．
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    print("The number of instances in the learning dataset:", len(train_dataset))
    print("The number of instances in the test dataset:", len(test_dataset))
    # 最初のインスタンスの情報を取得．
    first_train_image, first_train_target = train_dataset[0]
    print("The input vector of the first instance in the learning dataset:", first_train_image)
    print("Its shape:", first_train_image.shape)
    print("The target vector of the first instance in the learning dataset:", first_train_target)

if __name__ == "__main__":
    main()


# データを可視化します．可視化のために matplotlib というライブラリをインポートします．

# In[ ]:


#!/usr/bin/env python3
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def main():
    # MNISTデータセットの読み込み．
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # トレーニングデータセットから最初の画像を取得．
    first_train_image, first_train_target = train_dataset[0]

    # 画像を表示．
    plt.imshow(first_train_image.squeeze(), cmap="gray")
    plt.text(1, 2.5, int(first_train_target), fontsize=20, color="white")
    plt.show()

if __name__ == "__main__":
    main()


# ちなみに，このデータセットがダウンロードされている場所は `/content/data` です．以下のような BaSH のコマンドを打つことで確認することができます．

# In[ ]:


get_ipython().system(' ls /content/data')


# MNIST はこのような縦が28ピクセル，横が28ピクセルからなる手書き文字が書かれた（描かれた）画像です（0から9までの値）．それに対して，その手書き文字が0から9のどれなのかという正解データが紐づいています．この画像データを MLP に読み込ませ，それがどの数字なのかを当てるという課題に取り組みます．

# ## MLP の実装

# この節では MLP を実装します．MLP を実装することに加えて，どのように学習を進めるとより良い人工知能を構築できるのかについて紹介します．

# ### 簡単な MLP の実装

# 実際に MNIST を処理する MLP を実装する前に，とても簡単なデータを処理するための MLP を実装します．ここでは，以下のようなデータを利用します．これが学習セットです．ここでは MLP の実装の方法を紹介するだけなのでバリデーションセットもテストセットも使用しません．
# 
# 入力ベクトル | ターゲットベクトル
# :---: | :---:
# [ 1.1, 2.2, 3.0, 4.0 ] | [ 0 ]
# [ 2.0, 3.0, 4.0, 1.0 ] | [ 1 ]
# [ 2.0, 2.0, 3.0, 4.0 ] | [ 2 ]
# 
# すなわち，`[1.1, 2.2, 3.0, 4.0]` が人工知能へ入力されたら，`0` というクラスを返し，`[2.0, 3.0, 4.0, 1.0]` というベクトルが入力されたら `1` というクラスを返し，`[2.0, 2.0, 3.0, 4.0]` というベクトルが入力されたら `2` というクラスを返す人工知能を MLP で構築します．実際には以下のように書きます．

# In[ ]:


#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
torch.manual_seed(0)
np.random.seed(0)

def main():
    # GPUの使用の設定．
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # データセットの生成．
    tx = torch.tensor([[1.1, 2.2, 3.0, 4.0], [2.0, 3.0, 4.0, 1.0], [2.0, 2.0, 3.0, 4.0]], dtype=torch.float32).to(device)
    tt = torch.tensor([0, 1, 2], dtype=torch.long).to(device)

    # ネットワークの定義．
    model = Network().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # 学習ループ．
    for epoch in range(1, 3001):
        optimizer.zero_grad()
        ty = model(tx)
        traincost = criterion(ty, tt)
        prediction = ty.argmax(dim=1) # Accuracyを計算するために予測値を計算．
        trainacc = torch.tensor(torch.sum(prediction == tt).item() / len(tt)) # Accuracyを計算．
        traincost.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print("Epoch {:5d}: Training cost= {:.4f}, Training ACC= {:.4f}".format(epoch,traincost,trainacc))

    # 推論の例．
    tx1 = torch.tensor([[1.1, 2.2, 3.0, 4.0]], dtype=torch.float32).to(device)
    ty1 = model(tx1)
    print(ty1.cpu().detach().numpy())  # 結果をCPUに戻してnumpy配列に変換

    # 未知のデータに対する推論．
    tu = torch.tensor([[999, 888, 777, 666]], dtype=torch.float32).to(device)
    tp = model(tu)
    print(tp.cpu().detach().numpy())  # 結果をCPUに戻してnumpy配列に変換

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.d1 = nn.Linear(4, 10)  # 全結合層
        self.d2 = nn.Linear(10, 3)  # 出力層

    def forward(self, x):
        x = torch.relu(self.d1(x))
        x = torch.softmax(self.d2(x), dim=1)
        return x

if __name__ == "__main__":
    main()


# 上から説明します．以下の記述は GPU を利用するためのものです．もし GPU が利用できない環境だと CPU が利用されます．
# ```python
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Using device:", device)
# ```

# 以下では，上述のデータを生成しています．`tx` は入力ベクトル3つです．`tt` はそれに対応するターゲットベクトル（スカラ）3つです．
# ```python
#     # データセットの生成．
#     tx = torch.tensor([[1.1, 2.2, 3.0, 4.0], [2.0, 3.0, 4.0, 1.0], [2.0, 2.0, 3.0, 4.0]], dtype=torch.float32).to(device)
#     tt = torch.tensor([0, 1, 2], dtype=torch.long).to(device)
# ```

# 次に，以下のような記述があります．この記述によって未学習の人工知能を生成します．生成した人工知能は `model` です．
# ```python
#     model = Network().to(device)
# ```
# この未学習の人工知能を生成するための記述の本体はプログラムの最下層辺りにある以下の記述です．
# ```python
# class Network(nn.Module):
#     def __init__(self):
#         super(Network, self).__init__()
#         self.d1 = nn.Linear(4, 10)  # 全結合層
#         self.d2 = nn.Linear(10, 3)  # 出力層
# 
#     def forward(self, x):
#         x = torch.relu(self.d1(x))
#         x = torch.softmax(self.d2(x), dim=1)
#         return x
# ```
# ここに `nn.Linear(4, 10)` とありますが，これは10個のニューロンを持つ層を1個生成するための記述です．これによって生成される層の名前は `self.d1()` です．ここでは10個という値を設定していますが，これは100でも1万でも1兆でもなんでも良いです．解きたい課題にあわせて増やしたり減らしたりします．ここをうまく選ぶことでより良い人工知能を構築でき，腕の見せ所です．次に，`nn.Linear(10, 3)` という記述で3個のニューロンを持つ層を1個生成します．この3個という値は意味を持っています．入力するデータのクラスが0，1または2の3分類（クラス）であるからです．次の，`def forward(self,x):` という記述はこれ（`class Network()`）によって生成した人工知能を呼び出したときにどのような計算をさせるかを定義するものです．入力として `x` というベクトルが与えられたら，それに対して最初の層を適用し，次に，その出力に対して次の層を適用し，その値を出力する，と定義しています．構築した人工知能 `model` に対して `model.forward()` のような方法で呼び出すことができます．`torch.relu(self.d1(x))` という記述は活性化関数である ReLU を利用するためのものです．また，出力時の活性化関数にはソフトマックス関数，`torch.softmax()` を指定しています．ソフトマックス関数の出力ベクトルの要素を合計すると1になります．各要素の最小値は0です．よって出力結果を確率として解釈できます．
# 

# 次の以下の記述は，それぞれ，損失関数，正確度（ACC）を計算する関数，最急降下法の最適化法（パラメータの更新ルール）を定義するものです．これは，PyTorch ではこのように書くのだと覚えるものです．
# ```python
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters())
# ```

# ```{note}
# 実は，最終層でソフトマックス関数を使う必要はありません．`nn.CrossEntropyLoss()` は内部でソフトマックスを計算するので，これを損失関数として利用する場合は加えない方が良いです．このコードでは初学者用にあえて加えたに過ぎません．実際の利用では最終層のソフトマックス関数は外しましょう．
# ```

# 次に記述されている以下の部分は，実際の学習のループに関するものです．このループでデータを何度も何度も予測器（人工知能）に読ませ，そのパラメータを成長させます．この場合，3000回データを学習させます．また，学習100回毎に学習の状況を出力させます．
# ```python
#     # 学習ループ．
#     for epoch in range(1, 3001):
#         optimizer.zero_grad()
#         ty = model(tx)
#         traincost = criterion(ty, tt)
#         prediction = ty.argmax(dim=1) # Accuracyを計算するために予測値を計算．
#         trainacc = torch.tensor(torch.sum(prediction == tt).item() / len(tt)) # Accuracyを計算．
#         traincost.backward()
#         optimizer.step()
#         if epoch % 100 == 0:
#             print("Epoch {:5d}: Training cost= {:.4f}, Training ACC= {:.4f}".format(epoch,traincost,trainacc))
# ```
# この学習ループでは，最初に勾配の値を `optimizer.zero_grad()` にてゼロにします．PyTroch の仕様上，勾配を `.grad` に蓄積してしまうという性質があるからです．次の行では出力値を計算します．この出力値 `ty` と教師データ `tt` を `criterion()` にて比較することでコスト関数値 `traincost` を計算します．引き続き正確度を計算します．次の `traincost.backward()` はコスト関数から勾配を計算するためのものです．次の `optimizer.step()` にてニューラルネットワークのパラメータを更新します．

# 次の記述，以下の部分では学習がうまくいったのかを確認するために学習データのひとつを学習済みの人工知能に読ませて予測をさせています．この場合，最初のデータのターゲットベクトルは0なので0が出力されなければなりません．
# ```python
#     # 推論の例．
#     tx1 = torch.tensor([[1.1, 2.2, 3.0, 4.0]], dtype=torch.float32).to(device)
#     ty1 = model(tx1)
#     print(ty1.cpu().detach().numpy())  # 結果をCPUに戻してnumpy配列に変換
# ```
# 出力結果は以下のようになっているはずです．出力はソフトマックス関数なので各クラスの確率が表示されています．これを確認すると，最初のクラス（0）である確率が99%以上であると出力されています．よって，やはり人工知能は意図した通り成長したことが確認できます．
# ```
# [[9.9754351e-01 3.6117400e-04 2.0953205e-03]]
# ```
# 

# 次に，全く新たなデータを入力しています．
# ```python
#     # 未知のデータに対する推論．
#     tu = torch.tensor([[999, 888, 777, 666]], dtype=torch.float32).to(device)
#     tp = model(tu)
#     print(tp.cpu().detach().numpy())  # 結果をCPUに戻してnumpy配列に変換
# ```
# `[999,888,777,666]` というベクトルを入力したときにどのような出力がされるかということですが，この場合，以下のような出力がされています．このベクトルを入力したときの予測値は2であるとこの人工知能は予測したということです．
# ```
# [[0. 0. 1.]]
# ```
# 

# ### モジュールの挙動確認

# 以下では `nn.Linear()` の挙動を確認します．`nn.Linear()` はもちろんクラスの中でなければ使えない関数ではなく，`main()` の中でも呼び出して利用可能です．これで挙動を確認することでどのようにネットワークが構築されているか把握できるかもしれません．

# In[ ]:


#!/usr/bin/env python3
import torch
import torch.nn as nn
import numpy as np
torch.manual_seed(0)
np.random.seed(0)

def main():
    # データセットの生成
    tx = torch.tensor([[1.1, 2.2, 3.0, 4.0], [2.0, 3.0, 4.0, 1.0], [2.0, 2.0, 3.0, 4.0]], dtype=torch.float32)

    # 関数を定義
    d1 = nn.Linear(4, 10)
    relu = nn.ReLU()

    # データセットの最初の値を入力
    print("1-----------")
    print(relu(d1(tx[0:1])))

    # データセットの全部の値を入力
    print("2-----------")
    print(relu(d1(tx)))

    # 活性化関数を変更した関数を定義
    d1 = nn.Linear(4, 10)

    # データセットの最初の値を入力
    print("3-----------")
    print(d1(tx[0:1]))

    # データセットの全部の値を入力
    print("4-----------")
    print(d1(tx))

    # 最初の引数の値を変更した関数を定義
    d1 = nn.Linear(4, 4)

    # データセットの最初の値を入力
    print("5-----------")
    print(d1(tx[0:1]))

    # データセットの全部の値を入力
    print("6-----------")
    print(d1(tx))

    # 別の関数を定義
    d1 = nn.Linear(4, 4)
    d2 = nn.Linear(4, 5)
    relu = nn.ReLU()

    # データセットの最初の値を入力
    print("7-----------")
    y = relu(d2(d1(tx[0:1])))
    print(y)

    # データセットの全部の値を入力
    print("8-----------")
    y = relu(d2(d1(tx)))
    print(y)

if __name__ == "__main__":
    main()


# ```{note}
# このようなコードを動かすことでニューラルネットワークの中身を理解することができます．
# ```

# ### MNIST を利用した学習

# 次に，MNIST を処理して「0から9の数字が書かれた（描かれた）手書き文字を入力にして，その手書き文字が0から9のどれなのかを判別する人工知能」を構築します．以下のように書きます．

# In[ ]:


#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
torch.manual_seed(0)
np.random.seed(0)

def main():
    # GPUの使用の設定．
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ハイパーパラメータの設定．
    MAXEPOCH = 50
    MINIBATCHSIZE = 500
    UNITSIZE = 500

    # データの読み込みと前処理．
    transform = transforms.Compose([transforms.ToTensor()])
    learn_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # トレーニングセットとバリデーションセットの分割．
    train_dataset, valid_dataset = torch.utils.data.random_split(learn_dataset, [int(len(learn_dataset) * 0.9), int(len(learn_dataset) * 0.1)])

    # データローダーの設定．
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=MINIBATCHSIZE, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=len(valid_dataset), shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    # ネットワークの定義．
    model = Network(UNITSIZE, len(learn_dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # 学習ループ．
    for epoch in range(1, MAXEPOCH + 1):
        # トレーニング．
        model.train() # ドロップアウト等は動作するモード．
        traincost = 0.0
        for tx, tt in train_loader:
            tx, tt = tx.to(device), tt.to(device)
            optimizer.zero_grad()
            ty = model(tx)
            loss = criterion(ty, tt)
            loss.backward()
            optimizer.step()
            traincost += loss.item()
        traincost /= len(train_loader) # このlen(train_loader)はミニバッチの個数．
        # バリデーション．
        model.eval() # ドロップアウト等は動作しないモード．
        validcost = 0.0
        with torch.no_grad():
            for tx, tt in valid_loader:
                tx, tt = tx.to(device), tt.to(device)
                ty = model(tx)
                loss = criterion(ty, tt)
                validcost += loss.item()
        validcost /= len(valid_loader)
        # 学習過程の出力．
        print("Epoch {:4d}: Training cost= {:7.4f} Validation cost= {:7.4f}".format(epoch, traincost, validcost))

class Network(nn.Module):
    def __init__(self, UNITSIZE, OUTPUTSIZE):
        super(Network, self).__init__()
        self.flatten = nn.Flatten()
        self.d1 = nn.Linear(28*28, UNITSIZE)
        self.d2 = nn.Linear(UNITSIZE, OUTPUTSIZE)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.d1(x))
        x = self.d2(x)
        return x

if __name__ == "__main__":
    main()


# プログラムの中身について上から順に説明します．以下の部分はハイパーパラメータを設定する記述です．`MAXEPOCH` は計算させる最大エポックです．このエポックに至るまで繰り返しの学習をさせるということです．`MINIBATCHSIZE` とはミニバッチ処理でサンプリングするデータのサイズです．これが大きいとき実計算時間は短縮されます．この値が `1` のとき，学習法はオンライン学習法であり，この値がトレーニングセットのサイズと等しいとき，学習法は一括更新法です．ミニバッチの大きさは持っているマシンのスペックと相談しつつ，色々な値を試してみて一番良い値をトライアンドエラーで探します．`UNITSIZE` は MLP の層のサイズ，つまり，ニューロンの数です．
# 
# ```python
#     # ハイパーパラメータの設定．
#     MAXEPOCH = 50
#     MINIBATCHSIZE = 500
#     UNITSIZE = 500
# ```

# データの読み込みは上で説明したため省略し，以下の部分では読み込んだデータをトレーニングセットとバリデーションセットに分割しています．MNIST の学習セットは60000インスタンスからなりますが，その90%をトレーニングセットとして利用することにしています．
# ```python
#     # トレーニングセットとバリデーションセットの分割．
#     train_dataset, valid_dataset = torch.utils.data.random_split(learn_dataset, [int(len(learn_dataset) * 0.9), int(len(learn_dataset) * 0.1)])
# ```

# 以下の記述は，データローダーを設定するための記述です．PyTroch ではこのデータローダーという機能を利用してデータを人工知能に読ませます．`batch_size` でしたサイズのデータが第一引数で指定したデータセットより抽出されます．
# ```python
#     # データローダーの設定．
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=MINIBATCHSIZE, shuffle=True)
#     valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=len(valid_dataset), shuffle=False)
#     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
# ```

# ネットワークの定義は以下で行います．これは前述の例と同じです．
# ```python
#     # ネットワークの定義．
#     model = Network(UNITSIZE, len(learn_dataset.classes)).to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters())
# ```
# ネットワーク自体は以下の部分で定義されているのですが，前述の例と少し異なります．ここでは，28行28列の行列を784要素のベクトルに変換するための層 `self.flatten()` を定義しています．これにより，行列をベクトルへと変換します．
# ```python
# class Network(nn.Module):
#     def __init__(self, UNITSIZE, OUTPUTSIZE):
#         super(Network, self).__init__()
#         self.flatten = nn.Flatten()
#         self.d1 = nn.Linear(28*28, UNITSIZE)
#         self.d2 = nn.Linear(UNITSIZE, OUTPUTSIZE)
#     
#     def forward(self, x):
#         x = self.flatten(x)
#         x = torch.relu(self.d1(x))
#         x = self.d2(x)
#         return x
# ```

# 学習ループは以下で占めす通りです．最初に，`mode.train()` を実行し，ドロップアウト等は動作するモードにモデルを設定します．その後，データローダーでデータを読み出し，それらを GPU メモリに送り，ニューラルネットワークのパラメータ更新の計算を行います．バリデーションでは，`model.eval()` にてモデルをドロップアウト等が動作しないモードに変更し，その性能を計測します．
# ```python
#     # 学習ループ．
#     for epoch in range(1, MAXEPOCH + 1):
#         # トレーニング．
#         model.train() # ドロップアウト等は動作するモード．
#         traincost = 0.0
#         for tx, tt in train_loader:
#             tx, tt = tx.to(device), tt.to(device)
#             optimizer.zero_grad()
#             ty = model(tx)
#             loss = criterion(ty, tt)
#             loss.backward()
#             optimizer.step()
#             traincost += loss.item()
#         traincost /= len(train_loader) # このlen(train_loader)はミニバッチの個数．
#         # バリデーション．
#         model.eval() # ドロップアウト等は動作しないモード．
#         validcost = 0.0
#         with torch.no_grad():
#             for tx, tt in valid_loader:
#                 tx, tt = tx.to(device), tt.to(device)
#                 ty = model(tx)
#                 loss = criterion(ty, tt)
#                 validcost += loss.item()
#         validcost /= len(valid_loader)
#         # 学習過程の出力．
#         print("Epoch {:4d}: Training cost= {:7.4f} Validation cost= {:7.4f}".format(epoch, traincost, validcost))
# ```

# 次に，出力結果について説明します．このプログラムを実行するとエポックとその時のトレーニングコストとバリデーションコストが出力されます．
# ```
# Epoch    1: Training cost=  0.5361 Validation cost=  0.2647
# Epoch    2: Training cost=  0.2293 Validation cost=  0.1964
# Epoch    3: Training cost=  0.1672 Validation cost=  0.1539
# Epoch    4: Training cost=  0.1289 Validation cost=  0.1308
# Epoch    5: Training cost=  0.1041 Validation cost=  0.1163
# Epoch    6: Training cost=  0.0846 Validation cost=  0.1004
# Epoch    7: Training cost=  0.0706 Validation cost=  0.0950
# .
# .
# .
# ```
# これは各エポックのときの人工知能の性能です．エポックが50のとき，トレーニングのコストはとても小さい値です．コストは小さければ小さいほど良いので，学習はしっかりされていることが確認されます．しかし，これはトレーニングデータに対する人工知能の性能です．もしかしたらトレーニングデータに対してのみ性能を発揮できる，トレーニングデータに過剰に適合してしまった人工知能である可能性があります．だから，そうなっていないかどうかを確認する別のデータ，つまり，バリデーションデータセットにおけるコストも確認する必要があります．エポックが50のときのバリデーションのコストはエポック20くらいのときのコストより大きくなっています．すなわち，この人工知能はトレーニングデータに過剰に適合しています．おそらくエポック20くらいの人工知能が最も良い人工知能であって，これを最終的なプロダクトとして選択する必要があります．次の操作ではこれを行います．

# ### 学習曲線の描画

# 学習曲線とは横軸にエポック，縦軸にコストの値をプロットした図です．これを観察することで，どれくらいのエポックで学習が進み始めたか，人工知能の成長が止まったか，どのくらいのエポックで過剰適合が起きたか等を視覚的に理解することができます（慣れたら前述の結果のような数字を読むだけでこの図を想像できるようになるのだと思います）．

# In[ ]:


#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
torch.manual_seed(0)
np.random.seed(0)

def main():
    # GPUの使用の設定．
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ハイパーパラメータの設定．
    MAXEPOCH = 50
    MINIBATCHSIZE = 500
    UNITSIZE = 500

    # データの読み込みと前処理．
    transform = transforms.Compose([transforms.ToTensor()])
    learn_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # トレーニングセットとバリデーションセットの分割．
    train_dataset, valid_dataset = torch.utils.data.random_split(learn_dataset, [int(len(learn_dataset) * 0.9), int(len(learn_dataset) * 0.1)])

    # データローダーの設定．
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=MINIBATCHSIZE, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=len(valid_dataset), shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    # ネットワークの定義．
    model = Network(UNITSIZE, len(learn_dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # 学習ループ．
    liepoch, litraincost, livalidcost = [], [], []
    for epoch in range(1, MAXEPOCH + 1):
        # トレーニング．
        model.train() # ドロップアウト等は動作するモード．
        traincost = 0.0
        for tx, tt in train_loader:
            tx, tt = tx.to(device), tt.to(device)
            optimizer.zero_grad()
            ty = model(tx)
            loss = criterion(ty, tt)
            loss.backward()
            optimizer.step()
            traincost += loss.item()
        traincost /= len(train_loader) # このlen(train_loader)はミニバッチの個数．
        # バリデーション．
        model.eval() # ドロップアウト等は動作しないモード．
        validcost = 0.0
        with torch.no_grad():
            for tx, tt in valid_loader:
                tx, tt = tx.to(device), tt.to(device)
                ty = model(tx)
                loss = criterion(ty, tt)
                validcost += loss.item()
        validcost /= len(valid_loader)
        # 学習過程の出力．
        print("Epoch {:4d}: Training cost= {:7.4f} Validation cost= {:7.4f}".format(epoch, traincost, validcost))
        liepoch.append(epoch)
        litraincost.append(traincost)
        livalidcost.append(validcost)

    # 学習曲線の描画
    plt.plot(liepoch,litraincost,label="Training")
    plt.plot(liepoch,livalidcost,label="Validation")
    plt.ylim(0,0.2)
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.legend()
    plt.show()

class Network(nn.Module):
    def __init__(self, UNITSIZE, OUTPUTSIZE):
        super(Network, self).__init__()
        self.flatten = nn.Flatten()
        self.d1 = nn.Linear(28*28, UNITSIZE)
        self.d2 = nn.Linear(UNITSIZE, OUTPUTSIZE)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.d1(x))
        x = self.d2(x)
        return x

if __name__ == "__main__":
    main()


# 最初に，コードの変更部位について説明します．以下の部分を追加しました．これは描画に必要なライブラリである `matplotlib` を利用するための記述です．
# ```python
# import matplotlib.pyplot as plt
# ```

# 次に，学習ループの記述ですが，以下のように最初に `liepoch`，`litraincost`，`livalidcost` という3つの空の配列を用意しました．その後ループの最後で，これらの配列に，それぞれ，エポックの値，トレーニングのコストおよびバリデーションのコストをエポックを進めるたびに追加しています．
# ```python
#     # 学習ループ．
#     liepoch, litraincost, livalidcost = [], [], []
#     for epoch in range(1, MAXEPOCH + 1):
#         # トレーニング．
#         model.train() # ドロップアウト等は動作するモード．
#         traincost = 0.0
#         for tx, tt in train_loader:
#             tx, tt = tx.to(device), tt.to(device)
#             optimizer.zero_grad()
#             ty = model(tx)
#             loss = criterion(ty, tt)
#             loss.backward()
#             optimizer.step()
#             traincost += loss.item()
#         traincost /= len(train_loader) # このlen(train_loader)はミニバッチの個数．
#         # バリデーション．
#         model.eval() # ドロップアウト等は動作しないモード．
#         validcost = 0.0
#         with torch.no_grad():
#             for tx, tt in valid_loader:
#                 tx, tt = tx.to(device), tt.to(device)
#                 ty = model(tx)
#                 loss = criterion(ty, tt)
#                 validcost += loss.item()
#         validcost /= len(valid_loader)
#         # 学習過程の出力．
#         print("Epoch {:4d}: Training cost= {:7.4f} Validation cost= {:7.4f}".format(epoch, traincost, validcost))
#         liepoch.append(epoch)
#         litraincost.append(traincost)
#         livalidcost.append(validcost)
# ```

# 最後の以下の部分は学習曲線をプロットするためのコードです．
# ```python
#     # 学習曲線の描画    
#     plt.plot(liepoch,litraincost,label="Training")
#     plt.plot(liepoch,livalidcost,label="Validation")
#     plt.ylim(0,0.2)
#     plt.xlabel("Epoch")
#     plt.ylabel("Cost")
#     plt.legend()
#     plt.show()
# ```

# 結果を観ると，トレーニングセットにおけるコストの値はエポックを経るにつれて小さくなっていることがわかります．これは，人工知能が与えられたデータに適合していることを示しています．一方で，バリデーションセットにおけるコストの値は大体エポックが10と20の間くらいで下げ止まり，その後はコストが増加に転じています．このコストの増加，人工知能がこのデータセットに適合するのとは逆の方向に成長を始めたことを意味しています．この現象が起こった原因は，この人工知能がその成長に利用するデータセット（トレーニングデータセット）に（のみ）過剰に適合し，汎化性能を失ったことにあります．この曲線を観察する限り，エポックは大体10から20の間くらいに留めておいた方が良さそうです．このような画像を観て，大体20で学習を止める，みたいに決めても悪くはありませんが，もっと体系的な方法があるので次にその方法を紹介します．

# ### 早期終了

# 学習の早期終了（early stopping）とは過学習を防ぐための方法です．ここでは，ペイシェンス（patience）を利用した早期終了を紹介します．この方法では最も良い値のバリデーションコストを記録し続けます．そして学習を続け，そのベストなバリデーションコストを $n$ 回連続で更新できなかった場合，そこで学習を打ち切ります．この $n$ がペイシェンスと呼ばれる値です．ペイシェンスには我慢とか忍耐とかそのような意味があります．コードは以下のように書きます．

# In[ ]:


#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
torch.manual_seed(0)
np.random.seed(0)

def main():
    # GPUの使用の設定．
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ハイパーパラメータの設定．
    MAXEPOCH = 50
    MINIBATCHSIZE = 500
    UNITSIZE = 500
    PATIENCE = 5

    # データの読み込みと前処理．
    transform = transforms.Compose([transforms.ToTensor()])
    learn_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # トレーニングセットとバリデーションセットの分割．
    train_dataset, valid_dataset = torch.utils.data.random_split(learn_dataset, [int(len(learn_dataset) * 0.9), int(len(learn_dataset) * 0.1)])

    # データローダーの設定．
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=MINIBATCHSIZE, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=len(valid_dataset), shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    # ネットワークの定義．
    model = Network(UNITSIZE, len(learn_dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # 学習ループ．
    liepoch, litraincost, livalidcost = [], [], []
    patiencecounter, bestvalue = 0, 100000
    for epoch in range(1, MAXEPOCH + 1):
        # トレーニング．
        model.train() # ドロップアウト等は動作するモード．
        traincost = 0.0
        for tx, tt in train_loader:
            tx, tt = tx.to(device), tt.to(device)
            optimizer.zero_grad()
            ty = model(tx)
            loss = criterion(ty, tt)
            loss.backward()
            optimizer.step()
            traincost += loss.item()
        traincost /= len(train_loader) # このlen(train_loader)はミニバッチの個数．
        # バリデーション．
        model.eval() # ドロップアウト等は動作しないモード．
        validcost = 0.0
        with torch.no_grad():
            for tx, tt in valid_loader:
                tx, tt = tx.to(device), tt.to(device)
                ty = model(tx)
                loss = criterion(ty, tt)
                validcost += loss.item()
        validcost /= len(valid_loader)
        # 学習過程の出力．
        print("Epoch {:4d}: Training cost= {:7.4f} Validation cost= {:7.4f}".format(epoch, traincost, validcost))
        liepoch.append(epoch)
        litraincost.append(traincost)
        livalidcost.append(validcost)
        if validcost < bestvalue:
            bestvalue = validcost
            patiencecounter = 0
        else:
            patiencecounter += 1
        if patiencecounter == PATIENCE:
            break

    # 学習曲線の描画
    plt.plot(liepoch,litraincost,label="Training")
    plt.plot(liepoch,livalidcost,label="Validation")
    plt.ylim(0,0.2)
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.legend()
    plt.show()

class Network(nn.Module):
    def __init__(self, UNITSIZE, OUTPUTSIZE):
        super(Network, self).__init__()
        self.flatten = nn.Flatten()
        self.d1 = nn.Linear(28*28, UNITSIZE)
        self.d2 = nn.Linear(UNITSIZE, OUTPUTSIZE)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.d1(x))
        x = self.d2(x)
        return x

if __name__ == "__main__":
    main()


# プログラムには以下の部分を追加しました．今回は4回までコストが改善しなくても許すが，5回目は許さないということです．
# ```python
#     PATIENCE = 5
# ```

# 学習ループを以下のようにコードを追加しました．`patiencecounter` はコストが更新されなかった回数を数えるカウンタです．`bestvalue` は最も良いコストの値を記録する変数です．
# ```python
#     # 学習ループ．
#     liepoch, litraincost, livalidcost = [], [], []
#     patiencecounter, bestvalue = 0, 100000
#     for epoch in range(1, MAXEPOCH + 1):
#         # トレーニング．
#         model.train() # ドロップアウト等は動作するモード．
#         traincost = 0.0
#         for tx, tt in train_loader:
#             tx, tt = tx.to(device), tt.to(device)
#             optimizer.zero_grad()
#             ty = model(tx)
#             loss = criterion(ty, tt)
#             loss.backward()
#             optimizer.step()
#             traincost += loss.item()
#         traincost /= len(train_loader) # このlen(train_loader)はミニバッチの個数．
#         # バリデーション．
#         model.eval() # ドロップアウト等は動作しないモード．
#         validcost = 0.0
#         with torch.no_grad():
#             for tx, tt in valid_loader:
#                 tx, tt = tx.to(device), tt.to(device)
#                 ty = model(tx)
#                 loss = criterion(ty, tt)
#                 validcost += loss.item()
#         validcost /= len(valid_loader)
#         # 学習過程の出力．
#         print("Epoch {:4d}: Training cost= {:7.4f} Validation cost= {:7.4f}".format(epoch, traincost, validcost))
#         liepoch.append(epoch)
#         litraincost.append(traincost)
#         livalidcost.append(validcost)
#         if validcost < bestvalue:
#             bestvalue = validcost
#             patiencecounter = 0
#         else:
#             patiencecounter += 1
#         if patiencecounter == PATIENCE:
#             break
# ```
# 以下の部分で，もし最も良いコストよりさらに良いコストが得られたらベストなコストを更新し，また，ペイシェンスのカウンタを元に（`0`）戻す作業をし，それ以外の場合はペイシェンスのカウンタを1ずつ増やします．もし，カウンタの値があらかじめ設定したペイシェンスの値に達したら学習ループを停止します．
# ```python
#         if validcost < bestvalue:
#             bestvalue = validcost
#             patiencecounter = 0
#         else:
#             patiencecounter += 1
#         if patiencecounter == PATIENCE:
#             break
# ```

# 結果を観ると，過学習が起こっていなさそうなところで学習が停止されているのが解ります．

# ### モデルの保存と利用

# これまでに，早期終了を利用して良い人工知能が生成できるエポックが判明しました．機械学習の目的は当然，良い人工知能を開発することです．開発した人工知能は普通，別のサーバーとかトレーニングした時とは別の時間に利用したいはずです．ここで，この学習で発見した人工知能を保存して別のプログラムから，独立した人工知能として利用する方法を紹介します．最後に，テストセットでのその人工知能の性能を評価します．コードは以下のように変更します．

# In[ ]:


#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
torch.manual_seed(0)
np.random.seed(0)

def main():
    # GPUの使用の設定．
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ハイパーパラメータの設定．
    MAXEPOCH = 50
    MINIBATCHSIZE = 500
    UNITSIZE = 500
    PATIENCE = 5

    # データの読み込みと前処理．
    transform = transforms.Compose([transforms.ToTensor()])
    learn_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # トレーニングセットとバリデーションセットの分割．
    train_dataset, valid_dataset = torch.utils.data.random_split(learn_dataset, [int(len(learn_dataset) * 0.9), int(len(learn_dataset) * 0.1)])

    # データローダーの設定．
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=MINIBATCHSIZE, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=len(valid_dataset), shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    # ネットワークの定義．
    model = Network(UNITSIZE, len(learn_dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # 学習ループ．
    liepoch, litraincost, livalidcost = [], [], []
    patiencecounter, bestvalue = 0, 100000
    for epoch in range(1, MAXEPOCH + 1):
        # トレーニング．
        model.train() # ドロップアウト等は動作するモード．
        traincost = 0.0
        for tx, tt in train_loader:
            tx, tt = tx.to(device), tt.to(device)
            optimizer.zero_grad()
            ty = model(tx)
            loss = criterion(ty, tt)
            loss.backward()
            optimizer.step()
            traincost += loss.item()
        traincost /= len(train_loader) # このlen(train_loader)はミニバッチの個数．
        # バリデーション．
        model.eval() # ドロップアウト等は動作しないモード．
        validcost = 0.0
        with torch.no_grad():
            for tx, tt in valid_loader:
                tx, tt = tx.to(device), tt.to(device)
                ty = model(tx)
                loss = criterion(ty, tt)
                validcost += loss.item()
        validcost /= len(valid_loader)
        # 学習過程の出力．
        print("Epoch {:4d}: Training cost= {:7.4f} Validation cost= {:7.4f}".format(epoch, traincost, validcost))
        liepoch.append(epoch)
        litraincost.append(traincost)
        livalidcost.append(validcost)
        if validcost < bestvalue:
            bestvalue = validcost
            patiencecounter = 0
            torch.save(model.state_dict(), "mlp-mnist-model.pt") # モデルを保存するための記述．
        else:
            patiencecounter += 1
        if patiencecounter == PATIENCE:
            break

    # 学習曲線の描画
    plt.plot(liepoch,litraincost,label="Training")
    plt.plot(liepoch,livalidcost,label="Validation")
    plt.ylim(0,0.2)
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.legend()
    plt.show()

class Network(nn.Module):
    def __init__(self, UNITSIZE, OUTPUTSIZE):
        super(Network, self).__init__()
        self.flatten = nn.Flatten()
        self.d1 = nn.Linear(28*28, UNITSIZE)
        self.d2 = nn.Linear(UNITSIZE, OUTPUTSIZE)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.d1(x))
        x = self.d2(x)
        return x

if __name__ == "__main__":
    main()


# 以下の記述を追加しました．
# ```python
#             torch.save(model.state_dict(), "mlp-mnist-model.pt") # モデルを保存するための記述．
# ```

# 以下のシェルのコマンドを打つと，ディレクトリが新規に生成されていることを確認できます．

# In[ ]:


get_ipython().system(' ls')


# 最後に，以下のコードで保存したモデル（実体はパラメータ）を呼び出して，テストセットにてその性能を評価します．

# In[ ]:


#!/usr/bin/env python3
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
torch.manual_seed(0)
np.random.seed(0)

def main():
    # GPUの使用の設定．
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ハイパーパラメータの設定．
    MAXEPOCH = 50
    MINIBATCHSIZE = 500
    UNITSIZE = 500
    PATIENCE = 5

    # データの読み込みと前処理．
    transform = transforms.Compose([transforms.ToTensor()])
    learn_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # データローダーの設定．
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    # ネットワークの定義．
    model = Network(UNITSIZE, len(learn_dataset.classes)).to(device)

    # モデルの読み込み
    model.load_state_dict(torch.load("mlp-mnist-model.pt"))
    model.eval()
    criterion = nn.CrossEntropyLoss()

    # テストデータセットでの推論．
    testcost, testacc = 0, 0
    total_samples = len(test_dataset)
    with torch.no_grad():
        for tx, tt in test_loader:
            tx = tx.to(device)
            ty = model(tx)
            tt = tt.to(device)
            loss = criterion(ty, tt)
            testcost += loss.item()
            prediction = ty.argmax(dim=1) # Accuracyを計算するために予測値を計算．
            testacc += (prediction == tt).sum().item() / total_samples # Accuracyを計算．
    testcost /= len(test_loader)
    print("Test cost= {:7.4f} Test ACC= {:7.4f}".format(testcost,testacc))

    # テストセットの最初の画像だけに対する推論．
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    for tx, tt in test_loader:
        tx, tt = tx.to(device), tt.to(device)
        # テストセットの最初の画像を表示．
        plt.imshow(tx[0].cpu().squeeze(), cmap="gray")
        plt.text(1, 2.5, str(int(tt[0].item())), fontsize=20, color="white")
        plt.show()
        # 予測．
        ty = model(tx)
        output_vector = ty.cpu().detach().numpy()  # CPUに移動し、NumPy配列に変換
        print("Output vector:", output_vector)
        print("Argmax of the output vector:", np.argmax(output_vector))
        # 最初の画像の処理のみを行いたいため、ループを抜ける．
        break

class Network(nn.Module):
    def __init__(self, UNITSIZE, OUTPUTSIZE):
        super(Network, self).__init__()
        self.flatten = nn.Flatten()
        self.d1 = nn.Linear(28*28, UNITSIZE)
        self.d2 = nn.Linear(UNITSIZE, OUTPUTSIZE)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.d1(x))
        x = self.d2(x)
        return x

if __name__ == "__main__":
    main()


# 学習済みモデルは以下のような記述で読み込みます．
# ```python
#     # モデルの読み込み
#     model.load_state_dict(torch.load("mlp-mnist-model.pt"))
#     model.eval()
#     criterion = nn.CrossEntropyLoss()
# 
# ```

# 最後に，テストセットの最初の画像を予測器に入れてその結果を確認してみます．以下のコードで行います．
# ```python
#     # テストセットの最初の画像だけに対する推論．
#     test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
#     for tx, tt in test_loader:
#         tx, tt = tx.to(device), tt.to(device)
#         # テストセットの最初の画像を表示．
#         plt.imshow(tx[0].cpu().squeeze(), cmap="gray")
#         plt.text(1, 2.5, str(int(tt[0].item())), fontsize=20, color="white")
#         plt.show()
#         # 予測．
#         ty = model(tx)
#         output_vector = ty.cpu().detach().numpy()  # CPUに移動し、NumPy配列に変換
#         print("Output vector:", output_vector)
#         print("Argmax of the output vector:", np.argmax(output_vector))
#         # 最初の画像の処理のみを行いたいため、ループを抜ける．
#         break
# ```

# 実行すると，テストセットでも高い性能を示すことが確認できました．また，7が答えである画像を入力に，`7` を出力できていることを確認しました．

# ```{note}
# 終わりです．
# ```
