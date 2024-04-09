#!/usr/bin/env python
# coding: utf-8

# # 畳み込みニューラルネットワーク

# ## 基本的な事柄

# 画像データを処理することが得意なニューラルネットワークである畳み込みニューラルネットワーク（convolutional neural network（CNN））とそれに関する基本的な事柄をまとめます．

# ### CNN とは

# ほげ
# 
# <img src="https://github.com/yamada-kd/introduction-to-artificial-intelligence/blob/main/image/cnn_01.svg?raw=1" width="100%" />

# ### 畳み込み

# ほげ

# ### プーリング

# ほげ

# ## CNN の実装

# この節では CNN の使い方を紹介します．前章の MLP の実装を拡張する方法で CNN を実装します．

# ### 基となる MLP

# 以下のものは前章の最後で紹介した MLP を実装するためのものです．MNIST に対する予測器を MLP を用いて構築し，さらに構築した MLP のパラメータ情報を保存するものです．学習の際には学習曲線を出力し，また，早期停止によって過学習を避けるようにしています．

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


# ### CNN の計算

# これまでにオブジェクト指向の書き方を学びましたが，その有用性をこのコードの改造で実感できます．ここでは，以上の MLP の計算で定義したクラス，`Network` のみを CNN のものに置き換えることで，CNN を実装します．以下のようにします．

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
    model = Network(len(learn_dataset.classes)).to(device)
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
            torch.save(model.state_dict(), "cnn-mnist-model.pt")
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
    def __init__(self, OUTPUTSIZE):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2) # 畳み込み層1: 1チャネル入力, 32チャネル出力, カーネルサイズ5, ストライド1, パディング2．
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2) # 畳み込み層2: 32チャネル入力, 64チャネル出力, カーネルサイズ5, ストライド1, パディング2．
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # プーリング層: カーネルサイズ2, ストライド2．
        self.fc1 = nn.Linear(64 * 7 * 7, 500) # 全結合層1: 64*7*7入力, 500出力．
        self.fc2 = nn.Linear(500, OUTPUTSIZE) # 全結合層2（出力層）: 500入力, 出力サイズ（クラス数）．

    def forward(self, x): # 入力: [バッチサイズ, 1, 28, 28]
        x = self.pool(torch.relu(self.conv1(x)))  # [バッチサイズ, 32, 14, 14]
        x = self.pool(torch.relu(self.conv2(x)))  # [バッチサイズ, 64, 7, 7]
        x = x.view(-1, 64 * 7 * 7)  # [バッチサイズ, 64*7*7]
        x = torch.relu(self.fc1(x))  # [バッチサイズ, 500]
        x = self.fc2(x)  # [バッチサイズ, 出力サイズ]
        return x

if __name__ == "__main__":
    main()


# ネットワークを定義するクラスを以下のように書き換えました．`nn.Conv2d()` は畳み込み層のためのもの，`nn.MaxPool2d()` は最大値プーリングを計算するためのものです．また，``nn.Linear()` はこれまでにも紹介した通り全結合を計算するためのものです．
# ```python
# class Network(nn.Module):
#     def __init__(self, OUTPUTSIZE):
#         super(Network, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2) # 畳み込み層1: 1チャネル入力, 32チャネル出力, カーネルサイズ5, ストライド1, パディング2．
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2) # 畳み込み層2: 32チャネル入力, 64チャネル出力, カーネルサイズ5, ストライド1, パディング2．
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # プーリング層: カーネルサイズ2, ストライド2．
#         self.fc1 = nn.Linear(64 * 7 * 7, 500) # 全結合層1: 64*7*7入力, 500出力．
#         self.fc2 = nn.Linear(500, OUTPUTSIZE) # 全結合層2（出力層）: 500入力, 出力サイズ（クラス数）．
#     
#     def forward(self, x): # 入力: [バッチサイズ, 1, 28, 28]
#         x = self.pool(torch.relu(self.conv1(x)))  # [バッチサイズ, 32, 14, 14]
#         x = self.pool(torch.relu(self.conv2(x)))  # [バッチサイズ, 64, 7, 7]
#         x = x.view(-1, 64 * 7 * 7)  # [バッチサイズ, 64*7*7]
#         x = torch.relu(self.fc1(x))  # [バッチサイズ, 500]
#         x = self.fc2(x)  # [バッチサイズ, 出力サイズ]
#         return x
# ```

# 学習は以下のように進んでいます．パラメータサイズが大きいということは影響していますが，少なくとも MLP より学習が遅いということはないように思えます．
# ```
# Epoch    1: Training cost=  0.3785 Validation cost=  0.0988
# Epoch    2: Training cost=  0.0739 Validation cost=  0.0607
# Epoch    3: Training cost=  0.0476 Validation cost=  0.0510
# Epoch    4: Training cost=  0.0372 Validation cost=  0.0401
# Epoch    5: Training cost=  0.0273 Validation cost=  0.0338
# Epoch    6: Training cost=  0.0221 Validation cost=  0.0398
# Epoch    7: Training cost=  0.0204 Validation cost=  0.0298
# Epoch    8: Training cost=  0.0155 Validation cost=  0.0337
# Epoch    9: Training cost=  0.0112 Validation cost=  0.0368
# Epoch   10: Training cost=  0.0096 Validation cost=  0.0356
# Epoch   11: Training cost=  0.0079 Validation cost=  0.0338
# Epoch   12: Training cost=  0.0063 Validation cost=  0.0365
# ```

# 最後に，以上で保存したモデルを新たなプログラムで読み込んで，テストデータセットに対する予測をします．以下のように書きます．

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
    PATIENCE = 5

    # データの読み込みと前処理．
    transform = transforms.Compose([transforms.ToTensor()])
    learn_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # データローダーの設定．
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    # ネットワークの定義．
    model = Network(len(learn_dataset.classes)).to(device)

    # モデルの読み込み
    model.load_state_dict(torch.load("cnn-mnist-model.pt"))
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
    def __init__(self, OUTPUTSIZE):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2) # 畳み込み層1: 1チャネル入力, 32チャネル出力, カーネルサイズ5, ストライド1, パディング2．
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2) # 畳み込み層2: 32チャネル入力, 64チャネル出力, カーネルサイズ5, ストライド1, パディング2．
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # プーリング層: カーネルサイズ2, ストライド2．
        self.fc1 = nn.Linear(64 * 7 * 7, 500) # 全結合層1: 64*7*7入力, 500出力．
        self.fc2 = nn.Linear(500, OUTPUTSIZE) # 全結合層2（出力層）: 500入力, 出力サイズ（クラス数）．

    def forward(self, x): # 入力: [バッチサイズ, 1, 28, 28]
        x = self.pool(torch.relu(self.conv1(x)))  # [バッチサイズ, 32, 14, 14]
        x = self.pool(torch.relu(self.conv2(x)))  # [バッチサイズ, 64, 7, 7]
        x = x.view(-1, 64 * 7 * 7)  # [バッチサイズ, 64*7*7]
        x = torch.relu(self.fc1(x))  # [バッチサイズ, 500]
        x = self.fc2(x)  # [バッチサイズ, 出力サイズ]
        return x

if __name__ == "__main__":
    main()


# 高い予測性能で予測に成功していることが分かります．

# ```{note}
# 終わりです．
# ```
