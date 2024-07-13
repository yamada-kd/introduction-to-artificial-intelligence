#!/usr/bin/env python
# coding: utf-8

# # PyTorch の基本的な利用方法

# ## 基本操作

# この節では PyTorch の基本的な操作方法を紹介します．

# ### インポート

# NumPy と同じように PyTorch をインポートします．

# In[ ]:


#!/usr/bin/env python3
import torch

def main():
    print(torch.__version__)

if __name__ == "__main__":
    main()


# ### テンソル

# PyTorch では「テンソル」と呼ばれる NumPy の多次元配列に類似したデータ構造を用います．2行目で PyTorch をインポートします．5行目のテンソルを生成するためのコマンドは `torch.zeros()` で，これによって，全要素が `0` であるテンソルが生成されます．最初の引数には生成されるテンソルの次元数を指定します．また，データのタイプを指定することができますが以下の場合は32ビットのフロートの値を生成しています．

# In[ ]:


#!/usr/bin/env python3
import torch

def main():
    tx = torch.zeros([3, 3], dtype=torch.float32)
    print(tx)

if __name__ == "__main__":
    main()


# 以下のようにすると，整数を生成できます．

# In[ ]:


#!/usr/bin/env python3
import torch

def main():
    tx = torch.zeros([3, 3], dtype=torch.int32)  # ここが整数を生成するための記述
    print(tx)

if __name__ == "__main__":
    main()


# データのタイプを確認したい場合とテンソルのシェイプを確認したい場合は以下のようにします．

# In[ ]:


#!/usr/bin/env python3
import torch

def main():
    tx = torch.zeros([4, 3], dtype=torch.int32)
    print(tx.dtype)
    print(tx.shape)

if __name__ == "__main__":
    main()


# 一様分布に従う乱数を生成したい場合には以下のようにします．一様分布の母数（パラメータ）は最小値と最大値です．ここでは，最小値が-1で最大値が1の一様分布 $U(-1,1)$ に従う乱数を生成します．

# In[ ]:


#!/usr/bin/env python3
import torch

def main():
    tx = torch.rand(4, 3) * 2 - 1  # 0から1の乱数を生成して、-1から1の範囲に調整
    print(tx)

if __name__ == "__main__":
    main()


# 以下のようにしてもできます．

# In[ ]:


#!/usr/bin/env python3
import torch

def main():
    tx = torch.FloatTensor(4, 3).uniform_(-1, 1)  # -1から1の範囲の一様分布から乱数を生成
    print(tx)

if __name__ == "__main__":
    main()


# 上のコードセルを何度か繰り返し実行すると一様分布に従う4行3列のテンソルの値が生成されますが，1回ごとに異なる値が出力されているはずです．これは計算機実験をする際にとても厄介です．再現性が取れないからです．これを防ぐために「乱数の種」というものを設定します．以下のコードの3行目のような指定を追加します．ここでは，0という値を乱数の種に設定していますが，これはなんでも好きな値を設定して良いです．

# ```{note}
# 何度か繰り返し実行しましょう．
# ```

# In[ ]:


#!/usr/bin/env python3
import torch
torch.manual_seed(0)  # 乱数の種を設定

def main():
    tx = torch.FloatTensor(4, 3).uniform_(-1, 1)  # -1から1の範囲の一様分布から乱数を生成
    print(tx)

if __name__ == "__main__":
    main()


# ```{note}
# 普通，科学的な計算機実験をする際に乱数の種を固定せずに計算を開始することはあり得ません．乱数を使う場合は常に乱数の種を固定しておくことを習慣づける必要があります．
# ```

# テンソルは Python 配列より変換することもできます．

# In[ ]:


#!/usr/bin/env python3
import torch

def main():
    tx = torch.tensor([2, 4], dtype=torch.float32)
    print(tx)

if __name__ == "__main__":
    main()


# ### 四則計算

# テンソルの四則計算は以下のように行います．最初に足し算を行います．NumPy と同じようにやはり element-wise な計算です．実行結果は `tensor([3, 7])` となっており，テンソルが出力されていることがわかります．

# In[ ]:


#!/usr/bin/env python3
import torch

def main():
    tx = torch.add(torch.tensor([2, 4]), torch.tensor([1, 3]))  # テンソル同士の加算
    print(tx)

if __name__ == "__main__":
    main()


# 以下では，ふたつの NumPy 多次元配列を生成しそれらを足し合わせる場合も同じようにテンソルに変換します．

# In[ ]:


#!/usr/bin/env python3
import numpy as np
import torch

def main():
    na = np.array([[1, 2], [1, 3]])
    nb = np.array([[2, 3], [4, 5]])
    tx = torch.tensor(na) + torch.tensor(nb)  # NumPy配列をテンソルに変換してから加算
    print(tx)

if __name__ == "__main__":
    main()


# ```{note}
# このような型変換の柔軟性は TensorFlow の方があります．柔軟だから良いわけではありません．
# ```

# その他の四則演算は以下のように行います．

# In[ ]:


#!/usr/bin/env python3
import numpy as np
import torch

def main():
    ta = torch.tensor([[1, 2], [1, 3]])
    tb = torch.tensor([[2, 3], [5, 6]])
    print(torch.add(ta, tb))      # 加算
    print(torch.subtract(tb, ta)) # 減算
    print(torch.multiply(ta, tb)) # 乗算
    print(torch.divide(tb, ta))   # 除算

if __name__ == "__main__":
    main()


# 上の `torch.multiply()` はテンソルの要素ごとの積（アダマール積）を計算するための方法です．行列の積は以下のように `torch.matmul()` を利用します．

# In[ ]:


#!/usr/bin/env python3
import torch

def main():
    ta = torch.tensor([[1, 2], [1, 3]], dtype=torch.float32)
    tb = torch.tensor([[2, 3], [5, 6]], dtype=torch.float32)
    print(torch.matmul(ta, tb))

if __name__ == "__main__":
    main()


# テンソルもブロードキャストしてくれます．以下のようなテンソルとスカラの計算も良い感じで解釈して実行してくれます．

# In[ ]:


#!/usr/bin/env python3
import torch

def main():
    ta = torch.tensor([[1, 2], [1, 3]], dtype=torch.float32)
    print(torch.add(ta, 1))

if __name__ == "__main__":
    main()


# 以下のように `+` や `-` を使って記述することも可能です．

# In[ ]:


#!/usr/bin/env python3
import torch

def main():
    ta = torch.tensor([2, 4], dtype=torch.float32)
    tb = torch.tensor([5, 6], dtype=torch.float32)
    print(ta + tb)
    print(tb - ta)
    print(ta * tb)
    print(tb / ta)
    print(tb // ta)
    print(tb % ta)

if __name__ == "__main__":
    main()


# 二乗の計算やテンソルの要素の総和を求めるための便利な方法も用意されています．このような方法は状況に応じてその都度調べて使います．全部覚える必要はありません．

# In[ ]:


#!/usr/bin/env python3
import torch

def main():
    nx = torch.tensor([1, 2, 3], dtype=torch.float32)
    print(torch.square(nx))
    print(torch.sum(nx))

if __name__ == "__main__":
    main()


# ### 特殊な操作

# 以下のようなスライスの実装も NumPy と同じです．

# In[ ]:


#!/usr/bin/env python3
import torch

def main():
    tx = torch.tensor([[2, 4], [6, 8]], dtype=torch.float32)
    # 列のスライスを取得
    print(tx[:, 0])

if __name__ == "__main__":
    main()


# ```{note}
# これは2行2列の行列の1列目の値を取り出す操作です．
# ```

# テンソルのサイズの変更には `torch.reshape()` を利用します．

# In[ ]:


#!/usr/bin/env python3
import torch

def main():
    tx = torch.rand(4, 5, dtype=torch.float32)
    print(tx)
    print(torch.reshape(tx, [20]))
    print(torch.reshape(tx, [1, 20]))
    print(torch.reshape(tx, [5, 4]))
    print(torch.reshape(tx, [-1, 4]))

if __name__ == "__main__":
    main()


# 以上のプログラムの6行目では4行5列の行列が生成されています．これを，20要素からなるベクトルに変換するのが7行目の記述です．また，8行目の記述では1行20列の行列を生成できます．また，9行目は5行4列の行列を生成するためのものです．同じく10行目も5行4列の行列を生成します．ここでは，`torch.reshape()` の shape を指定するオプションの最初の引数に `-1` が指定されていますが，これのように書くと自動でその値が推測されます．この場合，`5` であると推測されています．

# ### 変数の変換

# これまでに，NumPy の多次元配列を PyTorch のテンソルに変換する方法は確認しました．テンソルを NumPy 配列に変換するには明示的に `numpy()` を指定する方法があります．6行目は NumPy 配列を生成します．8行目はその NumPy 配列をテンソルに変換します．さらに，NumPy 配列に戻すためには10行目のように `.numpy()` を利用します．

# In[ ]:


#!/usr/bin/env python3
import numpy as np
import torch

def main():
    na = np.ones(5)
    print("NumPy:", na)
    ta = torch.tensor(na, dtype=torch.float32)
    print("Tensor:", ta)
    na = ta.numpy()
    print("NumPy:", na)

if __name__ == "__main__":
    main()


# PyTorch では NumPy 配列，テンソル，に加えて GPU 上のテンソルを扱う必要があります．この3つの変数のタイプを自由に変換することができます．ただし，NumPy 配列から直接 GPU 上のテンソルへの変換はできません．

# In[ ]:


#!/usr/bin/env python3
import numpy as np
import torch

def main():
    na = np.ones(5)
    print("NumPy:", na)
    ta = torch.tensor(na, dtype=torch.float32)
    print("Torch:", ta)
    na = ta.numpy()
    print("NumPy:", na)
    device = torch.device("cuda")
    ca = ta.to(device)
    print("CUDA:", ca)
    ta = ca.to("cpu", dtype=torch.float32)
    print("Torch:", ta)

if __name__ == "__main__":
    main()


# 以上のプログラムにおいて，7行目は NumPy 配列を生成します．9行目はその NumPy 配列をテンソルに変換します．さらに，NumPy 配列に戻すためには11行目のように `.numpy()` を利用します．13行目では CUDA のデバイスを定義しています．この環境で GPU を利用するには上のメニューの「ランタイム」から「ランタイムのタイプを変更」と進み，「ハードウェアアクセラレータ」の「GPU」を選択します．定義したデバイスに変数を送るには `.to()` を利用します．14行目のように記述します．16行目は GPU 上の変数を CPU 上のテンソルへと戻す記述です．これは以下のように書くこともできます．

# In[ ]:


#!/usr/bin/env python3
import numpy as np
import torch

def main():
    na = np.ones(5)
    print("NumPy:", na)
    ta = torch.tensor(na, dtype=torch.float32)
    print("Torch:", ta)
    na = ta.numpy()
    print("NumPy:", na)
    ca = ta.cuda()
    print("CUDA:", ca)
    ta = ca.cpu()
    print("Torch:", ta)

if __name__ == "__main__":
    main()


# GPU を利用できない環境にてエラーを防ぐには以下のようにすると良いでしょう．

# In[ ]:


#!/usr/bin/env python3
import numpy as np
import torch

def main():
    na = np.ones(5)
    print("NumPy:", na)
    ta = torch.tensor(na, dtype=torch.float32)
    print("Torch:", ta)
    na = ta.numpy()
    print("NumPy:", na)

    if torch.cuda.is_available():
        ca = ta.cuda()
        print("CUDA:", ca)
        ta = ca.cpu()
        print("Torch:", ta)
    else:
        print("CUDA is not available.")

if __name__ == "__main__":
    main()


# ## 最急降下法

# 深層ニューラルネットワークのパラメータを更新するためには何らかの最適化法が利用されます．最も簡単な最適化法である最急降下法を実装します．

# ### 単一の変数に対する勾配

# 深層学習の最も基本的な構成要素は行列の掛け算と微分です．PyTorch はこれを行うライブラリです．自動微分機能を提供します．微分をしたい変数の生成は以下のように行います．テンソル型の変数を生成する際に，`requires_grad=True` を追加するだけです．

# In[ ]:


#!/usr/bin/env python3
import torch

def main():
    tx = torch.tensor(5, dtype=torch.float32, requires_grad=True)
    print(tx)

if __name__ == "__main__":
    main()


# ここでは勾配の計算を紹介するため，以下の式を考えます．
# 
# $y = x^2 + 2$
# 
# これに対して以下の偏微分を計算することができます．
# 
# $\dfrac{\partial y}{\partial x} = 2x$
# 
# よって $x=5$ のときの偏微分係数は以下のように計算できます．
# 
# $\left.\dfrac{\partial y}{\partial x}\right|_{x=5}=10$
# 
# これを PyTorch で実装すると以下のように書けます．微分は10行目のように `tape.gradient()` によって行います．

# In[ ]:


#!/usr/bin/env python3
import torch

def main():
    tx = torch.tensor(5, dtype=torch.float32, requires_grad=True)  # 勾配を追跡するためにrequires_grad=Trueを指定
    ty = tx**2 + 2  # 勾配を求めたい計算式
    ty.backward()  # 勾配を計算
    grad = tx.grad  # 計算された勾配を取得
    print(grad)

if __name__ == "__main__":
    main()


# ### 複数の変数に対する勾配

# 上の程度の微分だとこの自動微分機能はさほど有難くないかもしれませんが，以下のような計算となると，そこそこ有難くなってきます．以下では，(1, 2) の行列 `ts` と (2, 2) の行列 `tt` と (2, 1) の行列 `tu` を順に掛けることで，最終的に (1, 1) の行列の値，スカラー値を得ますが，それを `tt` で微分した値を計算しています（`tt` で偏微分したので得られる行列のシェイプは `tt` と同じ）．

# In[ ]:


#!/usr/bin/env python3
import torch

def main():
    # Definition
    ts = torch.tensor([[2, 1]], dtype=torch.float32)
    tt = torch.tensor([[2, 4], [6, 8]], dtype=torch.float32, requires_grad=True)  # これが変数．勾配の追跡を有効化．
    tu = torch.tensor([[4], [1]], dtype=torch.float32)
    # Calculation
    tz = torch.matmul(torch.matmul(ts, tt), tu)
    tz.backward()  # 勾配を計算．
    grad = tt.grad  # ttに関するtzの勾配を取得．
    print(grad)

if __name__ == "__main__":
    main()


# これは以下のような計算をしています．`tf.Variable()` で定義される行列は以下です：
# 
# $
#   t = \left[
#     \begin{array}{cc}
#       v & w \\
#       x & y \\
#     \end{array}
#   \right]
# $．
# 
# また，`tf.constant()` で定義される行列は以下です：
# 
# $s = \left[
#     \begin{array}{cc}
#       2 & 1 \\
#     \end{array}
#   \right]
# $，
# 
# $u = \left[
#     \begin{array}{c}
#       4 \\
#       1
#     \end{array}
#   \right]
# $．
# 
# これに対して11行目の計算で得られる値は以下です：
# 
# $z(v,w,x,y) = 8v+2w+4x+y$．
# 
# よってこれらを偏微分して，それぞれの変数がプログラム中で定義される値のときの値は以下のように計算されます：
# 
# $\left.\dfrac{\partial z}{\partial v}\right|_{(v,w,x,y)=(2,4,6,8)}=8$，
# 
# $\left.\dfrac{\partial z}{\partial w}\right|_{(v,w,x,y)=(2,4,6,8)}=2$，
# 
# $\left.\dfrac{\partial z}{\partial x}\right|_{(v,w,x,y)=(2,4,6,8)}=4$，
# 
# $\left.\dfrac{\partial z}{\partial y}\right|_{(v,w,x,y)=(2,4,6,8)}=1$．

# ```{note}
# これにコスト関数と活性化関数付けて最急降下法やったらニューラルネットワークです．自動微分すごい．
# ```

# ### 最急降下法の実装

# なぜ微分を求めたいかというと，勾配法（深層学習の場合，普通，最急降下法）でパラメータをアップデートしたいからです．以下では最急降下法を実装してみます．最急降下法は関数の最適化法です．ある関数に対して極小値（極大値）を計算するためのものです．以下のような手順で計算が進みます．
# 

# 1.   初期パラメータ（$\theta_0$）をランダムに生成します．
# 2.   もしパラメータ（$\theta_t$）が最適値または，最適値に近いなら計算をやめます．ここで，$t$ は以下の繰り返しにおける $t$ 番目のパラメータです．
# 3.   パラメータを以下の式によって更新し，かつ，$t$ の値を $1$ だけ増やします．ここで，$\alpha$ は学習率と呼ばれる更新の大きさを決める値で，$g_t$ は $t$ のときの目的の関数の勾配です．<br>
#     $\theta_{t+1}=\theta_t-\alpha g_t$
# 4.   ステップ2と3を繰り返します．
# 

# ここでは以下の関数を考えます．
# 
# $\displaystyle y=f(x)=\frac{1}{2}(x+1)^2+1$
# 
# よって勾配ベクトル場は以下のように計算されます．
# 
# $\nabla f=x+1$
# 
# 初期パラメータを以下のように決めます（実際にはランダムに決める）．
# 
# $x_0=1.6$
# 
# この関数の極小値を見つけたいのです．これは解析的に解くのはとても簡単で，括弧の中が0になる値，すなわち $x$ が $-1$ のとき，極小値 $y=1$ です．

# 最急降下法で解くと，以下の図のようになります．最急降下法は解析的に解くことが難しい問題を正解の方向へ少しずつ反復的に動かしていく方法です．

# <img src="https://github.com/yamada-kd/introduction-to-artificial-intelligence/blob/main/image/gradientDescent.svg?raw=1" width="100%" />

# これを PyTorch を用いて実装すると以下のようになります．出力中，`Objective` は目的関数の値，`Solution` はその時点での解です．最終的に $x=-0.9912\simeq-1$ のとき，最適値 $y=1$ が出力されています．

# In[ ]:


#!/usr/bin/env python3
import torch

def main():
    tx = torch.tensor(1.6, dtype=torch.float32, requires_grad=True)  # これが変数．勾配の追跡を有効化．
    epoch, update_value, lr = 1, 5, 0.1  # 更新値はダミー変数．
    while abs(update_value) > 0.001:
        ty = (1/2) * (tx + 1)**2 + 1
        ty.backward()  # 勾配を計算．
        with torch.no_grad():  # 勾配更新時には自動微分を無効化
            update_value = lr * tx.grad.item() # item()メソッドを使うことでPythonの数値を取得できる．
            tx -= update_value
            tx.grad.zero_()  # 勾配を手動でゼロクリア
        print("Epoch {:4d}:\tObjective = {:5.3f}\tSolution = {:7.4f}".format(epoch, ty, tx.detach().numpy()))
        epoch += 1

if __name__ == "__main__":
    main()


# 5行目で最初のパラメータを発生させています．通常は乱数によってこの値を決めますが，ここでは上の図に合わせて1.6とします．次の6行目では，最初のエポック，更新値，学習率を定義します．エポックとは（ここでは）パラメータの更新回数のことを言います．7行目は終了条件です．以上のような凸関数においては勾配の値が0になる点が本当の最適値（正しくは停留点）ではありますが，計算機的にはパラメータを更新する値が大体0になったところで計算を打ち切ります．この場合，「大体0」を「0.001」としました．8
# 行目は目的の関数，9行目で微分をしています．11行目は最急降下法で更新する値を計算しています．12行目の計算で `tx` をアップデートします．この12行目こそが上述の最急降下法の式です．

# ```{note}
# PyTorch はデフォルトで計算した勾配を蓄積する特徴を持っています．テンソルの `.grad` 属性に蓄積されます．つまり，`.backward()` を行った後にもう一度 `.backward()` を行うと最初に計算した勾配に次に計算した勾配の値が加算されるのです．この蓄積を失くすために `.zero_()` をする必要があります．
# ```

# ```{note}
# ここで最急降下法について説明しましたが，このような実装は PyTorch を利用する際にする必要はありません．PyTorch はこのような計算をしてくれる方法を提供してくれています．よって，ここの部分の意味が解らなかったとしても以降の部分は理解できます．
# ```

# ```{note}
# 終わりです．
# ```
