# はじめに
```{only} html
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![made-with-numpy](https://img.shields.io/badge/Made%20with-NumPy-1f425f.svg)](https://numpy.org/)
[![made-with-matplotlib](https://img.shields.io/badge/Made%20with-Matplotlib-1f425f.svg)](https://matplotlib.org/)
[![made-with-scikit-learn](https://img.shields.io/badge/Made%20with-scikit--learn-1f425f.svg)](https://scikit-learn.org/)
[![made-with-pytorch](https://img.shields.io/badge/Made%20with-PyTorch-1f425f.svg)](https://pytorch.org/)
![Colab](https://colab.research.google.com/assets/colab-badge.svg)
%[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yamada-kd/introduction-to-artificial-intelligence/blob/main)
```

## この教材について
この教材は，ハンズオン形式，または，伴走形式にて，**Python** の使い方，基本的な**機械学習法**の実装の仕方，**深層学習法**の実装の仕方を人工知能分野の初学者に紹介する教材です．

:::{panels}
:container:
:column: col-lg-6 px-2 py-2
:card:

---
**対象者**
^^^
これからデータ科学を用いた解析技術を身につけたい人．プログラミングの経験，機械学習に関する知識はないことを前提にしています．

---
**Python**
^^^
プログラミング言語 [Python](https://www.python.org/) の利用方法を紹介します．Python は機械学習法を実装するための便利なライブラリを有する，データ科学を利用した研究開発に便利な言語です．

```python
#!/usr/bin/env python3

def main():
    print("Hello world")
     
if __name__ == "__main__":
    main()
```

---
**scikit-learn**
^^^
[scikit-learn](https://scikit-learn.org/) は機械学習法を実現するためのライブラリです．決定木，サポートベクトルマシン，クラスタリング等の様々な方法を利用することができます．scikit-learn は利用方法がとても簡単でそれらをコマンド一発で実行することができます．機械学習アルゴリズムはそれぞれ性質の異なるものですが，scikit-learn には基本的な書き方があり，それさえ学んでしまえば，ある機械学習法で何らかのデータを解析するプログラムを別の機械学習法でデータを解析するプログラムに書き換えることが容易にできます．深層学習法を使う必要がない場合にはこのライブラリをまずは使ってみることは良い選択肢のひとつかもしれません．

---
**PyTorch**
^^^
[PyTorch](https://pytorch.org/) は世界で最も利用されている深層学習フレームワークのひとつです．scikit-learn ではできない複雑なニューラルネットワーク構造を実現することができます．もうひとつ，TensorFlow という人気のフレームワークがありますが，文法は共通の部分が多いため，PyTorch を学ぶことで TensorFlow で書かれたプログラムにも対応できます．PyTorch と TensorFlow（Subclassing API）の書き方は元々は [Chainer](https://chainer.org/) の書き方とほぼ同じです．

:::

### このコンテンツで学ぶこと
このコンテンツの目的は，ウェブベースの計算環境である Jupyter Notebook（このウェブページを形作っているもの）を利用して，Python や機械学習ライブラリの基本的な動作を習得することです．このコンテンツは東北大学大学院情報科学研究科のプログラミング初学者向けの授業で以前利用していた内容の一部を e-learning コンテンツとして再構築したものです．
### この環境について
Jupyter Notebook は Python を実行するための環境です．メモを取りながら Python のコードを実行することができます．この環境は，Python プログラムがコマンドライン上で実行される実際の環境とは少し異なるのですが，Python プログラムがどのように動くかということを簡単に確認しながら学習することができます．

## 教材の利用方法
この教材は Google Colaboratory（グーグルコラボラトリー）を利用して作られています．グーグルコラボラトリーは Jupyter Notebook のファイルをウェブブラウザから使えるように Google が用意してくれたアプリです．各ページの上部にロケットのアイコン <i class="fa fa-rocket" aria-hidden="true"></i> があるのでこれをクリックして各ページのファイルを Google Colaboratory 上で開いて利用してください．

### GPU の利用方法
グーグルコラボラトリーで GPU を利用するには上のメニューの「ランタイム」から「ランタイムのタイプを変更」と進み，「ハードウェアアクセラレータ」の「GPU」を選択します

### 開始前に行うこと
グーグルコラボラトリー自体の一番上の「ファイル」をクリックし，さらにポップアップで出てくる項目から「ドライブにコピーを保存」をクリックし，自身のグーグルドライブにこのウェブページ全体のソースを保存します（グーグルのアカウントが必要です）．こうすることによって，自分で書いたプログラムを実行することができるようになります．また，メモ等を自由に以下のスペースに追加することができるようになります．

### 進め方
上から順番に読み進めます．Python のコードが書かれている場合は実行ボタンをクリックして実行します．コード内の値を変えたり，関数を変えたりして挙動を確かめてみてください．

### コードセル
コードセルとは，Python のコードを書き込み実行するためのセルです．以下のような灰色のボックスで表示されていますす．ここにコードを書きます．実行はコードセルの左に表示される「実行ボタン」をクリックするか，コードセルを選択した状態で `Ctrl + Enter` を押します．環境によっては行番号が表示されていると思いますので注意してください（行番号の数字はプログラムの構成要素ではありません）．

```python
print("This is a code cell.")
```
