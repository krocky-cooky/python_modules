# pythonモジュール群
## 利用方法
```sh
git clone https://github.com/krocky-cooky/python_modules
echo export PATH=$PATH:追加したいコマンド検索パス >> ~/.bash_profile
source ~/.bash_profile
```
## ML
### neural.py
深層学習ライブラリであるneuralNetworkクラスを定義している。
```python
from ML import neuralNetwork

net = neuralNetwork(
    epoch = 20000,
    learning_rate = 0.1,
    batch_size = 100
)
net.set_layer([64,[100,50],10])
net.train(x_train,t_train)
accuracy = net.accuracy(x_test,t_test)
```

### algorithm.py
- KNeighborsClassifier  
k点近傍法でのクラス分類アルゴリズム
```python
from ML import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors = 4)
clf.fit(x_train,t_train)
pred = clf.predict(x_test,t_test)

accuracy = KNeighborsClassifier.one_leave_out_accuracy(data,target,n_neighbors = 4)
```
- KMeans  
k平均法によるクラスタ分類のアルゴリズム
```python
from ML import KMeans
clf = KMeans(n_clusters = 3)
classify = clf.fit(data)

clf.visualize()
clf.visualize3D()
```
- MultipleLinearRegression  
線形重回帰による予測のアルゴリズム
```python
from ML import MultipleLinearRegression

lr = MultipleLinearRegression()
lr.fit(x_train,t_train)
pred = lr.predict(x_test)
score = lr.score(x_test,t_test)
```
-HierarchicClustering
階層的クラスタリングモデル
```python
from ML import HierarchicClustering as HC

clf = HC(method = 'ward')
clf.fit(data)
clf.visualize()
```
-SoftmaxRegression
ソフトマックス回帰モデルの予測アルゴリズム
```python
from ML import SoftmaxRegression

rgr = SoftmaxRegression(iter = 100,learning_rate = 0.1)
rgr.train(x_train,t_train)
rgr.accuracy(x_test,t_test)

