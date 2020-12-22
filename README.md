# 自作pythonモジュール群
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
from neural import neuralNetwork

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
from algorithm import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors = 4)
clf.fit(x_train,t_train)
pred = clf.predict(x_test,t_test)

accuracy = KNeighborsClassifier.one_leave_out_accuracy(data,target,n_neighbors = 4)
```
- KMeans  
k平均法によるクラスタ分類のアルゴリズム
```python
from algorithm import KMeans
clf = KMeans(n_clusters = 3)
classify = clf.fit(data)

clf.visualize()
clf.visualize3D()
```
- MultipleLinearRegression  
線形重回帰による予測のアルゴリズム
```python
from algorithm import MultipleLinearRegression

lr = MultipleLinearRegression()
lr.fit(x_train,t_train)
pred = lr.predict(x_test)
score = lr.score(x_test,t_test)
```
