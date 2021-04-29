# NE19-1131J
# Akiya EGAWA
# エディタを使用する場合、スペルチェックで警告が出る場合がありますが、問題なく実行できます。

from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt

# 訓練データとテストデータの割合を調節する
train_size = 0.75
test_size = 1 - train_size

# アヤメのデータセットを読み込む
iris = datasets.load_iris()

X = iris.data
Y = iris.target

# データをプロットする
plt.scatter(X[:50, 0], X[:50, 1], color='r', marker='o', label='setosa' )
plt.scatter(X[50:100, 0], X[50:100, 1], color='g', marker='+', label='versicolor' )
plt.scatter(X[100:, 0], X[100:, 1], color='b', marker='x', label='verginica' )
plt.title("Iris Plants Database")
plt.xlabel('petal length(cm)')
plt.ylabel('petal width(cm)')
plt.legend()

# データを分割するインデックスを作る
iris_ss = ShuffleSplit(train_size=train_size, test_size=test_size, random_state=0)
train_index, test_index = next(iris_ss.split(X) )

# データを分割する
X_train, Y_train = X[train_index], Y[train_index] # 訓練データ
X_test, Y_test = X[test_index], Y[test_index] # テストデータ
clf = svm.SVC() # モデルを作る
clf.fit(X_train, Y_train) # 訓練する
print("【訓練データとテストデータの割合】\n訓練データ" + str(train_size * 100) + "%" + "\nテストデータ：" + str(test_size * 100) + "%")
print( '【正答率】\n' + str(clf.score(X_test, Y_test)) ) # 正答率を調べる
plt.show()