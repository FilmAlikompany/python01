# NE19-1131J
# Akiya EGAWA

from sklearn import datasets
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame

# データセットを読み込む
boston = datasets.load_boston() # ボストン市の住宅価格と関連データ
boston_df = DataFrame(boston.data) # DataFrame型にする
boston_df.columns = boston.feature_names # 列名を設定する
boston_df["Price"] = boston.target # 住宅価格を追加する

# 訓練データを作る
rooms_train = DataFrame(boston_df["RM"]) # 部屋数のデータを抜き出す
age_train = DataFrame(boston_df["AGE"]) # 1940年より前に建てられた物件数のデータを抜き出す
y_train = boston.target    # ターゲット（住宅価格）

model = linear_model.LinearRegression() # 回帰モデルを作る
model2 = linear_model.LinearRegression() # 回帰モデル2を作る

model.fit(rooms_train, y_train) # 訓練する
model2.fit(age_train, y_train) # 訓練する

# 部屋数のテストデータを作る
rooms_test = DataFrame(np.arange(rooms_train.values.min(), rooms_train.values.max(), 0.1))
age_test = DataFrame(np.arange(age_train.values.min(), age_train.values.max(), 0.1))

prices_test = model.predict(rooms_test)
prices_test2 = model2.predict(age_test)

# グラフ表示する
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True) # 1行2列のサブプロットを追加

ax1.scatter(rooms_train.values.ravel(), y_train, c= "b", alpha = 0.5) # 訓練データ
ax2.scatter(age_train.values.ravel(), y_train, c= "b", alpha = 0.5) # 訓練データ

ax1.plot(rooms_test.values.ravel(), prices_test, c = "r") # 回帰直線
ax2.plot(age_test.values.ravel(), prices_test2, c = "r") # 回帰直線

plt.title("Boston House Prices dataset")

plt.show()