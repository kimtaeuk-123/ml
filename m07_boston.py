# boston 회귀모델 

import numpy as np
import tensorflow as tf

from sklearn.datasets import load_diabetes, load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

# from sklearn.svm import LinearSVC, SVC, LinearSVR
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
# from sklearn.linear_model import LogisticRegression  # 회귀로 써있지만 분류로 쓰인다.
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

dataset = load_boston()
x = dataset.data
y = dataset.target

model = LinearRegression()
# model = KNeighborsRegressor()
# model = DecisionTreeRegressor()
# model = RandomForestRegressor()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.5, shuffle=True, random_state=66)

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

r2score = r2_score(y_test, y_pred)
print('score : ', r2score)

result = model.score(x_train, y_train)
print('result : ', result)

# accuracy = accuracy_score(y_test, y_pred)
# print('accuracy_score : ', accuracy)

# LinearSVC
