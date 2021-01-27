import numpy as np
import tensorflow as tf

from sklearn.datasets import load_diabetes, load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

from sklearn.svm import LinearSVC, SVC, LinearSVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression  # 회귀로 써있지만 분류로 쓰인다.
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

dataset = load_wine()
x = dataset.data
y = dataset.target

model = RandomForestClassifier()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.5, shuffle=True, random_state=66)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

r2score = r2_score(y_test, y_pred)
print('score : ', r2score)

accuracy = accuracy_score(y_test, y_pred)
print('accuracy_score : ', accuracy)

# LinearSVC
# score :  0.96217594560136
# accuracy_score :  0.9775280898876404

# DecisionTreeClassifier
# score :  0.9054398640033999
# accuracy_score :  0.9438202247191011

# KNeighborsClassifier
# score :  0.9054398640033999
# accuracy_score :  0.9438202247191011

# RandomForestClassifier
# score :  0.96217594560136
# accuracy_score :  0.9775280898876404