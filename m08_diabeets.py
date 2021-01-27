import numpy as np
import tensorflow as tf

from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

from sklearn.svm import LinearSVC, SVC, LinearSVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

dataset = load_diabetes()
x = dataset.data
y = dataset.target

model = LinearSVC()

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
# score :  -0.2911189502593887
# accuracy_score :  0.0