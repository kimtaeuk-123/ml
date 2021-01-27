import numpy as np
import tensorflow as tf

from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# x, y= load_iris(retrun_X_y=True) # 이것도 있다.



dataset = load_iris()

x = dataset.data
y = dataset.target
# print(dataset.DESCR)
# print(dataset.feature_names)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# print(x.shape) #(150, 4)
# print(y.shape) #(150, )
# print(x[:5])
# print(y)

from tensorflow.keras.utils import to_categorical
# from keras.utils.np_utils import to_categorical 위에랑 동일

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# print(y_train)
# print(y_train.shape)#(120,3)  #print(y.shape) (150,3)
# print(y_test)
# print(y_test.shape) #(30,3)

#2. 모델
model = LinearSVC()
model.fit(x,y)

#4.  평가 예측
# model.compile(loss='mse', optimizer='adam', metrics='acc')
# loss, acc = model.evaluate(x_test, y_test) 
result = model.score(x,y)
print(result)

# from sklearn.metrics import r2_score
# r2= r2_score(x, y)
# print('r2 = ', r2_score)

y_pred = model.predict(x[-5:-1])
print(y_pred)
# print(y[-5:-1])

acc = accuracy_score(y_test[-5:-1], y_pred)
print(acc)
# 결과치 나오게 수정argmax
# np.argmax(y_pred,axis=-1)
# print(y_pred)

# LinearSVC,
# 0.9666666666666667
# [2 2 2 2]
# [2 2 2 2]

# KNeighborsClassifier
# 0.9666666666666667
# [2 2 2 2]

# DecisionTreeClassifier()
# 1.0
# [2 2 2 2]

# RandomForestClassifier()
# 1.0
# [2 2 2 2]