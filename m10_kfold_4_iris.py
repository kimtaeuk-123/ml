import numpy as np
import tensorflow as tf

from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# x, y= load_iris(retrun_X_y=True) # 이것도 있다.
#1.데이터
dataset = load_iris()

x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=77)


kfold = KFold(n_splits=5, shuffle=True) # 5등분으로 나눈다 


#2. 모델
model = LinearSVC()
# model = SVC()
# model = LogisticRegression()
# model = KNeighborsClassifier()
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()




scores = cross_val_score(model,x_train,y_train, cv=kfold) # cross_validation
# fit , train 이 다 포함되어있다.



print('scores :' , scores)
'''
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
# scores : [1.         0.95833333 1.         0.95833333 0.91666667]

# SVC
# scores : [1.         1.         0.95833333 0.83333333 0.91666667]

# KNeighborsClassifier
# scores : [1.         1.         1.         1.         0.95833333]

# DecisionTreeClassifier()
# scores : [1.         0.95833333 0.95833333 0.95833333 0.95833333]

# RandomForestClassifier()
# scores : [1.         1.         0.91666667 1.         0.95833333]

# model = LogisticRegression()
# scores : [1.         1.         0.95833333 0.875      0.95833333]
'''