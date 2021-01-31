#34줄에 
import numpy as np
import tensorflow as tf

from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

from sklearn.svm import LinearSVC, SVC, LinearSVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression  # 회귀로 써있지만 분류로 쓰인다.
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

cancer = load_breast_cancer()
x = cancer.data
y = cancer.target

model = DecisionTreeClassifier()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.5, shuffle=True, random_state=66)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

models = [LinearSVC(), SVC(), KNeighborsClassifier(), DecisionTreeClassifier(), RandomForestClassifier(), LogisticRegression()]
for i in models:
    model = i
    model.fit(x_train, y_train) #range(len(models))안됨 fit 는 숫자를 못받는다..ㅠ??
    print('\n',i)
    result=model.score(x_test, y_test)
    print('score : ', result)
    y_pred=model.predict(x_test)
    accuracy = accuracy_score(y_pred,y_test)
    print('acc : ',accuracy)



'''
model.fit(x_train, y_train)

y_pred = model.predict(x_test[-5:-1])

r2score = r2_score(y_test[-5:-1], y_pred)
print('score : ', r2score)

accuracy = accuracy_score(y_test[-5:-1], y_pred)
print('accuracy_score : ', accuracy)
'''
# LinearSVC
# score :  0.0
# accuracy_score :  0.75

# KNeighborsClassifier
# score :  0.0
# accuracy_score :  0.75

# LogisticRegression
# score :  0.0
# accuracy_score :  0.75

#DecisionTreeClassifier
# score :  1.0
# accuracy_score :  1.0