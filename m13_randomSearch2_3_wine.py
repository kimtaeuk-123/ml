# 모델 : RandomForestClassfier
import numpy as np
import tensorflow as tf

from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# x, y= load_iris(retrun_X_y=True) # 이것도 있다.
#1.데이터
dataset = load_wine()

x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=44)

kfold = KFold(n_splits=5, shuffle=True) # 5등분으로 나눈다 

parameters = [
    {"C" :[1,10,100,1000], "kernel":["linear"]}, # 1주고 linear, 10주고 linear, ... 4번
    {"C" :[1,10,100], "kernel":["rbf"], "gamma":[0.001, 0.0001]}, #6번
    {"C" :[1,10,100,1000], "kernel":["sigmoid"],"gamma":[0.001, 0.0001]}] #8번



#2. 모델
# model = GridSearchCV(SVC(), parameters, cv=kfold )
model = RandomizedSearchCV(SVC(), parameters, cv=kfold)

scores = cross_val_score(model,x_train, y_train, cv=kfold) # cross_validation

#3.훈련 
model.fit(x_train, y_train)
# fit , train 이 다 포함되어있다.

#4. 평가, 예측
print("최적의 매개변수 : ", model.best_estimator_)

y_pred = model.predict(x_test) # gird 서치에서 가장 좋은놈을 빼준다 
print("최종정답률 : ", accuracy_score(y_test, y_pred))

aaa = model.score(x_test, y_test)

print("aaa : " , aaa)


# 최적의 매개변수 :  SVC(C=10, kernel='linear')
# 최종정답률 :  0.8888888888888888
# aaa :  0.8888888888888888