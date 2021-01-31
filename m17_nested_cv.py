# 모델 : RandomForestClassfier 느려짐
import numpy as np
import tensorflow as tf

from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# x, y= load_iris(retrun_X_y=True) # 이것도 있다.
#1.데이터
dataset = load_iris()

x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=44)
kfold = KFold(n_splits=5, shuffle=True) # 5등분으로 나눈다 

# 리스트형식 키밸류 형식, 대괄호: 딕셔너리 

parameters = [
    {"C" :[1,10,100,1000], "kernel":["linear"]}, # 1주고 linear, 10주고 linear, ... 4번
    {"C" :[1,10,100], "kernel":["rbf"], "gamma":[0.001, 0.0001]}, #3x2 6번
    {"C" :[1,10,100,1000], "kernel":["sigmoid"],"gamma":[0.001, 0.0001]}] #4x2 8번

#2. 모델
model = GridSearchCV(SVC(), parameters, cv=kfold ) # 5번돌아간다 
# scores = cross_val_score(model,x, y, cv=kfold) # cross_validation # 또 5번 돌아간다

# model = RandomForestClassifier()
for train_idx, val_idx in kfold.split(x,y): 
    # train fold, val fold 분할
    x_train = x[train_idx]
    x_test = x[train_idx]
    y_train = y[train_idx]
    y_test = y[train_idx]

    scores = cross_val_score(model, x_train, y_train, cv = kfold)
    print('score : ', scores)

print('교차검증점수 : ', scores) # SVC 에서 제일좋은값 

