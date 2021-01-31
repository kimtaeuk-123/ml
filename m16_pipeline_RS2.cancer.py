#전처리 하나와 모델을 합침

import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.pipeline import Pipeline, make_pipeline
import timeit

start_time = timeit.default_timer()
import warnings
warnings.filterwarnings('ignore')

dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=44)


# Pipeline은 전처리 + 모델해줘서 MinMaxScaler문 생략 가능 
# from sklearn.preprocessing import MinMaxScaler
# scale = MinMaxScaler()
# scale.fit(x_train)
# x_train = scale.transform(x_train)
# x_test = scale.transform(x_test)

parameters = [
    {"svc__C" :[1,10,100,1000], "svc__kernel":["linear"]}, # 1주고 linear, 10주고 linear, ... 4번
    {"svc__C" :[1,10,100], "svc__kernel":["rbf"], "svc__gamma":[0.001, 0.0001]}, #3x2 6번
    {"svc__C" :[1,10,100,1000], "svc__kernel":["sigmoid"],"svc__gamma":[0.001, 0.0001]}] #4x2 8번

parameters = [
    {"mal__C" :[1,10,100,1000], "mal__kernel":["linear"]}, # 1주고 linear, 10주고 linear, ... 4번
    {"mal__C" :[1,10,100], "mal__kernel":["rbf"], "mal__gamma":[0.001, 0.0001]}, #3x2 6번
    {"mal__C" :[1,10,100,1000], "mal__kernel":["sigmoid"],"mal__gamma":[0.001, 0.0001]}] #4x2 8번

# 언더바 (_) 두개 써줘야한다 

# 2. 모델
Pipe = Pipeline([('scale', MinMaxScaler()), ('mal', SVC())])   #SVC모델과 MinMax 를합친다 , 괄호 조심
# pipe = make_pipeline(StandardScaler(), SVC())  # 두가지 방법이 있다. 

# Pipeline 써주는 이유 : 트레인만 하는게 효과적, cv만큼 스케일링, 과적합 방지, 모델에 적합해서 성능이 강화 .....


model = GridSearchCV(Pipe, parameters, cv=5) 

model.fit(x_train, y_train)

results = model.score(x_test, y_test)
print('results : ', results)

# MinMaxScaler
# results :  0.9666666666666667

# StandardScaler
# results :  0.9666666666666667

