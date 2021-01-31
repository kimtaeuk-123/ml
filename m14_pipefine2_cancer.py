#전처리 하나와 모델을 합침
#모델을 SVC말고 써줘도 된다. 
#2번부터는 랜덤 포레스트

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

parameters = [
    {'n_estimators' : [100,200]},
    {'max_depth' : [6, 8, 10 ,12]},
    {'min_samples_leaf' : [3,5,7,10]},
    {'min_samples_split' : [2, 3, 5, 10]},
    {'n_jobs' : [-1, 2, 4]}
]

# 2. 모델
# model = Pipeline([('scale', MinMaxScaler()), ('malddong', SVC())])   #SVC모델과 MinMax 를합친다 , 괄호 조심
model = make_pipeline(StandardScaler(), RandomForestClassifier())  # 두가지 방법이 있다.

model.fit(x_train, y_train)

results = model.score(x_test, y_test)
print('results : ', results)

# MinMaxScaler
# results :  0.9649122807017544

# StandardScaler
# results :  0.9736842105263158

