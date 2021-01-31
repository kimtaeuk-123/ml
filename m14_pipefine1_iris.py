#전처리 하나와 모델을 합침

import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.datasets import load_iris
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

dataset = load_iris()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=44)


# Pipeline은 전처리 + 모델해줘서 MinMaxScaler문 생략 가능 
# from sklearn.preprocessing import MinMaxScaler
# scale = MinMaxScaler()
# scale.fit(x_train)
# x_train = scale.transform(x_train)
# x_test = scale.transform(x_test)

# 2. 모델
# model = Pipeline([('scale', MinMaxScaler()), ('malddong', SVC())])   #SVC모델과 MinMax 를합친다 , 괄호 조심
model = make_pipeline(StandardScaler(), SVC())  # 두가지 방법이 있다.

model.fit(x_train, y_train)

results = model.score(x_test, y_test)
print('results : ', results)

# MinMaxScaler
# results :  0.9666666666666667

# StandardScaler
# results :  0.9666666666666667

