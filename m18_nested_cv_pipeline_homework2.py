# 모델은 RandomForest 
# 파이프라인 엮어서 25번 돌리기!!
# 데이터는 wine 
import numpy as np
import tensorflow as tf

from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

dataset = load_diabetes()
x = dataset.data
y = dataset.data

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=44)

kfold = KFold(n_splits=5, shuffle=True)

parameters = [
    {"C" :[1,10,100,1000], "kernel":["linear"]}, # 1주고 linear, 10주고 linear, ... 4번
    {"C" :[1,10,100], "kernel":["rbf"], "gamma":[0.001, 0.0001]}, #3x2 6번
    {"C" :[1,10,100,1000], "kernel":["sigmoid"],"gamma":[0.001, 0.0001]}] #4x2 8번

model = GridSearchCV(SVC(), parameters, cv=kfold ) # 5번돌아간다 


scores = cross_val_score(model,x_train, y_train, cv=kfold) # cross_validation # 또 5번 돌아간다

print('교차검증점수 : ', scores) # SCV 에서 제일좋은값 

model = scores(StandardScaler(), SVC())  # 두가지 방법이 있다.

model.fit(x_train, y_train)

results = model.score(x_test, y_test)
print('results : ', results)