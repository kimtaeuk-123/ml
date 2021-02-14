# 데이터별로 5개 만든다.

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

dataset = load_iris()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=44)

kfold = KFold(n_splits=5, shuffle=True)

parameters = [
    {"n_estimators":[100, 200, 300], "learning_rate" :[0.1, 0.3, 0.001, 0.01],
     "max_depth":[4,5,6]},
    {"n_estimators":[90, 100, 110], "learning_rate" :[0.1, 0.001, 0.01],
     "max_depth":[4,5,6], "colsample_bytree":[0.6, 0.9, 1]},
    {"n_estimators":[90, 110], "learning_rate" :[0.1,0.001, 0.5],
     "max_depth":[4,5,6], "colsample_bytree":[0.6, 0.9, 1],
     "colsample_bylevel":[0.6, 0.7, 0.9]}
    
]
n_jobs= -1

model = GridSearchCV(XGBClassifier(), parameters, cv=kfold )

scores = cross_val_score(model,x_train, y_train, cv=kfold) # cross_validation

model.fit(x_train, y_train, eval_metric='logloss')

print("최적의 매개변수 : ", model.best_estimator_)

y_pred = model.predict(x_test) # gird 서치에서 가장 좋은놈을 빼준다 
print("최종정답률 : ", accuracy_score(y_test, y_pred))