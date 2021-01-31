# 모델 : RandomForestClasstifier

import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score

dataset = load_iris()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=44)

kfold = KFold(n_splits=5, shuffle=True) # 5등분으로 나눈다 

parameters = [
    {'n_estimators' : [100,200]},
    {'max_depth' : [6, 8, 10 ,12]}, #깊이 
    {'min_samples_leaf' : [3,5,7,10]},
    {'min_samples_split' : [2, 3, 5, 10]}, 
 # 'min_samples + split써줘서 오타.. ValueError: Invalid parameter min_samples+split for estimator RandomForestClassifier(). 
 # Check the list of available parameters with `estimator.get_params().keys()`.
 # 오타나면 수정 
    {'n_jobs' : [-1, 2, 4]} # cpu를 몇개 쓸꺼냐, -1 -> 전체, 2 -> 2개 4 -> 4개 
]


#2. 모델
model = GridSearchCV(RandomForestClassifier(), parameters, cv=kfold )

# scores = cross_val_score(model,x_train, y_train, cv=kfold) # cross_validation

#3.훈련 
model.fit(x_train, y_train)
# fit , train 이 다 포함되어있다.

#4. 평가, 예측
print("최적의 매개변수 : ", model.best_estimator_)

y_pred = model.predict(x_test) # gird 서치에서 가장 좋은놈을 빼준다 
print("최종정답률 : ", accuracy_score(y_test, y_pred))

# aaa = model.score(x_test, y_test)

# print("aaa : " , aaa)
