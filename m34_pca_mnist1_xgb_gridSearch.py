# m31로 만든 0.95 이상의 n_component=? 사용하여 
# dnn 모델을 만들것

# mnist dnn 보다 성능 좋게 만들어라 

# RandomSearch 로도 해볼것 
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