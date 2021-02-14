import numpy as np
#피처임포턴스가 0인 컬럼들을 제거 하여 데이터 셋을 재구성
#DescisionTree로 모델을 돌려서 acc 확인

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pandas as pd

#1. 데이터
dataset = load_iris()
x = dataset.data
y = dataset.target

df = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=77)

#타임걸기
import timeit
start_time = timeit.default_timer()

#2. 모델
# model = GradientBoostingClassifier(max_depth=4)
model = XGBClassifier(n_jobs=8, n_estimators=100)   #n_jobs -1 : cpu자원을 모두 쓰겠다 8일때랑 똑같아야한다 
#3. 훈련
model.fit(x_train, y_train, eval_metric= 'logloss', verbose=True,
            eval_set=[(x_train,y_train), (x_test, y_test)]) #waring 나올때 eval_metric 써준다 

#4. 평가, 예측
acc = model.score(x_test, y_test)

print(model.feature_importances_) #[0.0244404  0.01669101 0.00766884 0.95119975] 다 합치면 1
print('acc : ', acc)
#타임걸기
#n_jobs = -1,8,4,1 속도 비교 

fi = model.feature_importances_

new_data = []
feature = []

for i in range(len(fi)):
    if fi[i] != 0:
        new_data.append(dataset.data[:,i])
        feature.append(dataset.feature_names[i])

new_data = np.array(new_data)
new_data = np.transpose(new_data)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=77)

model.fit(x_train, y_train)
terminate_time = timeit.default_timer()

print("%f초 걸렸습니다." %(terminate_time-start_time))
# 0.128167 0.122586 0.120436 0.134993 -1 
# 0.126824 0.127463 0.137608 0.122113 8
# 0.119371 0.119338 0.124366 0.124781 1

import matplotlib.pyplot as plt

def plot_feature_importances_dataset(model):
    n_features = new_data.data.shape[1]
    plt.barh(np.arange(n_features), sorted(model.feature_importances_), align='center')
    plt.yticks(np.arange(n_features), dataset.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1,n_features)

plot_feature_importances_dataset(model)
plt.show()

# 0.9122807017543859/ worst perimeter

