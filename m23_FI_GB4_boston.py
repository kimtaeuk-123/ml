import numpy as np
#피처임포턴스가 0인 컬럼들을 제거 하여 데이터 셋을 재구성
#DescisionTree로 모델을 돌려서 acc 확인

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.datasets import load_breast_cancer, load_boston
from sklearn.model_selection import train_test_split
import pandas as pd

#1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target

df = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=77)

#2. 모델
model = GradientBoostingRegressor(max_depth=4)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
acc = model.score(x_test, y_test)

print(model.feature_importances_) #[0.0244404  0.01669101 0.00766884 0.95119975] 다 합치면 1
print('acc : ', acc)


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

