import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataset = load_wine()

x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=77)

model = DecisionTreeClassifier()

model.fit(x_train, y_train)

#################  모델 완성

fi = model.feature_importances_


new_data = []
feature = []

for i in range(len(fi)):  #len은 fi의 요소의 갯수를 가져온다!! 
    
    if fi[i] != 0:
        new_data.append(dataset.data[:,i])
        feature.append(dataset.feature_names[i])


new_data = np.array(new_data)
print(new_data.shape) #(12, 569)

new_data = np.transpose(new_data)
print(new_data.shape) #(569, 14)

x_train, x_test, y_train, y_test = train_test_split(new_data, dataset.target,train_size=0.8, random_state=77)

model.fit(x_train, y_train)

print(model.feature_importances_)

importances = model.feature_importances_

# importances= np.argsort(importances)[::-1]

print(importances)

# n_features 독립 변수의 수
# barh 가로막대
# ylim 축제한 (-1 까지 보이게하겠다. )
# align 정렬
def plot_feature_importances_dataset(model):
    
    n_features = new_data.shape[1]
    
    plt.barh(np.arange(n_features), sorted(importances), align = 'center')
    plt.yticks(np.arange(n_features), feature)
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    
    plt.ylim(-1, n_features)

plot_feature_importances_dataset(model)

plt.show()
