
import numpy as np
import tensorflow as tf

from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# x, y= load_iris(retrun_X_y=True) # 이것도 있다.
#1.데이터
dataset = load_diabetes()

x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=77)


kfold = KFold(n_splits=5, shuffle=True) # 5등분으로 나눈다 


#2. 모델
# model = LinearSVC()
# model = SVC()
# model = LogisticRegression()
# model = KNeighborsClassifier()
# model = DecisionTreeClassifier()
model = RandomForestClassifier()




scores = cross_val_score(model,x_train,y_train, cv=kfold) # cross_validation
# fit , train 이 다 포함되어있다.



print('scores :' , scores)
'''

# LinearSVC,
scores : [0.01408451 0.         0.02816901 0.         0.        ]

# SVC
scores : [0.         0.         0.02816901 0.01428571 0.        ]

# LogisticRegression()
scores : [0.         0.01408451 0.         0.         0.        ]

# KNeighborsClassifier
scores : [0.         0.         0.         0.01428571 0.        ]

# DecisionTreeClassifier()
scores : [0.         0.         0.         0.01428571 0.        ]

# RandomForestClassifier()
scores : [0.01408451 0.         0.         0.01428571 0.        ]

'''