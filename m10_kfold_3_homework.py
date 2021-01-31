# train test 나눈 다음에 train만 발리데이션 하지 말고, 
# kfold 한 후에 train_test_split 사용
import numpy as np
import tensorflow as tf

from sklearn.datasets import load_iris
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
dataset = load_iris()

x = dataset.data
y = dataset.target


kfold = KFold(n_splits=5, shuffle=True) 
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=77)


# 5등분으로 나눈다 

model = RandomForestClassifier()
for train_index, val_index in kfold.split(x,y): 
    # train fold, val fold 분할
    x_train = x[train_index]  #numpy라 ( 안쓰는건가
    x_test = x[train_index]
    y_train = y[train_index]
    y_test = y[train_index]

    scores = cross_val_score(model, x_train, y_train, cv = kfold)
    print('score : ', scores)

# score :  [1.         1.         0.95833333 1.         0.95833333]
# score :  [1.         0.91666667 1.         0.95833333 0.875     ]
# score :  [1.         1.         0.95833333 0.91666667 0.95833333]
# score :  [0.95833333 0.95833333 0.91666667 0.95833333 0.91666667]
# score :  [0.875      0.95833333 1.         1.         0.875     ]
