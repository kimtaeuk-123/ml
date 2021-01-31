#2진분류 로지스틱 리그레이션 가능 
import numpy as np
import tensorflow as tf

from sklearn.datasets import load_breast_cancer
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
dataset = load_breast_cancer()

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
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
scale.fit(x_train)
x_train = scale.transform(x_train)
x_test = scale.transform(x_test)

#스케일링하고 for문을 돌리면 오류없이 잘나온다 

models = [LinearSVC(), SVC(), KNeighborsClassifier(), DecisionTreeClassifier(), RandomForestClassifier(), LogisticRegression()]
for i in models:
    model = i
    print('\n ',i)
    score = cross_val_score(model, x_train, y_train, cv = kfold)
    print('score : ', score)




# scores = cross_val_score(model,x_train,y_train, cv=kfold) # cross_validation
# fit , train 이 다 포함되어있다.



# print('scores :' , scores)
'''

# LinearSVC,
scores : [0.59340659 0.95604396 0.84615385 0.85714286 0.93406593]

# SVC
scores : [0.86813187 0.95604396 0.87912088 0.95604396 0.91208791]

# LogisticRegression()
scores : [0.96703297 0.95604396 0.94505495 0.92307692 0.95604396]

# KNeighborsClassifier
scores : [0.9010989  0.93406593 0.92307692 0.9010989  0.94505495]

# DecisionTreeClassifier()
scores : [0.91208791 0.9010989  0.9010989  0.92307692 0.93406593]

# RandomForestClassifier()
scores : [0.94505495 0.96703297 0.94505495 0.95604396 0.94505495]

'''