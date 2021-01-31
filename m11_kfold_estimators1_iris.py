from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
# from sklearn.utils.testing import all_estimators
from sklearn.utils._testing import all_estimators
import warnings


from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore') # warning 무시하겠다! , 버전 안맞거나 difalut가 안맞아서 등등 

dataset = load_iris()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=44)

kfold = KFold(n_splits=5, shuffle=True)

allAlgorithms = all_estimators(type_filter = 'classifier') #분류모델

for (name, algorithm) in allAlgorithms:  # allAlgorithms 에서 인자가 2개 나온다 .
    try:                                 
        model = algorithm()

        score = cross_val_score(model, x_train, y_train, cv=kfold) #cv=5 
        # model.fit(x_train, y_train)
        # y_pred = model.predict(x_test)
        print(name, '의 정답률 : \n', score)
    except: #예외가 발생하면 
        print(name, '은 없는 놈!')
        

# import sklearn 
# print(sklearn.__version__) 버전 확인 