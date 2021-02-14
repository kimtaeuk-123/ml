# 컬럼의 갯수가 2천개 3천개 되면 속도,자원 손실이 난다 
# 0으로 된 곳 다 잘라낸다

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA   #decomposition - 분해 # PCA- 차원

datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) #(442, 10) (442,)

pca = PCA(n_components=10) # 10만개 있어도 componet 10 주면 10개로 줄어든다 
# 지금 컬럼의 숫자가 10개라 componesets10개 주면 1 나온다 
x2 = pca.fit_transform(x)    #fit +trnasform 합쳐짐
print(x2)
print(x2.shape)

pca_EVR = pca.explained_variance_ratio_
print(pca_EVR)
print(sum(pca_EVR)) 
#컬럼(? components) 7개: 0.9479436357350414 -압축률
#8개 : 0.9913119559917797
#9개 : 0.9991439470098977
#10개 : 1.0