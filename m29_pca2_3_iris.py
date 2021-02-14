# 컬럼의 갯수가 2천개 3천개 되면 속도,자원 손실이 난다 
# 0으로 된 곳 다 잘라낸다
# pca - 압축, 손실 feature_importances - 중요한걸 빼냄  


import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA   #decomposition - 분해 # PCA- 차원

datasets = load_iris()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) #(442, 10) (442,)

# pca = PCA(n_components=10) 
# x2 = pca.fit_transform(x)    #fit +trnasform 합쳐짐
# print(x2)
# print(x2.shape)

# pca_EVR = pca.explained_variance_ratio_
# print(pca_EVR)
# print(sum(pca_EVR)) 
#컬럼? 7개: 0.9479436357350414 -압축률
#8개 : 0.9913119559917797
#9개 : 0.9991439470098977
#10개 : 1.0

pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_) # 누적합 
print('cumsum : ', cumsum) 
d = np.argmax(cumsum >=0.95)+1
print('cumsum >= 0.95 : ', cumsum>=0.95)
print('d : ', d)

import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()