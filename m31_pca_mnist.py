import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA   #decomposition - 분해 # PCA- 차원
(x_train, _), (x_test, _) =mnist.load_data() # _ -> 안하겠다.

x = np.append(x_train, x_test, axis=0)

# print(x.shape) #(70000, 28, 28) 합쳐짐 

# 실습 
# pca를 통해 0.95 이상인거 몇개?
# pca 배운거 다 집어넣고 확인 

x = x.reshape(70000, 784)


pca = PCA(n_components = 730)
x2 = pca.fit_transform(x)
pca_EVR = pca.explained_variance_ratio_
print(pca_EVR)
print(sum(pca_EVR)) # 156 일때 0.95 넘는다 
pca = PCA()
pca.fit(x)

cumsum = np.cumsum(pca.explained_variance_ratio_) # 누적합 
# print('cumsum : ', cumsum) 
d = np.argmax(cumsum >=0.95)+1
# print('cumsum >= 0.95 : ', cumsum>=0.95)
# print('d : ', d) #154

