import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 10, 0.1)
y = np.sin(x) #numpy에서 sin 제공 

plt.plot(x,y)
plt.show()
