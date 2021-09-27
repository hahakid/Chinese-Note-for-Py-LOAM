import numpy as np

a=np.array([[-0.50323344],[-0.09552087],[-0.6340574]])
b=a.reshape(1,-1)[0]
print(np.einsum('i,i', b,b))