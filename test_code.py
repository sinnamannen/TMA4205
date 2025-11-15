import numpy as np

u = np.array([[1, 0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1,0], [1, 0, 1, 0, 1, 0,1], [0, 1, 0, 1, 0, 1,0], [1, 0, 1, 0, 1, 0,1], [0, 1, 0, 1, 0, 1,0 ]])
u = np.pad(u, pad_width=1, mode='constant', constant_values=0)
print(u)
print(u[1:-1:2,1:-1:2])
print(u[2:-1:2,2:-1:2])

print(4**(-1))




