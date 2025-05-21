import numpy as np

a = np.array([0,0,0,0,0,1,0,0, 0,0,0,0,0,1,0,0])
a_1 = np.packbits(a, bitorder='big')
print(a_1)
print(a_1.size)