import numpy as np

def linear_function(self, a: str) -> np.ndarray:
    assert len(a) == self.n, "Длина вектора a должна равняться размерности функции"
    a_vec = np.array(list(map(int, a)), dtype=int)
    f = np.zeros(1 << self.n, dtype=int)

    for x in range(1 << self.n):
        x_vec = np.array([(x >> i) & 1 for i in range(self.n)], dtype=int)
        f[x] = np.bitwise_xor.reduce(a_vec & x_vec)

    return f
