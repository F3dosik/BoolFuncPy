from numba import njit
import numpy as np

a = np.array([0,1,1,1])


@njit
def binom(n, k):
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    res = 1
    for i in range(1, k + 1):
        res = res * (n - i + 1) // i  # n(n-1)...(n-k+1) // k!
    return res
@njit
def generate_bitmask_indices_fast(n, d):
    total = binom(n, d)
    result = np.empty(total, dtype=np.uint32)

    idx = 0
    bits = np.arange(d, dtype=np.uint32)
    while True:
        # Преобразуем позиции в число
        num = 0
        for i in range(d):
            num |= 1 << bits[i]
        result[idx] = num
        idx += 1

        # Найти следующую комбинацию
        for i in range(d - 1, -1, -1):
            if bits[i] != i + n - d:
                break
        else:
            break  # всё сгенерировано

        bits[i] += 1
        for j in range(i + 1, d):
            bits[j] = bits[j - 1] + 1

    return result


@njit
def generate_vectors_fast(n: int, m: int):
    """Генерация всех векторов с весом < m (возвращает массив np.uint32)."""
    total = 0
    for k in range(1, m + 1):
        total += binom(n, k)
    print(total)
    result = np.empty(total, dtype=np.uint32)
    idx = 0

    for k in range(1, m + 1):
        bitmasks = generate_bitmask_indices_fast(n, k)
        for mask in bitmasks:
            result[idx] = mask
            idx += 1
    return result
print(generate_vectors_fast(4,2))