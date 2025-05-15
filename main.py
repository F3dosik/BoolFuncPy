import numpy as np


class BoolFunc:
    def __init__(self, truth_vector: str):
        size = len(truth_vector)

        if not (size > 0 and (size & (size - 1)) == 0):
            raise ValueError("Длина вектора значений должна быть степенью двойки!")

        if not all(c in '01' for c in truth_vector):
            raise ValueError("Строка содержит другие символы отличные от '0' и '1'!")

        self.tv = np.array(list(map(int, truth_vector)))
        self.size = size
        self.n = self.size.bit_length() - 1
        self.w = sum(self.tv)
        self._anf = None

    @property
    def anf(self):
        if self._anf is None:
            self._anf = self.mobius_transform()
        return self._anf

    def mobius_transform(self):
        g = []
        for a in range(self.n):
            g_a = 0
            for x in range(a + 1):
                if a | x == a:
                    g_a ^= self.tv[x]
            g.append(g_a)
        return np.array(g)

    def is_balanced(self):
        return self.w == 2 ** (self.n - 1)

    def hamming_distance(self, other: 'BoolFunc'):
        return sum(self.tv ^ other.tv)

    def walsh_hadamard_transform(self):
        return 1 << self.n


if __name__ == "__main__":
    f = BoolFunc("01010011")
    # print(f'Вектор значений: {f.tv}')
    # print(f'Вес функции: {f.w}')
    # print(f'Размер вектора значений: {f.size}')
    # print(f'Количество переменных: {f.n}')
    # print(f'Сбалансированность: {f.is_balanced()}')
    # g = BoolFunc("1000")
    # print(f'Расстояние Хэмминга: {f.hamming_distance(g)}')
    print(f.mobius_transform())
