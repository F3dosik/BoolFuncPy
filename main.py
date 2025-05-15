import numpy as np
import numpy.typing as npt


class BoolFunc:
    def __init__(self, truth_vector: str):
        size = len(truth_vector)

        if not (size > 0 and (size & (size - 1)) == 0):
            raise ValueError("Длина вектора значений должна быть степенью двойки!")

        if not all(c in '01' for c in truth_vector):
            raise ValueError(f"Ошибка: вектор значений {truth_vector} содержит символы, отличные от '0' и '1'.")

        self.tv = np.array(list(map(int, truth_vector)))
        self.size = size
        self.n = self.size.bit_length() - 1
        self.w = sum(self.tv)
        self._anf = None
        self._walsh_spec = None

    @property
    def anf(self):
        if self._anf is None:
            self._anf = self.mobius_transform(self.tv)
        return self._anf

    def mobius_transform(self, f):
        g = f.copy()  # Копируем исходный массив
        n = self.n  # Количество битов
        for i in range(n):  # Перебираем все биты
            for a in range(1 << n):  # Перебираем все возможные маски
                if (a >> i) & 1:  # Если i-й бит в маске `a` установлен
                    g[a] ^= g[a ^ (1 << i)]  # Добавляем (XOR) значение подмаски без этого бита
        return g

    def linear_function(self, a: str):
        a = np.array(list(map(int, a)))
        f = np.zeros(1 << len(a), dtype=int)
        for i in range(self.n):
            if a[i]:
                f[1 << i] = 1
        return f

    def dot(self, a: str, b: str) -> int:
        assert len(a) == self.size, "Размерность скаляра не совпадает с размерностью функции!"
        assert all(c in '01' for c in a), f"Ошибка: скаляр {a} содержит символы, отличные от '0' и '1'."
        a = np.array(list(map(int, a)))
        res = 0
        for i in range(self.size):
            res ^= (a[i] & self.anf[i])
        return res

    def is_balanced(self):
        return self.w == 2 ** (self.n - 1)

    def hamming_distance(self, other):
        return sum(self.tv ^ other)

    @property
    def walsh_spec(self):
        if self._walsh_spec is None:
            self._walsh_spec = self.walsh_hadamard_transform()
        return self._walsh_spec

    def walsh_hadamard_transform(self):
        spec = []
        for a in range(1 << self.n):
            bits = np.array([(a >> i) & 1 for i in reversed(range(self.n))])
            spec.append((1 << self.n) - 2 * self.hamming_distance(self.linear_function()))
        return np.array(spec)


if __name__ == "__main__":
    f = BoolFunc("0111")
    # print(f'Вектор значений: {f.tv}')
    # print(f'Вес функции: {f.w}')
    # print(f'Размер вектора значений: {f.size}')
    # print(f'Количество переменных: {f.n}')
    # print(f'Сбалансированность: {f.is_balanced()}')
    # g = BoolFunc("1000")
    # print(f'Расстояние Хэмминга: {f.hamming_distance(g)}')
    print(f.linear_function("11"))
