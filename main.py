from itertools import combinations

import numpy as np
from numba import njit
import numpy.typing as npt
import timeit


class BoolFunc:
    def __init__(self, truth_vector: str):
        size = len(truth_vector)  # 2^n
        # Корректность размера введенной строки: размер > 0 и размер = 2^n;
        if not (size > 0 and (size & (size - 1)) == 0):
            raise ValueError("Длина вектора значений должна быть степенью двойки!")
        # Корректность содержимого веденной строки: строка должна содержать только 0 и 1;
        if set(truth_vector) - {'0', '1'}:
            raise ValueError(f"Ошибка: вектор значений {truth_vector} содержит символы, отличные от '0' и '1'.")

        self.size = size  # Размер вектора значений
        self.n = self.size.bit_length() - 1  # log2(size)

        self.tv = np.frombuffer(truth_vector.encode(), dtype=np.uint8) - ord('0')  # tv - truth vector, вектор значений
        '''Быстрое преобразование строки с бинарными данными ("0110...") в числовой массив для математических операций:
           - .encode() преобразует строку в байтовый объект (тип bytes), используя кодировку по умолчанию (обычно UTF-8);
           - np.frombuffer интерпретирует байты как массив чисел типа uint8;
           - (- ord('0')) - преобразует ASCII-коды цифр в соответствующие числа.'''
        self.tv_packed = np.packbits(self.tv)  # Упаковывает tv в байты (uint8), каждые 8 бит упакованы в один байт.

        # Отложенные вычисления, инициализация при необходимости

        self._sv = None  # sign_vector - знаковый или полярный вектор.
        self._popcount_table8 = None
        self._popcount_table16 = None
        self._w = None  # Вес.
        self._anf = None
        self._walsh_spec = None

    @property
    def sv(self):
        """Вычисление знакового вектора: замена в векторе значений 0 -> 1, 1 -> -1"""
        if self._sv is None:
            self._sv = 1 - 2 * self.tv.astype(np.int8)
        return self._sv

    @property
    def popcount_table8(self):
        """Загрузка popcount_table8 из файла"""
        if self._popcount_table8 is None:
            self._popcount_table8 = np.load("popcount8.npz")["popcount8"]
        return self._popcount_table8

    @property
    def popcount_table16(self):
        """Загрузка popcount_table16 из файла"""
        if self._popcount_table16 is None:
            self._popcount_table16 = np.load("popcount16.npz")["popcount16"]
        return self._popcount_table16

    @property
    def w(self) -> int:
        """Вычисление веса функции.
         Для n >= 16 используется упакованный вектор значений и заранее вычисленный popcount_table8;
         Для n < 16 используется np.sum от вектора значений.
         """
        if self._w is None:
            if self.n >= 16:
                return int(self.popcount_table8[self.tv_packed].sum())
        return int(np.sum(self.tv))

    @property
    def anf(self):
        """Вычисление АНФ функции с помощью быстрого преобразование Мёбиуса"""
        if self._anf is None:
            self._anf = fmt(self.tv, self.n)
        return self._anf

    @property
    def walsh_spec(self):
        """Вычисление спектра с помощью быстрого преобразования Уолша-Адамара"""
        if self._walsh_spec is None:
            self._walsh_spec = fwht(self.sv)
        return self._walsh_spec

    @property
    def is_balanced(self) -> bool:
        return self.w == 1 << (self.n - 1)

    def hamming_distance(self, other: np.ndarray) -> int:
        """Расстояние Хэмминга"""
        if not isinstance(other, np.ndarray):
            raise TypeError("Ожидается NumPy-массив.")
        if other.shape != self.tv.shape:
            raise ValueError("Размерность другого вектора должна совпадать.")

        return int(np.sum(self.tv ^ other))

    def generate_by_popcount(self, m: int):
        """Генерация через popcount (эффективно для n ≤ 16)."""
        return fgbp(self.size, m, self.popcount_table16)

    def generate_by_combinations(self, m: int):
        """Генерация через combinations (эффективно для n > 16)."""
        for k in range(1, m + 1):
            for positions in combinations(range(self.n), k):
                yield sum(1 << i for i in positions)

    def is_correlation_immune(self, order: int) -> bool:
        """Проверяет, является ли функция корреляционно-иммунной заданного порядка.

        Args:
            order (int): Порядок корреляционной иммунности для проверки

        Returns:
            bool: True если функция корреляционно-иммунна порядка 'order', иначе False
        """
        if order < 0:
            return True

        # Для n < 16 используем быстрый метод с popcount
        if self.n < 16:
            indices = self.generate_by_popcount(order)
            return np.all(self.walsh_spec[indices] == 0)

        # Для n >= 16 используем генератор combinations
        for a in self.generate_by_combinations(order):
            if self.walsh_spec[a] != 0:
                return False
        return True

    def walsh_hadamard_transform(self):
        H2 = np.array([[1, 1], [1, -1]], dtype=int)
        H2n = np.eye(self.size, dtype=int)
        for i in range(1, self.n + 1):
            A = np.kron(np.kron(np.eye(1 << (self.n - i), dtype=int), H2), np.eye(1 << (i - 1), dtype=int))
            H2n @= A
        return self.sv @ H2n

    def mobius_transform(self, f):
        g = f.copy()  # Копируем исходный массив
        n = self.n  # Количество битов
        for i in range(n):  # Перебираем все биты
            for a in range(1 << n):  # Перебираем все возможные маски
                if (a >> i) & 1:  # Если i-й бит в маске `a` установлен
                    g[a] ^= g[a ^ (1 << i)]  # Добавляем (XOR) значение подмаски без этого бита
        return g

    # def linear_function(self, a: str):
    #     a = np.array(list(map(int, a)))
    #     f = np.zeros(1 << len(a), dtype=int)
    #     for i in range(self.n):
    #         if a[i]:
    #             f[1 << i] = 1
    #     return f
    #
    # def dot(self, a: str, b: str) -> int:
    #     assert len(a) == self.size, "Размерность скаляра не совпадает с размерностью функции!"
    #     assert all(c in '01' for c in a), f"Ошибка: скаляр {a} содержит символы, отличные от '0' и '1'."
    #     a = np.array(list(map(int, a)))
    #     res = 0
    #     for i in range(self.size):
    #         res ^= (a[i] & self.anf[i])
    #     return res


@njit
def fmt(g, n):
    for i in range(n):
        step = 1 << i
        block_size = step << 1
        for block_start in range(0, 1 << n, block_size):
            for j in range(step):
                a = block_start + step + j
                g[a] ^= g[a - step]
    return g


@njit
def fwht(arr):
    res = arr.copy()
    n = res.shape[0]
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                x = res[j]
                y = res[j + h]
                res[j] = x + y
                res[j + h] = x - y
        h *= 2
    return res


import numpy as np
from numba import njit


@njit
def fgbp(size: int, m: int, popcount_table: np.ndarray) -> np.ndarray:
    """Генерирует массив чисел, где вес (popcount) <= m.

    Args:
        size: Максимальное число (2^n).
        m: Максимальный вес.
        popcount_table: Массив с предвычисленными весами (popcount).

    Returns:
        np.ndarray: Массив подходящих чисел.
    """
    # Сначала считаем количество подходящих элементов
    count = 0
    for a in range(1, size):
        if popcount_table[a] <= m:
            count += 1
    # Создаём массив нужного размера
    result = np.empty(count, dtype=np.uint32)
    idx = 0
    for a in range(1, size):
        if popcount_table[a] <= m:
            result[idx] = a
            idx += 1
    return result


if __name__ == "__main__":
    # Заранее вычисленные и сохраненные значения popcount_table*
    # popcount_table8 - массив, который хранит количество единичных битов для всех возможных 8-битных чисел (от 0 до 255).
    # popcount_table8 = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)
    # np.savez_compressed("popcount8.npz", popcount8=popcount_table8)
    # # popcount_table16 - массив, который хранит количество единичных битов для всех возможных 8-битных чисел (от 0 до 255).
    # popcount_table16 = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)
    # np.savez_compressed("popcount16.npz", popcount16=popcount_table16)

    # f = BoolFunc("1" * (1 << 21))
    # # print(f'Вектор значений: {f.tv}')
    # # print(f'Вес функции: {f.w}')
    # # print(f'Размер вектора значений: {f.size}')
    # # print(f'Количество переменных: {f.n}')
    # # print(f'Сбалансированность: {f.is_balanced()}')
    # # g = BoolFunc("1000")
    # # print(f'Расстояние Хэмминга: {f.hamming_distance(g)}')
    # # time1 = timeit.timeit(lambda: f.walsh_hadamard_transform(), globals=globals(), number=100)
    # # dummy = np.array([0, 1, 1, 1], dtype=np.int64)
    # # fwht(dummy)  # JIT-компиляция происходит здесь
    # # time2 = timeit.timeit(lambda: f.fast_walsh_hadamard_transform, globals=globals(), number=100)
    # # print("Функция 1: среднее время =", time1 / 100)
    # # print("Функция 2: среднее время =", time2 / 100)
    # time1 = timeit.timeit(lambda: f.mobius_transform(f.tv), globals=globals(), number=100)
    # dummy = np.array([0, 1, 1, 1], dtype=np.int64)
    # fmt(dummy, 2)  # JIT-компиляция происходит здесь
    # time2 = timeit.timeit(lambda: f.fast_mobius_transform(), globals=globals(), number=100)
    # print("Функция 1: среднее время =", time1 / 100)  # Функция 1: среднее время = 4.8954639914499785
    # print("Функция 2: среднее время =", time2 / 100)  # Функция 2: среднее время = 0.018933899059993563
    # fgbp(4, 2, f.popcount_table16)  # JIT-компиляция происходит здесь
    # time1 = timeit.timeit(lambda: f.generate_by_popcount(4), globals=globals(), number=100)
    #
    #
    # def run_generate():
    #     for _ in f.generate_by_combinations(4):
    #         pass
    #
    #
    # time2 = timeit.timeit(lambda: run_generate(), globals=globals(), number=100)
    # print("Функция 1: среднее время =", time1 / 100)
    # print("Функция 2: среднее время =", time2 / 100)
    # print(f"Ускорение: {time1 / time2}x")
    f = BoolFunc("11000110")
    print(f.generate_by_popcount(1))
    print(f.walsh_spec)
    print(f.is_correlation_immune(1))
