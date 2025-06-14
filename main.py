import numpy as np
from numba import njit, prange


class BoolFunc:
    def __init__(self, truth_vector: str):
        # Корректность содержимого веденной строки: строка должна содержать только 0 и 1;
        if set(truth_vector) - {'0', '1'}:
            raise ValueError(f"Ошибка: вектор значений {truth_vector} содержит символы, отличные от '0' и '1'.")
        size = len(truth_vector)  # 2^n
        # Корректность размера введенной строки: размер > 0 и размер = 2^n;
        if not (size > 0 and (size & (size - 1)) == 0):
            raise ValueError("Длина вектора значений должна быть степенью двойки!")

        self.size = size  # Размер вектора значений
        self.n = self.size.bit_length() - 1  # log2(size)

        self.tv = np.frombuffer(truth_vector.encode(), dtype=np.uint8) - ord('0')  # tv - truth vector, вектор значений
        '''Быстрое преобразование строки с бинарными данными ("0110...") в числовой массив для математических операций:
           - .encode() преобразует строку в байтовый объект (тип bytes), используя кодировку по умолчанию (обычно UTF-8);
           - np.frombuffer интерпретирует байты как массив чисел типа uint8;
           - (- ord('0')) - преобразует ASCII-коды цифр в соответствующие числа.'''

        # Упаковывает tv в байты (uint8), каждые 8 бит упакованы в один байт.
        self.tv_packed = np.packbits(self.tv, bitorder='big')

        # Отложенные вычисления, инициализация при необходимости
        self._sv = None  # sign_vector - знаковый или полярный вектор.
        self._popcount_table8 = None
        self._popcount_table16 = None
        self._w = None  # Вес.
        self._anf = None
        self._walsh_spec = None

    @classmethod
    def from_array(cls, arr: np.ndarray):
        if arr.dtype != np.uint8 or not np.isin(arr, [0, 1]).all():
            raise ValueError("Массив должен содержать только 0 и 1 (uint8).")
        size = arr.size
        if size == 0 or (size & (size - 1)) != 0:
            raise ValueError("Размер массива должен быть больше 0 и степенью двойки.")

        obj = cls.__new__(cls)  # Низкоуровневый способ создать объект класса, в обход  __init__ методу.
        obj.size = size
        obj.n = size.bit_length() - 1
        obj.tv = arr
        obj.tv_packed = np.packbits(arr)
        obj._sv = obj._popcount_table8 = obj._popcount_table16 = obj._w = obj._anf = obj._walsh_spec = None
        return obj

    @classmethod
    def from_packed(cls, packed: np.ndarray, n: int = None):
        bits = np.unpackbits(packed, bitorder='big')
        if n is not None:
            size = 1 << n
            if bits.size < size:
                raise ValueError(f"Недостаточно данных: требуется {size} бит, получено {bits.size}")
            bits = bits[:size]
        return cls.from_array(bits)

    @classmethod
    def from_bytes(cls, byte_data: bytes, n: int = None):
        packed = np.frombuffer(byte_data, dtype=np.uint8)
        return cls.from_packed(packed, n)

    @classmethod
    def random(cls, n: int):
        size = 1 << n

        # Объект генератора псевдослучайных чисел, основанный на современном алгоритме PCG64
        rng = np.random.default_rng()
        tv = rng.integers(0, 2, size=size, dtype=np.uint8)  # Случайный массив 0 и 1 размера size
        return cls.from_array(tv)

    def save_to_bin(self, filename: str) -> None:
        """
        Сохраняет булеву функцию в бинарный файл:
        [1 байт n][упакованные значения tv_packed]
        """
        with open(filename, 'wb') as f:
            f.write(bytes([self.n]))
            f.write(self.tv_packed.tobytes())

    @classmethod
    def load_from_bin(cls, filename: str) -> 'BoolFunc':
        with open(filename, 'rb') as f:
            n = int.from_bytes(f.read(1), 'big')
            packed = np.frombuffer(f.read(), dtype=np.uint8)
        return cls.from_packed(packed, n=n)

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
    def w(self):
        """Вычисление веса функции."""
        if self._w is None:
            if self.n >= 16:
                self._w = int(self.popcount_table8[self.tv_packed].sum())
            else:
                self._w = int(np.sum(self.tv))
        return self._w

    @property
    def anf(self):
        """Вычисление АНФ функции с помощью быстрого преобразование Мёбиуса"""
        if self._anf is None:
            fmt([0, 1, 1, 0], 2)
            self._anf = self.mobius_transform(self.tv.copy(), self.n)
        return self._anf

    @property
    def walsh_spec(self):
        """Вычисление спектра с помощью быстрого преобразования Уолша-Адамара"""
        if self._walsh_spec is None:
            self._walsh_spec = fwht(self.sv.astype(np.int64))
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

    @staticmethod
    def mobius_transform(f, n):
        return fmt(f, n)

    @property
    def nonlinearity(self):
        if self.n > 16:
            max_abs = max_abs_par(self.walsh_spec)[0][0]
        else:
            max_abs = max_abs_seq(self.walsh_spec)[0][0]
        return (1 << (self.n - 1)) - (abs(max_abs) >> 1)

    @property
    def best_affine_approximation(self):
        max_abs = max_abs_par(self.walsh_spec) if self.n >= 20 else max_abs_seq(self.walsh_spec)
        min_dist = self.size
        best_appr = None
        for coeff, index in max_abs:
            a = int_to_bitarray(index, self.n)
            b_bit = 0 if coeff > 0 else 1
            anf = self.linear_function(a)
            candidate = self.mobius_transform(anf, self.n) ^ b_bit
            dist = self.hamming_distance(candidate)
            if dist < min_dist:
                min_dist = dist
                best_appr = candidate

        return best_appr

    @staticmethod
    def linear_function(a: np.ndarray):
        packed = linear_function_bitpacked_seq(a) if a.shape[0] < 20 else linear_function_bitpacked_par(a)
        return np.unpackbits(packed, bitorder='little')[:1 << a.shape[0]]

    @property
    def algebraic_degree(self):
        if self.w is not None:
            if self.w % 2 != 0:
                return self.n
            if self.n <= 12:
                d_max = upper_deg(self.w, self.n)
                return max(i.bit_count() for i in range(1 << self.n) if self.anf[i] and i.bit_count() <= d_max)
            elif self.n > 12:
                degree_candidates = get_degree_candidates(self.w, self.n)
                degree_index_sets = [generate_bitmask_indices_fast(self.n, d) for d in degree_candidates]
                for i in range(len(degree_index_sets) - 1, -1, -1):
                    for ind in degree_index_sets[i]:
                        if self.anf[ind]:
                            return degree_candidates[i]
        else:
            if self.n >= 12:
                return algebraic_degree_anf(self.anf)
        return max(i.bit_count() for i in range(self.size) if self.anf[i])

    def is_correlation_immune(self, order: int) -> bool:
        """Проверяет, является ли функция корреляционно-иммунной заданного порядка."""
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

    def generate_by_popcount(self, m: int):
        """Генерация булевых векторов с весом <= m через popcount (эффективно для n ≤ 16)."""
        return fgbp(self.size, m, self.popcount_table16)

    def generate_by_combinations(self, m: int):
        """Генерация булевых векторов с весом <= m через combinations (эффективно для n > 16)."""
        return generate_vectors_fast(self.n, m)

    def boolean_derivative(self, a: int):
        if a >= self.size:
            raise ValueError(f"Некорректное направление: требуется направление размером {self.n} бит.")
        if a == 0:
            return np.zeros(self.size, dtype=np.uint8)
        x_a = np.arange(self.size) ^ a
        return np.array([self.tv[i] ^ self.tv[x_a[i]] for i in range(self.size)])

    def visualize_anf(self):
        terms = []
        if not np.any(self.tv):
            return 0
        for i, coeff in enumerate(self.anf):
            if coeff == 0:
                continue
            if i == 0:
                terms.append("1")
                continue
            monomial = []
            for j in range(self.n):
                if (i >> j) & 1:
                    monomial.append(f"x{j + 1}")
            terms.append("".join(monomial))
        return " ⊕ ".join(terms)

    @property
    def find_fictive_vars(self):
        var_flags = 0
        for i in range(self.anf.shape[0]):
            if self.anf[i]:
                var_flags |= i
        fictive_vars = [i for i in range(self.n) if not (var_flags & (1 << i))]
        return fictive_vars if fictive_vars else None


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


@njit
def fgbp(size: int, m: int, popcount_table: np.ndarray) -> np.ndarray:
    """Генерирует массив чисел, где вес (popcount) <= m."""
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


@njit
def max_abs_seq(arr: np.ndarray):
    max_val = np.abs(arr[0])
    result = [(arr[0], 0)]
    for i in range(1, arr.shape[0]):
        val = arr[i]
        abs_val = abs(val)
        if abs_val > max_val:
            max_val = abs_val
            result = [(val, i)]
        elif abs_val == max_val:
            result.append((val, i))
    return result


@njit(parallel=True)
def max_abs_par(arr: np.ndarray):
    max_val = np.abs(arr[0])
    result = [(arr[0], 0)]
    for i in prange(1, arr.shape[0]):
        val = arr[i]
        abs_val = abs(val)
        if abs_val > max_val:
            max_val = abs_val
            result = [(val, i)]
        elif abs_val == max_val:
            result.append((val, i))
    return result


@njit
def linear_function_bitpacked_seq(a):
    n = a.shape[0]
    bit_len = 1 << n
    num_bytes = (bit_len + 7) // 8
    ax = np.zeros(num_bytes, dtype=np.uint8)
    for i in range(n):
        if a[i]:
            index = 1 << i
            ax[index // 8] |= 1 << (index % 8)
    return ax


@njit(parallel=True)
def linear_function_bitpacked_par(a):
    n = a.shape[0]
    bit_len = 1 << n
    num_bytes = (bit_len + 7) // 8
    ax = np.zeros(num_bytes, dtype=np.uint8)
    for i in prange(n):
        if a[i]:
            index = 1 << i
            ax[index // 8] |= 1 << (index % 8)
    return ax


@njit
def int_to_bitarray(x: int, n: int) -> np.ndarray:
    out = np.empty(n, dtype=np.uint8)
    for i in range(n):
        out[n - 1 - i] = (x >> i) & 1
    return out


@njit
def algebraic_degree_anf(anf):
    max_deg = 0
    for i in range(anf.shape[0]):
        if anf[i]:
            deg = 0
            x = i
            while x:
                x &= x - 1
                deg += 1
            if deg > max_deg:
                max_deg = deg
    return max_deg


@njit
def upper_deg(w, n):
    deg = 0
    for d in range(1, n + 1):
        low = 1 << (n - d)
        high = (1 << n) - low
        if low <= w <= high:
            deg = d
    return deg


@njit
def get_degree_candidates(w, n):
    candidates = np.empty(n, dtype=np.uint8)
    i = 0
    for d in range(1, n + 1):
        low = 1 << (n - d)
        high = (1 << n) - low
        if low <= w <= high:
            candidates[i] = d
            i += 1
    return candidates[:i]


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
    """Генерация всех векторов с весом <= m (возвращает массив np.uint32)."""
    total = 0
    for k in range(1, m + 1):
        total += binom(n, k)
    result = np.empty(total, dtype=np.uint32)
    idx = 0

    for k in range(1, m + 1):
        bitmasks = generate_bitmask_indices_fast(n, k)
        for mask in bitmasks:
            result[idx] = mask
            idx += 1
    return result
