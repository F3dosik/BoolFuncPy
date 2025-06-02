from numba import njit
import numpy as np
# if __name__ == "__main__":
#     pass
    # Заранее вычисленные и сохраненные значения popcount_table*
    # popcount_table8 - массив, который хранит количество единичных битов для всех возможных 8-битных чисел (от 0 до 255).
    # popcount_table8 = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)
    # np.savez_compressed("popcount8.npz", popcount8=popcount_table8)
    # # popcount_table16 - массив, который хранит количество единичных битов для всех возможных 16-битных чисел (от 0 до 65535).
    # popcount_table16 = np.array([bin(i).count('1') for i in range(65536)], dtype=np.uint8)
    # np.savez_compressed("popcount16.npz", popcount16=popcount_table16)

    # ------------------------------------------Проверка from_str-------------------------------------------------------
    # f_from_str = BoolFunc("0100")
    # print(f"{f_from_str.tv} - вектор значений через str")  # -> [0 1 0 0] - вектор значений через str
    # ------------------------------------------Проверка from_arr-------------------------------------------------------
    # f_from_arr = BoolFunc.from_array(np.array([0, 1, 0, 0], dtype=np.uint8))
    # print(f"{f_from_arr.tv} - вектор значений через arr")  # -> [0 1 0 0] - вектор значений через arr
    # ------------------------------------------Проверка from_packed----------------------------------------------------
    # f_from_packed = BoolFunc.from_packed(np.array([64], dtype=np.uint8), 2)  # 01000000 = 2^6 = 64 bitorder = 'big'
    # print(f"{f_from_packed.tv} - вектор значений через packed arr")  # -> [0 1 0 0] - вектор значений через packed arr
    #
    # f_from_packed = BoolFunc.from_packed(np.array([53], dtype=np.uint8))  # 00110101 = 2^5 + 2^4 + 2^2 + 1 = 53
    # print(f"{f_from_packed.tv} - вектор значений через packed arr")
    # # -> [0 0 1 1 0 1 0 1] - вектор значений через packed arr
    #
    # f_from_packed = BoolFunc.from_packed(np.array([38, 168], dtype=np.uint8))
    # print(f"{f_from_packed.tv} - вектор значений через packed arr")
    # # -> [0 0 1 0 0 1 1 0 1 0 1 0 1 0 0 0] - вектор значений через packed arr
    # ------------------------------------------Проверка from_bytes-----------------------------------------------------
    # f_from_bytes = BoolFunc.from_bytes(bytes([0b01000000]), 2)  # Используем bitorder = 'big', сначала младшие биты.
    # print(f"{f_from_bytes.tv} - вектор значений через bytes")  # -> [0 1 0 0] - вектор значений через bytes
    #
    # f_from_bytes = BoolFunc.from_bytes(bytes([0b00110101]), 3)  # Используем bitorder = 'big', сначала младшие биты.
    # print(f"{f_from_bytes.tv} - вектор значений через bytes")  # -> [0 0 1 1 0 1 0 1] - вектор значений через bytes
    #
    # f_from_bytes = BoolFunc.from_bytes(bytes([0b00100110,0b10101000]))  # Используем bitorder = 'big', сначала младшие биты.
    # print(f"{f_from_bytes.tv} - вектор значений через bytes")
    # # -> [0 0 1 0 0 1 1 0 1 0 1 0 1 0 0 0] - вектор значений через bytes
    # ------------------------------Проверка генераций случайных функций от n переменных--------------------------------
    # f_random = BoolFunc.random(3)
    # print(f_random.tv) # -> [0 0 1 1 1 1 1 1]
    # ------------------------------------Сохранение функций в *.bin файл ----------------------------------------------
    # f_for_save = BoolFunc("0110")
    # f_for_save.save_to_bin("xor.bin")
    # f_for_load = BoolFunc.load_from_bin("xor.bin")
    # print(f_for_load.tv)
    # print(f_for_load.anf)
    #
    # f_for_save = BoolFunc.from_bytes(bytes([0b00100110,0b10101000]))
    # f_for_save.save_to_bin("func_4.bin")
    # f_for_load = BoolFunc.load_from_bin("func_4.bin")
    # print(f_for_load.tv)
    # ---------------------------------Проверка веса .bin файла функции от 30 переменных--------------------------------
    # f_for_save_size = BoolFunc.random(30)
    # f_for_save_size.save_to_bin("func_30.bin")
    # ls -lah -> -rw-rw-r-- 1 fedos fedos 129M мая 22 14:54 func_30.bin,
    # Так как 2^30/2^3 = 128 Мб
    # ---------------------------------Проверка скорости загрузки функции от 30 переменных------------------------------
    # f_for_save_size = BoolFunc.random(30)
    # f_for_save_size.save_to_bin("func_30.bin")
    # time_for_load = timeit.timeit(lambda: BoolFunc.load_from_bin("func_30.bin"), globals=globals(), number=100)
    # print("Инициализации функции от 30 переменных: среднее время =", time_for_load / 100)
    # Инициализации функции от 30 переменных: среднее время = 5.785267059630005
    # ----------------------------------Проверка скорости функций-------------------------------------------------------
    # time_for_random = timeit.timeit(lambda: BoolFunc.random(30), globals=globals(), number=100)
    # print("Генерация функции от 30 переменных: среднее время =", time_for_random / 100)
    # Генерация функции от 30 переменных: среднее время = 7.411581638700008
# if __name__ == "__main__":
#     f = BoolFunc.random(3)
#     print(f'Вектор значений случайной функции от 3 перменных - {f.tv}')
#     print(f'Вектор АНФ случайной функции от 3 перменных - {f.anf}')
#     print(f'АНФ представление случайной функции от 3 перменных - {f.visualize_anf()}')
#     print("Производные по всем направлениям:")
#     for i in range(8):
#         print(f"Производная по направлению {bin(i)}")
#         d = BoolFunc.from_array(f.boolean_derivative(i))
#         print(d.visualize_anf())
if __name__ == "__main__":
    f = BoolFunc("01111010")
    print(f'Вектор значений: {f.tv}')
    print(f'Знаковый вектор: {f.sv}')
    print(f'Вес: {f.w}')
    print(f'Вектор АНФ: {f.anf}')
    print(f'АНФ: {f.visualize_anf()}')
    print(f'Спектр: {f.walsh_spec}')
    print(f'Сбалансированность: {f.is_balanced}')
    print(f'Нелинейность: {f.nonlinearity}')
    print(f'Лучшее аффинное приближение: {f.best_affine_approximation}')
    g = BoolFunc.from_array(f.best_affine_approximation)
    print(f'АНФ аффинного приближения: {g.visualize_anf()}')
    print(f'Алгебраическая степень: {f.algebraic_degree}')
    print(f'Корреляционная иммунность порядка 2: {f.is_correlation_immune(2)}')
    print(f'Фиктивные переменные: {f.find_fictive_vars}')
    print("Производные по всем направлениям:")
    print("-----------------------------------------")
    for i in range(8):
        print(f"Производная по направлению {bin(i)}")
        d = BoolFunc.from_array(f.boolean_derivative(i))
        print(d.visualize_anf())
    print("-----------------------------------------")
