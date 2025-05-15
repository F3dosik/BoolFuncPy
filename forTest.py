def mobius_transform(self):
    g = self.tv.copy()  # Копируем исходный массив
    n = self.n          # Количество битов
    for i in range(n):  # Перебираем все биты
        for a in range(1 << n):  # Перебираем все возможные маски
            if (a >> i) & 1:     # Если i-й бит в маске `a` установлен
                g[a] ^= g[a ^ (1 << i)]  # Добавляем (XOR) значение подмаски без этого бита
    return g