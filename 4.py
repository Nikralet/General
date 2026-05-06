import numpy as np
import matplotlib.pyplot as plt

# Параметры модели
N = 1000                # число спинов
J = 1.0                 # энергия взаимодействия (в единицах J)

# ------------------------------------------------------------
# Функция одного шага Метрополиса (случайный спин)
# ------------------------------------------------------------
def metropolis_step(spins, T):
    """
    Выполняет один шаг алгоритма Метрополиса для случайно выбранного спина.
    Возвращает изменение суммы спинов.
    """
    i = np.random.randint(N)
    left = (i - 1) % N
    right = (i + 1) % N
    old_spin = spins[i]
    # Изменение энергии при перевороте (в единицах J)
    deltaE = 2.0 * old_spin * (spins[left] + spins[right])
    # Условие переворота
    if deltaE <= 0.0 or np.random.random() < np.exp(-deltaE / T):
        spins[i] = -old_spin
        return -2 * old_spin          # изменение суммы спинов
    return 0

# ------------------------------------------------------------
# 1. Релаксация среднего спина при фиксированной температуре
# ------------------------------------------------------------
T_fixed = 2.0                      # безразмерная температура kT/J
sweeps_relax = 2000                 # число полных проходов по системе

# Начальная конфигурация: все спины вверх
spins = np.ones(N, dtype=int)
M = np.mean(spins)                  # текущее значение среднего спина

# Массивы для записи
sweep_numbers = []
M_values = []

for sweep in range(sweeps_relax):
    for _ in range(N):
        delta_sum = metropolis_step(spins, T_fixed)
        M += delta_sum / N           # обновляем средний спин
    sweep_numbers.append(sweep)
    M_values.append(M)

# Построение графика релаксации
plt.figure(figsize=(10, 5))
plt.plot(sweep_numbers, M_values, lw=1)
plt.xlabel('Номер прохода (sweep)')
plt.ylabel('Средний спин M')
plt.title(f'Релаксация среднего спина при T = {T_fixed}')
plt.grid(True, alpha=0.3)
plt.show()

# ------------------------------------------------------------
# 2. Зависимость равновесного среднего спина от температуры
# ------------------------------------------------------------
# Для каждой температуры моделируем одну реализацию с начальным состоянием "все вверх"
# и усредняем спин по времени после термализации.
# Такой подход даёт ненулевое значение среднего спина, которое убывает с ростом T,
# поскольку система не успевает полностью перевернуться за время моделирования.
# (В строгом смысле равновесное среднее <s> в 1D равно нулю при T>0,
# но для конечных систем и конечного времени наблюдения получается именно такая картина.)

T_min, T_max, T_step = 0.2, 6.0, 0.2
temperatures = np.arange(T_min, T_max + 1e-9, T_step)
eq_sweeps = 500                      # проходов на термализацию
prod_sweeps = 1000                    # проходов для сбора статистики

avg_M = []                            # средний спин для каждой T

for T in temperatures:
    # Начальная конфигурация: все спины вверх
    spins = np.ones(N, dtype=int)
    # Термализация
    for _ in range(eq_sweeps):
        for _ in range(N):
            metropolis_step(spins, T)
    # Сбор статистики (усреднение спина по времени)
    M_accum = 0.0
    for _ in range(prod_sweeps):
        for _ in range(N):
            metropolis_step(spins, T)
        M_accum += np.mean(spins)      # добавляем текущее значение (без модуля!)
    avg_M.append(M_accum / prod_sweeps)
    print(f"T = {T:.2f}, <M> = {avg_M[-1]:.4f}")

# Построение графика <M> от температуры
plt.figure(figsize=(10, 5))
plt.plot(temperatures, avg_M, 'o-', markersize=4)
plt.xlabel('Температура $k_B T / J$')
plt.ylabel('Средний спин $\\langle M \\rangle$')
plt.title('Зависимость среднего спина от температуры (начальное состояние: все вверх)')
plt.grid(True, alpha=0.3)
plt.show()
