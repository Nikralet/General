import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import eig
from matplotlib.animation import FuncAnimation

# ==================== ПАРАМЕТРЫ СИСТЕМЫ ====================
L1 = 0.5  # длина первого маятника, м
L2 = 1.0  # длина второго маятника, м
m1 = 1.0  # масса первого груза, кг
m2 = 1.0  # масса второго груза, кг
g = 9.81  # ускорение свободного падения, м/с^2
k = 5.0  # жёсткость пружины, Н/м
d = 0.5  # расстояние между точками подвеса, м

# Начальные условия (в радианах)
theta1_0 = np.radians(30)  # начальный угол первого
theta2_0 = np.radians(-20)  # начальный угол второго
omega1_0 = 0.0  # начальная угловая скорость первого
omega2_0 = 0.0  # начальная угловая скорость второго

# Временной интервал
t_span = (0, 20)
t_eval = np.linspace(0, 20, 2000)  # больше точек для нелинейной динамики

"""Переключатель режима: 0 - линейный, 1 - нелинейный (слабая нелинейность)"""
nonlinear = 1  # измените на 0 для линейного приближения


# ==================== ЛИНЕАРИЗОВАННАЯ СИСТЕМА ====================
# Вычисляем матрицы масс и жёсткости для малых колебаний

def compute_linear_matrices():
    """Возвращает матрицы M и K для линеаризованной системы, а также собственные частоты и векторы."""
    # Расстояние между грузами в положении равновесия
    dx0 = d
    dy0 = L1 - L2  # y2 - y1 = (-L2) - (-L1) = L1 - L2
    l0 = np.hypot(dx0, dy0)

    # Матрица масс (диагональная)
    M = np.array([[m1 * L1 ** 2, 0],
                  [0, m2 * L2 ** 2]])

    # Гравитационная часть матрицы жёсткости (диагональная)
    K_grav = np.array([[m1 * g * L1, 0],
                       [0, m2 * g * L2]])

    # Упругая часть: вычисляем элементы K_spring
    # Вспомогательные величины
    alpha = (l0 - d) / l0  # (l0 - d)/l0
    beta = 1 / l0 ** 2  # 1/l0^2

    # Элементы матрицы упругой жёсткости (вывод в приложении)
    K11_s = k * (alpha * L1 ** 2 + beta * (dx0 * L1) ** 2)
    K12_s = -k * (alpha * L1 * L2 + beta * dx0 * L1 * (dx0 + L2))  # K12 = K21
    K22_s = k * (alpha * L2 ** 2 + beta * (dx0 + L2) ** 2)

    K_spring = np.array([[K11_s, K12_s],
                         [K12_s, K22_s]])

    # Полная матрица жёсткости
    K = K_grav + K_spring

    # Решение обобщённой проблемы собственных значений: (K - ω² M) v = 0
    eigvals, eigvecs = eig(np.linalg.inv(M) @ K)
    omega_sq = np.real(eigvals)
    # Сортировка по возрастанию частоты
    idx = np.argsort(omega_sq)
    omega_sq = omega_sq[idx]
    eigvecs = eigvecs[:, idx]

    # Собственные частоты (берём положительные корни)
    omega = np.sqrt(np.abs(omega_sq))

    # Нормировка собственных векторов (для удобства)
    for i in range(2):
        eigvecs[:, i] = eigvecs[:, i] / np.linalg.norm(eigvecs[:, i])
        # Устанавливаем знак так, чтобы первая компонента была положительной
        if eigvecs[0, i] < 0:
            eigvecs[:, i] = -eigvecs[:, i]

    return M, K, omega, eigvecs


# Получаем линейные характеристики
M_lin, K_lin, omega_lin, v_lin = compute_linear_matrices()

print("ЛИНЕЙНАЯ СИСТЕМА (нормальные моды)")
print("=" * 50)
print(f"Собственные частоты: ω₁ = {omega_lin[0]:.4f} рад/с, ω₂ = {omega_lin[1]:.4f} рад/с")
print(f"Форма первой моды: θ₁ : θ₂ = {v_lin[0, 0]:.4f} : {v_lin[1, 0]:.4f} "
      f"({'синфазно' if v_lin[0, 0] * v_lin[1, 0] > 0 else 'противофазно'})")
print(f"Форма второй моды: θ₁ : θ₂ = {v_lin[0, 1]:.4f} : {v_lin[1, 1]:.4f} "
      f"({'синфазно' if v_lin[0, 1] * v_lin[1, 1] > 0 else 'противофазно'})")
print()


# ==================== ПРАВЫЕ ЧАСТИ УРАВНЕНИЙ ДВИЖЕНИЯ ====================
def coupled_pendulums(t, y):
    """
    y = [θ1, θ2, ω1, ω2]
    Возвращает [dθ1/dt, dθ2/dt, dω1/dt, dω2/dt]
    Режим (линейный/нелинейный) управляется глобальной переменной nonlinear.
    """
    theta1, theta2, omega1, omega2 = y

    if nonlinear == 0:
        # ---------- ЛИНЕЙНОЕ ПРИБЛИЖЕНИЕ ----------
        # Используем линеаризованные выражения:
        # sinθ ≈ θ, cosθ ≈ 1, Δx ≈ d + L2θ2 - L1θ1, Δy ≈ L1 - L2, l ≈ l0
        # При этом (l-d)/l ≈ (l0-d)/l0 (константа) — но для линейности нужно учесть,
        # что (l-d)/l * (Δx cosθ + Δy sinθ) линеаризуется.
        # Упростим: будем использовать готовую линейную систему M·ddotθ + K·θ = 0,
        # где K уже вычислена. Тогда dω/dt = -M^{-1} K θ.
        # Это эквивалентно линеаризованным уравнениям.
        # Вычислим ускорения через матрицы:
        theta = np.array([theta1, theta2])
        domega = -np.linalg.inv(M_lin) @ K_lin @ theta
        return [omega1, omega2, domega[0], domega[1]]

    else:
        # ---------- НЕЛИНЕЙНОЕ ПРИБЛИЖЕНИЕ (СЛАБАЯ НЕЛИНЕЙНОСТЬ) ----------
        # Используем точные выражения, но можно также ввести разложение до третьего порядка,
        # однако для простоты оставим точные нелинейные уравнения (они уже содержат нелинейность).
        # Это позволит увидеть эффекты при конечных амплитудах.
        x1 = L1 * np.sin(theta1)
        y1 = -L1 * np.cos(theta1)
        x2 = d + L2 * np.sin(theta2)
        y2 = -L2 * np.cos(theta2)

        dx = x2 - x1
        dy = y2 - y1
        l = np.hypot(dx, dy)
        if l < 1e-12:
            l = 1e-12

        factor1 = (k / (m1 * L1)) * (l - d) / l
        factor2 = (k / (m2 * L2)) * (l - d) / l

        domega1 = - (g / L1) * np.sin(theta1) + factor1 * (dx * np.cos(theta1) + dy * np.sin(theta1))
        domega2 = - (g / L2) * np.sin(theta2) - factor2 * (dx * np.cos(theta2) + dy * np.sin(theta2))

        return [omega1, omega2, domega1, domega2]


# Начальный вектор
y0 = [theta1_0, theta2_0, omega1_0, omega2_0]

# Численное интегрирование
sol = solve_ivp(coupled_pendulums, t_span, y0, t_eval=t_eval, method='RK45')

# Извлекаем результаты
t = sol.t
theta1 = sol.y[0]
theta2 = sol.y[1]

# ==================== ВИЗУАЛИЗАЦИЯ ====================
plt.figure(figsize=(14, 10))

# 1. Графики углов
plt.subplot(2, 2, 1)
plt.plot(t, np.degrees(theta1), 'b-', label='θ₁')
plt.plot(t, np.degrees(theta2), 'r-', label='θ₂')
plt.xlabel('Время (с)')
plt.ylabel('Угол (градусы)')
plt.title('Углы отклонения маятников')
plt.legend()
plt.grid(True)

# 2. Спектр Фурье
plt.subplot(2, 2, 2)
dt = t[1] - t[0]
n = len(t)
freq = np.fft.fftfreq(n, dt)[:n // 2]
fft1 = np.fft.fft(theta1)[:n // 2]
fft2 = np.fft.fft(theta2)[:n // 2]
plt.plot(freq, np.abs(fft1) / np.max(np.abs(fft1)), 'b-', label='θ₁', alpha=0.7)
plt.plot(freq, np.abs(fft2) / np.max(np.abs(fft2)), 'r-', label='θ₂', alpha=0.7)
# Отметим собственные частоты
for w in omega_lin:
    plt.axvline(w / (2 * np.pi), color='k', linestyle='--', alpha=0.5)
plt.xlabel('Частота (Гц)')
plt.ylabel('Нормированная амплитуда')
plt.title('Спектр колебаний')
plt.xlim(0, max(omega_lin / (2 * np.pi)) * 2)
plt.grid(True)
plt.legend()

# 3. Фазовая траектория (θ₁, ω₁)
plt.subplot(2, 2, 3)
plt.plot(np.degrees(theta1), sol.y[2], 'b-', lw=1)
plt.xlabel('θ₁ (градусы)')
plt.ylabel('ω₁ (рад/с)')
plt.title('Фазовый портрет первого маятника')
plt.grid(True)

# 4. Фазовая траектория (θ₂, ω₂)
plt.subplot(2, 2, 4)
plt.plot(np.degrees(theta2), sol.y[3], 'r-', lw=1)
plt.xlabel('θ₂ (градусы)')
plt.ylabel('ω₂ (рад/с)')
plt.title('Фазовый портрет второго маятника')
plt.grid(True)

plt.tight_layout()
plt.show()

# ==================== АНИМАЦИЯ (опционально) ====================
# Раскомментируйте для просмотра анимации
if True:  # измените на True, чтобы включить анимацию
    fig, ax = plt.subplots(figsize=(6, 6))
    max_len = max(L1, L2)
    ax.set_xlim(-1.2 * max_len, d + 1.2 * max_len)
    ax.set_ylim(-1.2 * max_len, 0.2 * max_len)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.plot(0, 0, 'ko', markersize=8)
    ax.plot(d, 0, 'ko', markersize=8)

    line1, = ax.plot([], [], 'b-', lw=2)
    line2, = ax.plot([], [], 'r-', lw=2)
    mass1, = ax.plot([], [], 'bo', markersize=10)
    mass2, = ax.plot([], [], 'ro', markersize=10)
    spring, = ax.plot([], [], 'g--', lw=1.5)
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


    def animate(i):
        x1 = L1 * np.sin(theta1[i])
        y1 = -L1 * np.cos(theta1[i])
        x2 = d + L2 * np.sin(theta2[i])
        y2 = -L2 * np.cos(theta2[i])
        line1.set_data([0, x1], [0, y1])
        line2.set_data([d, x2], [0, y2])
        mass1.set_data([x1], [y1])
        mass2.set_data([x2], [y2])
        spring.set_data([x1, x2], [y1, y2])
        time_text.set_text(f'time = {t[i]:.2f} s')
        return line1, line2, mass1, mass2, spring, time_text


    ani = FuncAnimation(fig, animate, frames=len(t), interval=5, blit=True)
    plt.show()