import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# ============================================================
# 1. Модель динамики AQFP (безразмерные уравнения, β_c = 0)
# ============================================================
def aqfp_rhs(state, t, params):
    """
    Правая часть системы ОДУ для адиабатического квантового параметрона.
    state = [phi1, phi2]  — фазы на джозефсоновских переходах.
    params = (beta_L, beta_q, Ax, omega, phi_in)
    Возбуждение: phi_x(t) = Ax * sin(omega * t).
    Входной ток: phi_in (может быть константой или функцией времени).
    """
    phi1, phi2 = state
    beta_L, beta_q, Ax, omega, phi_in = params
    phi_x = Ax * np.sin(omega * t)

    dphi1 = -np.sin(phi1) - (phi1 - phi_x)/beta_L - (phi1 - phi2 - phi_in)/beta_q
    dphi2 = -np.sin(phi2) - (phi2 + phi_x)/beta_L + (phi1 - phi2 - phi_in)/beta_q

    return [dphi1, dphi2]

def output_current(state, params):
    """Вычисление нормированного выходного тока i_q."""
    phi1, phi2 = state
    beta_L, beta_q, Ax, omega, phi_in = params
    return (phi1 - phi2 - phi_in) / beta_q

# ============================================================
# 2. Симуляция на нескольких периодах и вычисление среднего
# ============================================================
def simulate_and_average(params, n_periods=50, n_points_per_period=200, transient_ratio=0.3):
    """
    Интегрирует ОДУ на n_periods периодов возбуждения,
    возвращает среднее значение i_q по последним (1-transient_ratio)*100% периодам.
    """
    beta_L, beta_q, Ax, omega, phi_in = params
    T = 2*np.pi / omega                # период возбуждения
    t_max = n_periods * T
    t_eval = np.linspace(0, t_max, n_periods * n_points_per_period)

    # Начальные условия (вблизи нуля)
    y0 = [0.0, 0.0]

    # Интегрирование
    sol = odeint(aqfp_rhs, y0, t_eval, args=(params,), rtol=1e-6, atol=1e-8)
    phi1_arr, phi2_arr = sol[:, 0], sol[:, 1]

    # Вычисление выходного тока
    iq_arr = (phi1_arr - phi2_arr - phi_in) / beta_q

    # Усреднение по установившемуся режиму
    n_transient = int(transient_ratio * len(t_eval))
    iq_steady = iq_arr[n_transient:]

    return np.mean(iq_steady)

# ============================================================
# 3. Исследование карты выпрямления (среднее i_q)
# ============================================================
def explore_rectification():
    """
    Сканирование по амплитуде Ax и частоте omega при фиксированных β_L, β_q.
    Строим 2D карту среднего значения i_q.
    """
    beta_L = 0.4
    beta_q = 1.6
    phi_in = 0.0

    Ax_vals = np.linspace(0.5, 3.0, 15)
    omega_vals = np.linspace(0.05, 0.5, 15)

    avg_iq = np.zeros((len(Ax_vals), len(omega_vals)))

    for i, Ax in enumerate(Ax_vals):
        for j, omega in enumerate(omega_vals):
            params = (beta_L, beta_q, Ax, omega, phi_in)
            avg_iq[i, j] = simulate_and_average(params, n_periods=30)
            print(f"Ax={Ax:.2f}, omega={omega:.3f} -> <iq> = {avg_iq[i,j]:.4f}")

    # Визуализация
    plt.figure(figsize=(8,6))
    plt.contourf(omega_vals, Ax_vals, avg_iq, levels=20, cmap='RdBu')
    plt.colorbar(label=r'$\langle i_q \rangle$')
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$A_x$')
    plt.title(f'Rectification effect (β_L={beta_L}, β_q={beta_q})')
    plt.show()

# ============================================================
# 4. Градиентный спуск для максимизации среднего значения
# ============================================================
def gradient_descent_maximize(init_params, lr=0.01, delta=0.05, max_iter=50, tol=1e-4):
    """
    Простой градиентный спуск с конечными разностями для максимизации |<iq>|.
    Варьируемые параметры: Ax, omega (β_L, β_q считаем фиксированными).
    """
    params = np.array(init_params, dtype=float)
    beta_L, beta_q, phi_in = params[0], params[1], params[4]  # фиксированные

    def objective(Ax_omega):
        Ax, omega = Ax_omega
        p = (beta_L, beta_q, Ax, omega, phi_in)
        return -abs(simulate_and_average(p, n_periods=20))  # минус для минимизации

    history = []
    for it in range(max_iter):
        Ax, omega = params[2], params[3]
        val = objective([Ax, omega])
        history.append(-val)

        # Конечные разности для градиента
        grad_Ax = (objective([Ax + delta, omega]) - objective([Ax - delta, omega])) / (2*delta)
        grad_omega = (objective([Ax, omega + delta]) - objective([Ax, omega - delta])) / (2*delta)
        grad = np.array([grad_Ax, grad_omega])

        # Обновление параметров
        new_Ax = Ax - lr * grad_Ax
        new_omega = omega - lr * grad_omega

        # Ограничения (неотрицательность и минимальные значения)
        new_Ax = max(0.1, new_Ax)
        new_omega = max(0.01, new_omega)

        params[2], params[3] = new_Ax, new_omega

        print(f"Iter {it}: Ax={new_Ax:.4f}, omega={new_omega:.4f}, |<iq>|={-val:.5f}")

        if np.linalg.norm(grad) < tol:
            break

    return params, history

# ============================================================
# Демонстрация
# ============================================================
if __name__ == "__main__":
    # 1. Исследование карты выпрямления
    explore_rectification()

    # 2. Оптимизация градиентным спуском
    # Фиксируем β_L=0.4, β_q=1.6, φ_in=0
    init_params = [0.4, 1.6, 2.0, 0.2, 0.0]  # beta_L, beta_q, Ax, omega, phi_in
    opt_params, hist = gradient_descent_maximize(init_params, lr=0.05, delta=0.1, max_iter=30)

    print("\nОптимальные параметры (β_L, β_q, Ax, omega, φ_in):")
    print(opt_params)

    # Построение графика сходимости
    plt.figure()
    plt.plot(hist, 'o-')
    plt.xlabel('Итерация')
    plt.ylabel(r'$|\langle i_q \rangle|$')
    plt.title('Сходимость градиентного спуска')
    plt.grid(True)
    plt.show()