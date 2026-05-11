"""
Лабораторная работа №7: Численное решение ОДУ
Вариант 20
y' = 21y - 8y^3 - x^2 * y,  y(0) = 1,  x in [0, 1]
Методы: Рунге-Кутта 4, Эйлер (явный), Эйлер (неявный)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Create output directory for plots
PLOT_DIR = "./img"

os.makedirs(PLOT_DIR, exist_ok=True)

def f(x, y):
    """Правая часть ОДУ: f(x,y) = 21y - 8y^3 - x^2 * y"""
    return 21*y - 8*y**3 - x**2 * y

def df_dy(x, y):
    """Частная производная f по y: df/dy = 21 - 24y^2 - x^2"""
    return 21 - 24*y**2 - x**2

def euler_explicit(f, x0, y0, x_end, N):
    """Явный метод Эйлера"""
    h = (x_end - x0) / N
    x = np.linspace(x0, x_end, N + 1)
    y = np.zeros(N + 1)
    y[0] = y0
    for i in range(N):
        y[i+1] = y[i] + h * f(x[i], y[i])
    return x, y

def euler_implicit(f, df_dy, x0, y0, x_end, N, tol=1e-12, max_iter=50):
    """
    Неявный метод Эйлера.
    Решает y_{n+1} = y_n + h * f(x_{n+1}, y_{n+1}) методом Ньютона.
    """
    h = (x_end - x0) / N
    x = np.linspace(x0, x_end, N + 1)
    y = np.zeros(N + 1)
    y[0] = y0
    for n in range(N):
        x_np1 = x[n+1]
        yn = y[n]
        # Начальное приближение: явный шаг Эйлера
        y_new = yn + h * f(x_np1, yn)
        # Метод Ньютона
        for _ in range(max_iter):
            F = y_new - yn - h * f(x_np1, y_new)
            dF = 1 - h * df_dy(x_np1, y_new)
            if abs(dF) < 1e-15:
                break
            delta = F / dF
            y_new = y_new - delta
            if abs(delta) < tol:
                break
        y[n+1] = y_new
    return x, y

def runge_kutta_4(f, x0, y0, x_end, N):
    """Метод Рунге-Кутта 4-го порядка"""
    h = (x_end - x0) / N
    x = np.linspace(x0, x_end, N + 1)
    y = np.zeros(N + 1)
    y[0] = y0
    for n in range(N):
        k1 = h * f(x[n], y[n])
        k2 = h * f(x[n] + h/2, y[n] + k1/2)
        k3 = h * f(x[n] + h/2, y[n] + k2/2)
        k4 = h * f(x[n] + h, y[n] + k3)
        y[n+1] = y[n] + (k1 + 2*k2 + 2*k3 + k4) / 6
    return x, y

def stability_coeff_explicit(x, y, h):
    """Коэффициент усиления погрешности для явного метода Эйлера: g = 1 + h * df/dy"""
    return 1 + h * df_dy(x, y)

def stability_coeff_implicit(x, y, h):
    """Коэффициент усиления погрешности для неявного метода Эйлера: g = 1 / (1 - h * df/dy)"""
    return 1 / (1 - h * df_dy(x, y))

# Параметры
x0, y0, x_end = 0.0, 1.0, 1.0
N_fine = 10000  # Очень мелкий шаг для "эталона"
N_values = [50, 100, 200]  # Исследуемые значения

print("=" * 60)
print("Лабораторная работа №7: Численное решение ОДУ")
print("Вариант 20: y' = 21y - 8y^3 - x^2 * y,  y(0) = 1")
print("=" * 60)

# ============================
# 1. RK4 как эталонный метод
# ============================
print("\n--- 1. Метод Рунге-Кутта 4-го порядка ---")
x_ref, y_ref = runge_kutta_4(f, x0, y0, x_end, N_fine)
print(f"Эталонное решение (N={N_fine}): y(1) = {y_ref[-1]:.10f}")
print(f"Шаг эталона: h = {(x_end-x0)/N_fine:.6f}")

# RK4 для разных N
rk4_results = {}
for N in N_values:
    x_rk, y_rk = runge_kutta_4(f, x0, y0, x_end, N)
    rk4_results[N] = (x_rk, y_rk)
    # Интерполируем эталон для сравнения
    y_ref_interp = np.interp(x_rk, x_ref, y_ref)
    abs_err = np.abs(y_rk - y_ref_interp)
    rel_err = abs_err / np.abs(y_ref_interp)
    print(f"\nN = {N}, h = {1.0/N:.4f}")
    print(f"  y(1) = {y_rk[-1]:.8f}")
    print(f"  |y - y_ref| at x=1: {abs_err[-1]:.8e}")
    print(f"  Относительная погрешность at x=1: {rel_err[-1]:.8e}")

# ============================
# 2. Явный метод Эйлера
# ============================
print("\n--- 2. Явный метод Эйлера ---")
euler_results = {}
for N in N_values:
    x_e, y_e = euler_explicit(f, x0, y0, x_end, N)
    euler_results[N] = (x_e, y_e)
    y_ref_interp = np.interp(x_e, x_ref, y_ref)
    abs_err = np.abs(y_e - y_ref_interp)
    rel_err = abs_err / np.abs(y_ref_interp)
    # Устойчивость
    h = 1.0 / N
    g_vals = [stability_coeff_explicit(x_e[i], y_e[i], h) for i in range(len(x_e))]
    g_max = max(abs(g) for g in g_vals)
    print(f"\nN = {N}, h = {h:.4f}")
    print(f"  y(1) = {y_e[-1]:.8f}")
    print(f"  |y - y_ref| at x=1: {abs_err[-1]:.8e}")
    print(f"  max |g| = {g_max:.4f} (устойчив: {g_max <= 1})")

# ============================
# 3. Неявный метод Эйлера
# ============================
print("\n--- 3. Неявный метод Эйлера ---")
implicit_results = {}
for N in N_values:
    x_i, y_i = euler_implicit(f, df_dy, x0, y0, x_end, N)
    implicit_results[N] = (x_i, y_i)
    y_ref_interp = np.interp(x_i, x_ref, y_ref)
    abs_err = np.abs(y_i - y_ref_interp)
    rel_err = abs_err / np.abs(y_ref_interp)
    # Устойчивость
    h = 1.0 / N
    g_vals = [stability_coeff_implicit(x_i[j], y_i[j], h) for j in range(len(x_i))]
    g_max = max(abs(g) for g in g_vals)
    print(f"\nN = {N}, h = {h:.4f}")
    print(f"  y(1) = {y_i[-1]:.8f}")
    print(f"  |y - y_ref| at x=1: {abs_err[-1]:.8e}")
    print(f"  max |g| = {g_max:.4f} (устойчив: {g_max <= 1})")

# ============================
# Графики
# ============================

# График 1: RK4 для N=50, 100, 200
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_ref, y_ref, 'k--', linewidth=1.5, label=f'RK4, N={N_fine} (эталон)', alpha=0.7)
colors = ['#2196F3', '#FF9800', '#4CAF50']
for i, N in enumerate(N_values):
    x_rk, y_rk = rk4_results[N]
    ax.plot(x_rk, y_rk, '-o', color=colors[i], markersize=2, linewidth=1.5, label=f'RK4, N={N}')
ax.set_xlabel('x')
ax.set_ylabel('y(x)')
ax.set_title('Решение ОДУ методом Рунге-Кутта при разном числе шагов')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'rk4_comparison.png'), dpi=150)
plt.close()
print("\n  График сохранён: rk4_comparison.png")

# График 2: Явный метод Эйлера
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_ref, y_ref, 'k--', linewidth=1.5, label='RK4 эталон', alpha=0.7)
for i, N in enumerate(N_values):
    x_e, y_e = euler_results[N]
    ax.plot(x_e, y_e, '-s', color=colors[i], markersize=2, linewidth=1.5, label=f'Эйлер явный, N={N}')
ax.set_xlabel('x')
ax.set_ylabel('y(x)')
ax.set_title('Решение ОДУ явным методом Эйлера')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'euler_explicit.png'), dpi=150)
plt.close()
print("  График сохранён: euler_explicit.png")

# График 3: Неявный метод Эйлера
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_ref, y_ref, 'k--', linewidth=1.5, label='RK4 эталон', alpha=0.7)
for i, N in enumerate(N_values):
    x_i, y_i = implicit_results[N]
    ax.plot(x_i, y_i, '-^', color=colors[i], markersize=2, linewidth=1.5, label=f'Эйлер неявный, N={N}')
ax.set_xlabel('x')
ax.set_ylabel('y(x)')
ax.set_title('Решение ОДУ неявным методом Эйлера')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'euler_implicit.png'), dpi=150)
plt.close()
print("  График сохранён: euler_implicit.png")

# График 4: Сравнение всех трёх методов для N=200
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_ref, y_ref, 'k--', linewidth=1.5, label='RK4 эталон (N=10000)', alpha=0.7)
x_rk, y_rk = rk4_results[200]
ax.plot(x_rk, y_rk, '-o', color='#2196F3', markersize=2, linewidth=1.5, label='RK4, N=200')
x_e, y_e = euler_results[200]
ax.plot(x_e, y_e, '-s', color='#FF9800', markersize=2, linewidth=1.5, label='Эйлер явный, N=200')
x_i, y_i = implicit_results[200]
ax.plot(x_i, y_i, '-^', color='#4CAF50', markersize=2, linewidth=1.5, label='Эйлер неявный, N=200')
ax.set_xlabel('x')
ax.set_ylabel('y(x)')
ax.set_title('Сравнение трёх методов (N=200)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'all_methods_comparison.png'), dpi=150)
plt.close()
print("  График сохранён: all_methods_comparison.png")

# График 5: Погрешности явного метода Эйлера
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, N in enumerate(N_values):
    x_e, y_e = euler_results[N]
    y_ref_interp = np.interp(x_e, x_ref, y_ref)
    abs_err = np.abs(y_e - y_ref_interp)
    rel_err = abs_err / (np.abs(y_ref_interp) + 1e-15)
    axes[idx].semilogy(x_e, abs_err, '-', color='red', label='Абс. погрешность')
    axes[idx].semilogy(x_e, rel_err, '--', color='blue', label='Отн. погрешность')
    axes[idx].set_title(f'N = {N}, h = {1.0/N:.4f}')
    axes[idx].set_xlabel('x')
    axes[idx].set_ylabel('Погрешность')
    axes[idx].legend(fontsize=8)
    axes[idx].grid(True, alpha=0.3)
plt.suptitle('Погрешности явного метода Эйлера', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'euler_explicit_errors.png'), dpi=150)
plt.close()
print("  График сохранён: euler_explicit_errors.png")

# График 6: Погрешности неявного метода Эйлера
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, N in enumerate(N_values):
    x_i, y_i = implicit_results[N]
    y_ref_interp = np.interp(x_i, x_ref, y_ref)
    abs_err = np.abs(y_i - y_ref_interp)
    rel_err = abs_err / (np.abs(y_ref_interp) + 1e-15)
    axes[idx].semilogy(x_i, abs_err, '-', color='red', label='Абс. погрешность')
    axes[idx].semilogy(x_i, rel_err, '--', color='blue', label='Отн. погрешность')
    axes[idx].set_title(f'N = {N}, h = {1.0/N:.4f}')
    axes[idx].set_xlabel('x')
    axes[idx].set_ylabel('Погрешность')
    axes[idx].legend(fontsize=8)
    axes[idx].grid(True, alpha=0.3)
plt.suptitle('Погрешности неявного метода Эйлера', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'euler_implicit_errors.png'), dpi=150)
plt.close()
print("  График сохранён: euler_implicit_errors.png")

# График 7: Устойчивость - коэффициенты g для N=50
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, N in enumerate(N_values):
    h = 1.0 / N
    x_e, y_e = euler_results[N]
    g_exp = [abs(stability_coeff_explicit(x_e[i], y_e[i], h)) for i in range(len(x_e))]
    x_i, y_i = implicit_results[N]
    g_imp = [abs(stability_coeff_implicit(x_i[j], y_i[j], h)) for j in range(len(x_i))]
    axes[idx].plot(x_e, g_exp, '-', color='orange', label='|g| явный Эйлер')
    axes[idx].plot(x_i, g_imp, '-', color='green', label='|g| неявный Эйлер')
    axes[idx].axhline(y=1, color='red', linestyle='--', label='Граница устойчивости')
    axes[idx].set_title(f'N = {N}, h = {h:.4f}')
    axes[idx].set_xlabel('x')
    axes[idx].set_ylabel('|g|')
    axes[idx].legend(fontsize=8)
    axes[idx].grid(True, alpha=0.3)
plt.suptitle('Коэффициенты усиления погрешности', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'stability_coefficients.png'), dpi=150)
plt.close()
print("  График сохранён: stability_coefficients.png")

# График 8: Сравнение RK4 для разных N - одним графиком
fig, ax = plt.subplots(figsize=(10, 6))
for i, N in enumerate(N_values):
    x_rk, y_rk = rk4_results[N]
    y_ref_interp = np.interp(x_rk, x_ref, y_ref)
    abs_err = np.abs(y_rk - y_ref_interp)
    ax.semilogy(x_rk, abs_err, '-o', color=colors[i], markersize=2, label=f'RK4, N={N}')
ax.set_xlabel('x')
ax.set_ylabel('Абсолютная погрешность')
ax.set_title('Погрешности метода Рунге-Кутта при разном числе шагов')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'rk4_errors.png'), dpi=150)
plt.close()
print("  График сохранён: rk4_errors.png")

# ============================
# Таблицы результатов
# ============================
print("\n" + "=" * 60)
print("СВОДНАЯ ТАБЛИЦА: значения y(1) для всех методов")
print("=" * 60)
print(f"{'Метод':<25} {'N=50':>15} {'N=100':>15} {'N=200':>15}")
print("-" * 75)
y_ref_1 = y_ref[-1]
print(f"{'RK4 эталон':<25} {y_ref_1:>15.10f}")
for name, results in [('RK4', rk4_results), ('Эйлер явный', euler_results), ('Эйлер неявный', implicit_results)]:
    vals = [f"{results[N][1][-1]:>15.8f}" for N in N_values]
    print(f"{name:<25} {vals[0]:>15} {vals[1]:>15} {vals[2]:>15}")

print("\n" + "=" * 60)
print("СВОДНАЯ ТАБЛИЦА: абсолютная погрешность |y(1) - y_ref|")
print("=" * 60)
print(f"{'Метод':<25} {'N=50':>15} {'N=100':>15} {'N=200':>15}")
print("-" * 75)
for name, results in [('RK4', rk4_results), ('Эйлер явный', euler_results), ('Эйлер неявный', implicit_results)]:
    errs = []
    for N in N_values:
        y_interp = np.interp(1.0, results[N][0], results[N][1])
        errs.append(f"{abs(y_interp - y_ref_1):>.6e}")
    print(f"{name:<25} {errs[0]:>15} {errs[1]:>15} {errs[2]:>15}")

plt.show()
print("\n" + "=" * 60)
print("Все графики сохранены в папке:", PLOT_DIR)
print("=" * 60)