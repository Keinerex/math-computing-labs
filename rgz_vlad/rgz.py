"""
РГЗ: Методы приближения функций
Вариант 28
x = [0, 1, 2, 3], y = [0, 5, 6, 1]
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.polynomial import polynomial as P
import os

PLOT_DIR = "./img"
os.makedirs(PLOT_DIR, exist_ok=True)

def savefig(name):
    plt.savefig(os.path.join(PLOT_DIR, name), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {name}")

# ============================================================
# ИСХОДНЫЕ ДАННЫЕ
# ============================================================
x_nodes = np.array([0.0, 1.0, 2.0, 3.0])
y_nodes = np.array([0.0, 5.0, 6.0, 1.0])

print("=" * 60)
print("РГЗ: Методы приближения функций")
print("Вариант 28")
print(f"x = {x_nodes}")
print(f"y = {y_nodes}")
print("=" * 60)

# ============================================================
# МЕТОД 1: Интерполяция через решение СЛАУ
# ============================================================
print("\n--- 1. Интерполяция через решение СЛАУ ---")
# P(x) = a0 + a1*x + a2*x^2 + a3*x^3
# Составляем матрицу Вандермонда
V = np.vander(x_nodes, increasing=True)
coeffs_slau = np.linalg.solve(V, y_nodes)
print(f"Коэффициенты полинома: {coeffs_slau}")
P_slau = np.poly1d(coeffs_slau[::-1])  # для удобства вычислений

def eval_poly(coeffs, x):
    """coeffs[0] + coeffs[1]*x + coeffs[2]*x^2 + ..."""
    return sum(c * x**i for i, c in enumerate(coeffs))

# Проверка в узлах
print("Проверка в узлах:")
for xn, yn in zip(x_nodes, y_nodes):
    yp = eval_poly(coeffs_slau, xn)
    print(f"  P({xn}) = {yp:.6f}, точное = {yn:.6f}, погр = {abs(yp-yn):.2e}")

# ============================================================
# МЕТОД 2: Интерполяция Лагранжа
# ============================================================
print("\n--- 2. Интерполяция методом Лагранжа ---")

def lagrange_interp(x_nodes, y_nodes, x):
    """Интерполяционный многочлен Лагранжа"""
    n = len(x_nodes)
    result = 0.0
    for i in range(n):
        Li = 1.0
        for j in range(n):
            if i != j:
                Li *= (x - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        result += y_nodes[i] * Li
    return result

print("Проверка в узлах:")
for xn, yn in zip(x_nodes, y_nodes):
    yp = lagrange_interp(x_nodes, y_nodes, xn)
    print(f"  L({xn}) = {yp:.6f}, точное = {yn:.6f}, погр = {abs(yp-yn):.2e}")

# ============================================================
# МЕТОД 3: Интерполяция методом Ньютона
# ============================================================
print("\n--- 3. Интерполяция методом Ньютона ---")

def newton_interp(x_nodes, y_nodes, x):
    """Интерполяционный многочлен Ньютона"""
    n = len(x_nodes)
    # Таблица разделенных разностей
    F = np.zeros((n, n))
    F[:, 0] = y_nodes
    for j in range(1, n):
        for i in range(n - j):
            F[i, j] = (F[i+1, j-1] - F[i, j-1]) / (x_nodes[i+j] - x_nodes[i])
    
    # Вычисление полинома
    result = F[0, 0]
    prod = 1.0
    for i in range(1, n):
        prod *= (x - x_nodes[i-1])
        result += F[0, i] * prod
    return result, F

print("Проверка в узлах:")
for xn, yn in zip(x_nodes, y_nodes):
    yp, _ = newton_interp(x_nodes, y_nodes, xn)
    print(f"  N({xn}) = {yp:.6f}, точное = {yn:.6f}, погр = {abs(yp-yn):.2e}")

# Таблица разделенных разностей
_, F_table = newton_interp(x_nodes, y_nodes, 0)
print("Таблица разделенных разностей:")
print(F_table)

# ============================================================
# МЕТОД 4: Интерполяция кубическими сплайнами
# ============================================================
print("\n--- 4. Интерполяция кубическими сплайнами ---")
from scipy.interpolate import CubicSpline

cs = CubicSpline(x_nodes, y_nodes, bc_type='natural')
print("Коэффициенты сплайна (на каждом отрезке):")
for i in range(len(x_nodes)-1):
    c = c_list = cs.c[:, i]
    print(f"  Отрезок [{x_nodes[i]}, {x_nodes[i+1]}]:")
    print(f"    S(x) = {c[0]:.4f}*(x-{x_nodes[i]})^3 + {c[1]:.4f}*(x-{x_nodes[i]})^2 + {c[2]:.4f}*(x-{x_nodes[i]}) + {c[3]:.4f}")

print("Проверка в узлах:")
for xn, yn in zip(x_nodes, y_nodes):
    yp = cs(xn)
    print(f"  S({xn}) = {yp:.6f}, точное = {yn:.6f}, погр = {abs(yp-yn):.2e}")

# ============================================================
# МЕТОД 5: Аппроксимация полиномом (МНК)
# ============================================================
print("\n--- 5. Аппроксимация полиномом (МНК) ---")

# МНК полиномом степени m (m < n)
for m in [1, 2]:
    coeffs_mnk = np.polyfit(x_nodes, y_nodes, m)
    P_mnk = np.poly1d(coeffs_mnk)
    residuals = y_nodes - P_mnk(x_nodes)
    sse = np.sum(residuals**2)
    print(f"\nМНК полином степени m={m}:")
    print(f"  Коэффициенты (от старшей): {coeffs_mnk}")
    print(f"  Сумма квадратов отклонений: {sse:.6f}")
    for xn, yn in zip(x_nodes, y_nodes):
        yp = P_mnk(xn)
        print(f"  P({xn}) = {yp:.4f}, точное = {yn:.4f}, отклонение = {yp-yn:.4f}")

coeffs_mnk = np.polyfit(x_nodes, y_nodes, 2)
P_mnk = np.poly1d(coeffs_mnk)

# ============================================================
# П.1: ЗНАЧЕНИЕ ФУНКЦИИ В ТОЧКЕ x = 2.58
# ============================================================
print("\n--- П.1: Значение функции при x = 2.58 ---")
x_test = 2.58

y_slau = eval_poly(coeffs_slau, x_test)
y_lag = lagrange_interp(x_nodes, y_nodes, x_test)
y_new, _ = newton_interp(x_nodes, y_nodes, x_test)
y_spl = cs(x_test)
y_mnk = P_mnk(x_test)

print(f"\nПри x = {x_test}:")
print(f"  СЛАУ:      P({x_test}) = {y_slau:.6f}")
print(f"  Лагранж:   L({x_test}) = {y_lag:.6f}")
print(f"  Ньютон:    N({x_test}) = {y_new:.6f}")
print(f"  Сплайн:    S({x_test}) = {y_spl:.6f}")
print(f"  МНК:       P({x_test}) = {y_mnk:.6f}")

# ============================================================
# П.2: ЭКСТРАПОЛЯЦИЯ НА ДВА ШАГА ВПЕРЁД (x = 4, x = 5)
# ============================================================
print("\n--- П.2: Экстраполяция на x = 4 и x = 5 ---")
x_extra = [4.0, 5.0]

print(f"\n{'Метод':<15} {'x=4':>12} {'x=5':>12}")
print("-" * 40)
for name, func in [
    ('СЛАУ', lambda x: eval_poly(coeffs_slau, x)),
    ('Лагранж', lambda x: lagrange_interp(x_nodes, y_nodes, x)),
    ('Ньютон', lambda x: newton_interp(x_nodes, y_nodes, x)[0]),
    ('Сплайн', lambda x: cs(x)),
    ('МНК', lambda x: P_mnk(x))
]:
    vals = [func(xe) for xe in x_extra]
    print(f"  {name:<12} {vals[0]:>12.4f} {vals[1]:>12.4f}")

# ============================================================
# П.3: ИНТЕГРИРОВАНИЕ И ДИФФЕРЕНЦИРОВАНИЕ (для полинома СЛАУ)
# ============================================================
print("\n--- П.3: Интегрирование и дифференцирование ---")
print("(для полинома, полученного методом СЛАУ)")

# Аналитический интеграл полинома
# P(x) = a0 + a1*x + a2*x^2 + a3*x^3
# Integral = a0*x + a1*x^2/2 + a2*x^3/3 + a3*x^4/4
a0, a1, a2, a3 = coeffs_slau
I_analytic = (a0*3 + a1*9/2 + a2*27/3 + a3*81/4) - (a0*0 + a1*0 + a2*0 + a3*0)
print(f"\nИнтеграл от P(x) на [0, 3] (аналитический): {I_analytic:.6f}")

# Численное интегрирование (Симпсон)
def simpson(f, a, b, N=1000):
    h = (b - a) / N
    result = f(a) + f(b)
    for i in range(1, N):
        if i % 2 == 0:
            result += 2 * f(a + i*h)
        else:
            result += 4 * f(a + i*h)
    return result * h / 3

f_slau = lambda x: eval_poly(coeffs_slau, x)
I_numeric = simpson(f_slau, 0, 3, N=1000)
print(f"Интеграл (Симпсон, N=1000):                    {I_numeric:.6f}")
print(f"Абсолютная погрешность:                        {abs(I_analytic - I_numeric):.2e}")

# Производная в точке x = 2.58
# P'(x) = a1 + 2*a2*x + 3*a3*x^2
deriv_analytic = a1 + 2*a2*x_test + 3*a3*x_test**2
print(f"\nПроизводная P'(x) в точке x = {x_test}:")
print(f"  Аналитическая:  P'({x_test}) = {deriv_analytic:.6f}")

# Численная производная (центральная разность)
h_diff = 1e-6
deriv_numeric = (eval_poly(coeffs_slau, x_test + h_diff) - eval_poly(coeffs_slau, x_test - h_diff)) / (2*h_diff)
print(f"  Численная:      P'({x_test}) = {deriv_numeric:.6f}")
print(f"  Погрешность:    {abs(deriv_analytic - deriv_numeric):.2e}")

# ============================================================
# ГРАФИКИ
# ============================================================
x_plot = np.linspace(-0.5, 5.5, 500)

# График 1: Все методы на одном графике
fig, ax = plt.subplots(figsize=(12, 7))
ax.scatter(x_nodes, y_nodes, s=100, c='red', zorder=5, label='Узлы интерполяции')

y_slau_plot = [eval_poly(coeffs_slau, x) for x in x_plot]
y_lag_plot = [lagrange_interp(x_nodes, y_nodes, x) for x in x_plot]
y_new_plot = [newton_interp(x_nodes, y_nodes, x)[0] for x in x_plot]
y_spl_plot = cs(x_plot)
y_mnk_plot = P_mnk(x_plot)

ax.plot(x_plot, y_slau_plot, 'b-', linewidth=2, label='СЛАУ (полином 3-й степени)')
ax.plot(x_plot, y_lag_plot, 'g--', linewidth=1.5, label='Лагранж')
ax.plot(x_plot, y_new_plot, 'm:', linewidth=1.5, label='Ньютон')
ax.plot(x_plot, y_spl_plot, 'c-.', linewidth=2, label='Сплайн (естественный)')
ax.plot(x_plot, y_mnk_plot, 'orange', linewidth=2, label='МНК (m=2)')

# Экстраполяция
ax.axvline(x=3, color='gray', linestyle='--', alpha=0.5, label='Граница экстраполяции')
ax.fill_betweenx([-20, 20], 3, 5.5, alpha=0.05, color='yellow')

ax.set_xlim(-0.5, 5.5)
ax.set_ylim(-8, 10)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('РГЗ Вариант 28: Методы приближения функций\nx = [0,1,2,3], y = [0,5,6,1]')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
savefig('all_methods.png')

# График 2: Отдельно - только узлы и полином СЛАУ
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x_nodes, y_nodes, s=150, c='red', zorder=5, label='Узлы')
ax.plot(x_plot, y_slau_plot, 'b-', linewidth=2, label=f'P(x) = {a3:.4f}x³ + {a2:.4f}x² + {a1:.4f}x + {a0:.4f}')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Интерполяция через решение СЛАУ')
ax.legend()
ax.grid(True, alpha=0.3)
savefig('method_slau.png')

# График 3: Лагранж
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x_nodes, y_nodes, s=150, c='red', zorder=5, label='Узлы')
y_lag_full = [lagrange_interp(x_nodes, y_nodes, x) for x in x_plot]
ax.plot(x_plot, y_lag_full, 'g-', linewidth=2, label='Лагранж')
# Базисные функции
for i in range(len(x_nodes)):
    Li_x = []
    for x in x_plot:
        Li = 1.0
        for j in range(len(x_nodes)):
            if i != j:
                Li *= (x - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        Li_x.append(Li * y_nodes[i])
    ax.plot(x_plot, Li_x, '--', alpha=0.4, label=f'l_{i}(x)*y_{i}')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Интерполяция методом Лагранжа')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
savefig('method_lagrange.png')

# График 4: Ньютон
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x_nodes, y_nodes, s=150, c='red', zorder=5, label='Узлы')
y_new_full = [newton_interp(x_nodes, y_nodes, x)[0] for x in x_plot]
ax.plot(x_plot, y_new_full, 'm-', linewidth=2, label='Ньютон')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Интерполяция методом Ньютона')
ax.legend()
ax.grid(True, alpha=0.3)
savefig('method_newton.png')

# График 5: Сплайны
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x_nodes, y_nodes, s=150, c='red', zorder=5, label='Узлы')
for i in range(len(x_nodes)-1):
    x_seg = np.linspace(x_nodes[i], x_nodes[i+1], 100)
    ax.plot(x_seg, cs(x_seg), linewidth=2.5, label=f'S_{i}(x) на [{x_nodes[i]},{x_nodes[i+1]}]')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Интерполяция кубическими сплайнами (естественные граничные условия)')
ax.legend()
ax.grid(True, alpha=0.3)
savefig('method_spline.png')

# График 6: МНК
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x_nodes, y_nodes, s=150, c='red', zorder=5, label='Узлы')
y_mnk_full = P_mnk(x_plot)
ax.plot(x_plot, y_mnk_full, 'orange', linewidth=2, label=f'МНК m=2')
# Отклонения
for xn, yn in zip(x_nodes, y_nodes):
    yn_mnk = P_mnk(xn)
    ax.plot([xn, xn], [yn, yn_mnk], 'k--', alpha=0.5)
    ax.plot(xn, yn_mnk, 'go', markersize=6)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Аппроксимация методом наименьших квадратов (m=2)')
ax.legend()
ax.grid(True, alpha=0.3)
savefig('method_mnk.png')

# График 7: Экстраполяция
fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(x_nodes, y_nodes, s=150, c='red', zorder=5, label='Узлы')
ax.plot(x_plot, y_slau_plot, 'b-', linewidth=2, alpha=0.7, label='СЛАУ')
ax.plot(x_plot, y_lag_plot, 'g--', linewidth=1.5, alpha=0.7, label='Лагранж')
ax.plot(x_plot, y_spl_plot, 'c-.', linewidth=2, alpha=0.7, label='Сплайн')
ax.plot(x_plot, y_mnk_plot, 'orange', linewidth=2, alpha=0.7, label='МНК')
ax.axvline(x=3, color='gray', linestyle='--', alpha=0.5)
ax.axvspan(3, 5.5, alpha=0.05, color='yellow', label='Зона экстраполяции')
ax.set_xlim(-0.5, 5.5)
ax.set_ylim(-15, 15)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Экстраполяция на два шага вперед (x = 4, x = 5)')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
savefig('extrapolation.png')

# График 8: Интегрирование
fig, ax = plt.subplots(figsize=(10, 6))
x_int = np.linspace(0, 3, 500)
y_int = [eval_poly(coeffs_slau, x) for x in x_int]
ax.fill_between(x_int, 0, y_int, alpha=0.3, color='blue', label=f'Integral = {I_analytic:.4f}')
ax.plot(x_int, y_int, 'b-', linewidth=2, label='P(x) (СЛАУ)')
ax.axvline(x=0, color='k', linewidth=0.5)
ax.axvline(x=3, color='k', linewidth=0.5)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title(f'Численное интегрирование на [0, 3]: integral = {I_analytic:.6f}')
ax.legend()
ax.grid(True, alpha=0.3)
savefig('integration.png')

print("\n" + "=" * 60)
print("Все графики сохранены в папке:", PLOT_DIR)
print("=" * 60)
