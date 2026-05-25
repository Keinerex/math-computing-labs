"""
РГЗ: Методы приближения функций
Вариант 11
x = [0, 1, 2, 3], y = [0, 4, 5, 2]
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import os

PLOT_DIR = "./img"
os.makedirs(PLOT_DIR, exist_ok=True)

def savefig(name):
    plt.savefig(os.path.join(PLOT_DIR, name), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {name}")

def eval_poly(coeffs, x):
    return sum(c * x**i for i, c in enumerate(coeffs))

def lagrange_interp(x_nodes, y_nodes, x):
    n = len(x_nodes)
    result = 0.0
    for i in range(n):
        Li = 1.0
        for j in range(n):
            if i != j:
                Li *= (x - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        result += y_nodes[i] * Li
    return result

def newton_interp(x_nodes, y_nodes, x):
    n = len(x_nodes)
    F = np.zeros((n, n))
    F[:, 0] = y_nodes
    for j in range(1, n):
        for i in range(n - j):
            F[i, j] = (F[i+1, j-1] - F[i, j-1]) / (x_nodes[i+j] - x_nodes[i])
    result = F[0, 0]
    prod = 1.0
    for i in range(1, n):
        prod *= (x - x_nodes[i-1])
        result += F[0, i] * prod
    return result, F

def simpson(f, a, b, N=1000):
    h = (b - a) / N
    result = f(a) + f(b)
    for i in range(1, N):
        result += (4 if i % 2 else 2) * f(a + i*h)
    return result * h / 3

# ============================================================
# ИСХОДНЫЕ ДАННЫЕ ВАРИАНТ 11
# ============================================================
x_nodes = np.array([0.0, 1.0, 2.0, 3.0])
y_nodes = np.array([0.0, 4.0, 5.0, 2.0])

print("=" * 60)
print("РГЗ: Методы приближения функций")
print("Вариант 11")
print(f"x = {x_nodes}")
print(f"y = {y_nodes}")
print("=" * 60)

# ============================================================
# МЕТОД 1: СЛАУ
# ============================================================
print("\n--- 1. Интерполяция через решение СЛАУ ---")
V = np.vander(x_nodes, increasing=True)
coeffs_slau = np.linalg.solve(V, y_nodes)
a0, a1, a2, a3 = coeffs_slau
print(f"P(x) = {a3:.4f}x^3 + {a2:.4f}x^2 + {a1:.4f}x + {a0:.4f}")
for xn, yn in zip(x_nodes, y_nodes):
    print(f"  P({xn}) = {eval_poly(coeffs_slau, xn):.6f}, точн = {yn}, погр = {abs(eval_poly(coeffs_slau, xn)-yn):.2e}")

# ============================================================
# МЕТОД 2: Лагранж
# ============================================================
print("\n--- 2. Интерполяция методом Лагранжа ---")
for xn, yn in zip(x_nodes, y_nodes):
    print(f"  L({xn}) = {lagrange_interp(x_nodes, y_nodes, xn):.6f}")

# ============================================================
# МЕТОД 3: Ньютон
# ============================================================
print("\n--- 3. Интерполяция методом Ньютона ---")
_, F_table = newton_interp(x_nodes, y_nodes, 0)
print("Таблица разделенных разностей:")
print(F_table)
for xn, yn in zip(x_nodes, y_nodes):
    yp, _ = newton_interp(x_nodes, y_nodes, xn)
    print(f"  N({xn}) = {yp:.6f}")

# ============================================================
# МЕТОД 4: Сплайны
# ============================================================
print("\n--- 4. Интерполяция кубическими сплайнами ---")
cs = CubicSpline(x_nodes, y_nodes, bc_type='natural')
for i in range(len(x_nodes)-1):
    c = cs.c[:, i]
    print(f"  S{i}(x) на [{x_nodes[i]},{x_nodes[i+1]}]: {c[0]:.4f}(x-{x_nodes[i]})^3 + {c[1]:.4f}(x-{x_nodes[i]})^2 + {c[2]:.4f}(x-{x_nodes[i]}) + {c[3]:.4f}")

# ============================================================
# МЕТОД 5: МНК
# ============================================================
print("\n--- 5. Аппроксимация полиномом (МНК) ---")
for m in [1, 2]:
    coeffs_mnk = np.polyfit(x_nodes, y_nodes, m)
    P_mnk = np.poly1d(coeffs_mnk)
    sse = np.sum((y_nodes - P_mnk(x_nodes))**2)
    print(f"m={m}: coeffs={coeffs_mnk}, SSE={sse:.4f}")
coeffs_mnk = np.polyfit(x_nodes, y_nodes, 2)
P_mnk = np.poly1d(coeffs_mnk)

# ============================================================
# П.1: ЗНАЧЕНИЕ В ТОЧКЕ x = 2.58
# ============================================================
print("\n--- П.1: Значение при x = 2.58 ---")
x_test = 2.58
methods = {
    'СЛАУ': lambda x: eval_poly(coeffs_slau, x),
    'Лагранж': lambda x: lagrange_interp(x_nodes, y_nodes, x),
    'Ньютон': lambda x: newton_interp(x_nodes, y_nodes, x)[0],
    'Сплайн': lambda x: cs(x),
    'МНК': lambda x: P_mnk(x)
}
for name, f in methods.items():
    print(f"  {name:12s}: {f(x_test):.6f}")

# ============================================================
# П.2: ЭКСТРАПОЛЯЦИЯ
# ============================================================
print("\n--- П.2: Экстраполяция на x = 4, 5 ---")
print(f"{'Метод':<15} {'x=4':>10} {'x=5':>10}")
print("-" * 40)
for name, f in methods.items():
    print(f"  {name:<12s} {f(4.0):>10.4f} {f(5.0):>10.4f}")

# ============================================================
# П.3: ИНТЕГРАЛ И ПРОИЗВОДНАЯ
# ============================================================
print("\n--- П.3: Интеграл и производная ---")
f_slau = lambda x: eval_poly(coeffs_slau, x)
I_an = a0*3 + a1*9/2 + a2*27/3 + a3*81/4
I_num = simpson(f_slau, 0, 3, 1000)
print(f"Интеграл [0,3]: аналит = {I_an:.6f}, Симпсон = {I_num:.6f}, погр = {abs(I_an-I_num):.2e}")

deriv_an = a1 + 2*a2*x_test + 3*a3*x_test**2
h = 1e-6
deriv_num = (eval_poly(coeffs_slau, x_test+h) - eval_poly(coeffs_slau, x_test-h)) / (2*h)
print(f"P'(2.58): аналит = {deriv_an:.6f}, числ = {deriv_num:.6f}, погр = {abs(deriv_an-deriv_num):.2e}")

# ============================================================
# ГРАФИКИ
# ============================================================
x_plot = np.linspace(-0.5, 5.5, 500)

# Все методы
fig, ax = plt.subplots(figsize=(12, 7))
ax.scatter(x_nodes, y_nodes, s=100, c='red', zorder=5, label='Узлы')
for name, color, ls in [('СЛАУ','b','-'),('Лагранж','g','--'),('Ньютон','m',':'),('Сплайн','c','-.')]:
    yp = [methods[name](x) for x in x_plot]
    ax.plot(x_plot, yp, color=color, linestyle=ls, linewidth=2, label=name)
ax.plot(x_plot, [P_mnk(x) for x in x_plot], color='orange', linewidth=2, label='МНК (m=2)')
ax.axvline(x=3, color='gray', linestyle='--', alpha=0.5)
ax.axvspan(3, 5.5, alpha=0.05, color='yellow')
ax.set_xlim(-0.5, 5.5); ax.set_ylim(-8, 8)
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_title('РГЗ Вариант 11: Методы приближения функций\nx=[0,1,2,3], y=[0,4,5,2]')
ax.legend(); ax.grid(True, alpha=0.3)
savefig('all_methods.png')

# Отдельные графики
for name, fname in [('СЛАУ','method_slau'),('Лагранж','method_lagrange'),('Ньютон','method_newton'),('Сплайн','method_spline'),('МНК','method_mnk')]:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x_nodes, y_nodes, s=150, c='red', zorder=5, label='Узлы')
    if name == 'Сплайн':
        for i in range(len(x_nodes)-1):
            x_seg = np.linspace(x_nodes[i], x_nodes[i+1], 100)
            ax.plot(x_seg, cs(x_seg), linewidth=2.5, label=f'S{i}')
    else:
        yp = [methods[name](x) for x in x_plot]
        ax.plot(x_plot, yp, linewidth=2, label=name)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title(f'Вариант 11: {name}')
    ax.legend(); ax.grid(True, alpha=0.3)
    savefig(f'{fname}.png')

# Экстраполяция
fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(x_nodes, y_nodes, s=150, c='red', zorder=5, label='Узлы')
for name, color, ls in [('СЛАУ','b','-'),('Лагранж','g','--'),('Сплайн','c','-.')]:
    ax.plot(x_plot, [methods[name](x) for x in x_plot], color=color, linestyle=ls, linewidth=2, label=name, alpha=0.7)
ax.plot(x_plot, [P_mnk(x) for x in x_plot], color='orange', linewidth=2, alpha=0.7, label='МНК')
ax.axvline(x=3, color='gray', linestyle='--', alpha=0.5)
ax.axvspan(3, 5.5, alpha=0.05, color='yellow', label='Экстраполяция')
ax.set_xlim(-0.5, 5.5); ax.set_ylim(-10, 10)
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_title('Экстраполяция на два шага вперед')
ax.legend(loc='upper left'); ax.grid(True, alpha=0.3)
savefig('extrapolation.png')

# Интегрирование
fig, ax = plt.subplots(figsize=(10, 6))
x_int = np.linspace(0, 3, 500)
y_int = [eval_poly(coeffs_slau, x) for x in x_int]
ax.fill_between(x_int, 0, y_int, alpha=0.3, color='blue')
ax.plot(x_int, y_int, 'b-', linewidth=2, label=f'integral = {I_an:.6f}')
ax.axhline(y=0, color='k', linewidth=0.5)
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_title(f'Численное интегрирование на [0, 3]: {I_an:.6f}')
ax.legend(); ax.grid(True, alpha=0.3)
savefig('integration.png')

print("\nВсе графики сохранены в папке:", PLOT_DIR)
