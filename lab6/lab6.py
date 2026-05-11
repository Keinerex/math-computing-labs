import os

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

SAVE_DIR = "./img"
os.makedirs(SAVE_DIR, exist_ok=True)

np.random.seed(42)

# ===================== ВАРИАНТ 20 =====================

# --- Табличные данные для дифференцирования (Таблица 4) ---
x_tab = np.array([0.0, 2.4, 4.8, 7.2])
y_tab = np.array([1.06, 0.01, -0.2, 0.06])
h_tab = 2.4

# Точки для вычисления производных (a, b, c)
diff_points = [1.0, 3.0, 5.0]

# --- Аналитическая функция для дифференцирования (Таблица 5) ---
def f(x):
    return np.sin(np.cos(x))

def df_exact(x):
    return -np.sin(x) * np.cos(np.cos(x))

def d2f_exact(x):
    return -np.cos(x) * np.cos(np.cos(x)) - np.sin(x)**2 * np.sin(np.cos(x))

# --- Интеграл (Таблица 3) ---
def integrand(x):
    return 2 * np.cos(x**2) / (x + x**3)

A_INT, B_INT = 0.1, 1.0
I_exact, _ = integrate.quad(integrand, A_INT, B_INT)

# ===================== 1. ЧИСЛЕННОЕ ДИФФЕРЕНЦИРОВАНИЕ =====================

# 1.1 Табличные данные
def left_diff(y, h):
    """Левое разностное отношение: (y_i - y_{i-1}) / h"""
    return (y[1:] - y[:-1]) / h

def right_diff(y, h):
    """Правое разностное отношение: (y_{i+1} - y_i) / h"""
    return (y[1:] - y[:-1]) / h

def central_diff(y, h):
    """Центральное разностное отношение: (y_{i+1} - y_{i-1}) / (2h)"""
    return (y[2:] - y[:-2]) / (2 * h)

def second_diff(y, h):
    """Вторая производная: (y_{i-1} - 2y_i + y_{i+1}) / h^2"""
    return (y[:-2] - 2 * y[1:-1] + y[2:]) / h**2

# Вычислим производные в точках a=1.0, b=3.0, c=5.0 используя интерполяцию
# Так как точки не совпадают с узлами таблицы, используем линейную интерполяцию

def interp_value(x_nodes, y_nodes, x):
    """Линейная интерполяция"""
    for i in range(len(x_nodes) - 1):
        if x_nodes[i] <= x <= x_nodes[i+1]:
            return y_nodes[i] + (y_nodes[i+1] - y_nodes[i]) * (x - x_nodes[i]) / (x_nodes[i+1] - x_nodes[i])
    return y_nodes[-1]

# Вычисляем производные в точках с помощью численных формул на интерполированных значениях
print("=== 1.1 Численное дифференцирование таблично заданной функции ===")
print(f"{'Точка':>8} | {'Левое (3.4)':>12} | {'Правое (3.5)':>12} | {'Центр (3.7)':>12} | {'Вторая (3.8)':>12}")
print("-" * 75)

for pt in diff_points:
    # Находим интервал
    for i in range(len(x_tab) - 1):
        if x_tab[i] <= pt <= x_tab[i+1]:
            idx = i
            break
    else:
        idx = len(x_tab) - 2
    
    # Локальные значения
    x0, x1, x2 = x_tab[idx], x_tab[idx+1], x_tab[min(idx+2, len(x_tab)-1)]
    y0, y1, y2 = y_tab[idx], y_tab[idx+1], y_tab[min(idx+2, len(x_tab)-1)]
    h_local = x1 - x0
    
    # Левое и правое
    left = (y1 - y0) / h_local
    right = (y2 - y1) / h_local if idx < len(x_tab)-2 else (y1 - y0) / h_local
    
    # Центральное (если возможно)
    if idx < len(x_tab) - 2:
        central = (y2 - y0) / (2 * h_local)
    else:
        central = left
    
    # Вторая производная (если возможно)
    if idx < len(x_tab) - 2:
        second = (y0 - 2*y1 + y2) / h_local**2
    else:
        second = np.nan
    
    print(f"{pt:>8.1f} | {left:>12.6f} | {right:>12.6f} | {central:>12.6f} | {second:>12.6f}")

# 1.2 Аналитическая функция: графики и погрешности
x_fine = np.linspace(0, 2*np.pi, 500)
d1_exact = df_exact(x_fine)
d2_exact = d2f_exact(x_fine)

# Приближённые производные
hs = [0.5, 0.1, 0.01]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

for j, h in enumerate(hs):
    # Первая производная — правое разностное отношение
    d1_right = (f(x_fine + h) - f(x_fine)) / h
    # Первая производная — центральное
    d1_central = (f(x_fine + h) - f(x_fine - h)) / (2*h)
    # Вторая производная
    d2_approx = (f(x_fine - h) - 2*f(x_fine) + f(x_fine + h)) / h**2
    
    # Графики 1-й производной
    ax = axes[0, j]
    ax.plot(x_fine, d1_exact, 'k-', lw=2, label='Точная')
    ax.plot(x_fine, d1_right, 'r--', alpha=0.6, label=f'Правое, h={h}')
    ax.plot(x_fine, d1_central, 'b:', alpha=0.7, label=f'Центральное, h={h}')
    ax.set_title(f'Первая производная, h={h}')
    ax.set_xlabel('x')
    ax.set_ylabel("f'(x)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 2*np.pi)
    
    # Графики 2-й производной
    ax = axes[1, j]
    ax.plot(x_fine, d2_exact, 'k-', lw=2, label='Точная')
    ax.plot(x_fine, d2_approx, 'g--', alpha=0.6, label=f'Прибл., h={h}')
    ax.set_title(f'Вторая производная, h={h}')
    ax.set_xlabel('x')
    ax.set_ylabel("f''(x)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 2*np.pi)

plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/fig1_derivatives.png", dpi=150, bbox_inches='tight')
plt.show()

# Зависимость погрешности от h (в фиксированной точке x = π/2)
x_fix = np.pi / 2
h_range = np.logspace(-15, 0, 100)

err1_right = np.abs((f(x_fix + h_range) - f(x_fix)) / h_range - df_exact(x_fix))
err1_central = np.abs((f(x_fix + h_range) - f(x_fix - h_range)) / (2*h_range) - df_exact(x_fix))
err2 = np.abs((f(x_fix - h_range) - 2*f(x_fix) + f(x_fix + h_range)) / h_range**2 - d2f_exact(x_fix))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.loglog(h_range, err1_right, 'r-', label="Правое разностное (3.5)")
ax.loglog(h_range, err1_central, 'b-', label="Центральное (3.7)")
ax.axvline(0.5, color='gray', ls='--', alpha=0.5)
ax.axvline(0.1, color='gray', ls='--', alpha=0.5)
ax.axvline(0.01, color='gray', ls='--', alpha=0.5)
ax.set_xlabel('h')
ax.set_ylabel('Погрешность |O₁(h)|')
ax.set_title("Погрешность первой производной в x=π/2")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.loglog(h_range, err2, 'g-', label="Вторая производная (3.8)")
ax.axvline(0.5, color='gray', ls='--', alpha=0.5)
ax.axvline(0.1, color='gray', ls='--', alpha=0.5)
ax.axvline(0.01, color='gray', ls='--', alpha=0.5)
ax.set_xlabel('h')
ax.set_ylabel('Погрешность |O₂(h)|')
ax.set_title("Погрешность второй производной в x=π/2")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/fig2_error_vs_h.png", dpi=150, bbox_inches='tight')
plt.show()

# Таблица ошибок дифференцирования
print("\n=== Таблица 1: Максимальная ошибка дифференцирования ===")
print(f"{'h':>8} | {'Первая (правое)':>18} | {'Первая (центр.)':>18} | {'Вторая':>18}")
print("-" * 70)
for h in [0.5, 0.1, 0.01]:
    d1_r = (f(x_fine + h) - f(x_fine)) / h
    d1_c = (f(x_fine + h) - f(x_fine - h)) / (2*h)
    d2_a = (f(x_fine - h) - 2*f(x_fine) + f(x_fine + h)) / h**2
    e1r = np.max(np.abs(d1_r - d1_exact))
    e1c = np.max(np.abs(d1_c - d1_exact))
    e2 = np.max(np.abs(d2_a - d2_exact))
    print(f"{h:>8.2f} | {e1r:>18.6f} | {e1c:>18.6f} | {e2:>18.6f}")

# ===================== 2. ЧИСЛЕННОЕ ИНТЕГРИРОВАНИЕ =====================

# 2.1 Методы Ньютона-Котеса

def rectangle_mid(f, a, b, n):
    """Метод средних прямоугольников"""
    h = (b - a) / n
    result = 0.0
    for i in range(n):
        x_mid = a + h * (i + 0.5)
        result += f(x_mid)
    return result * h

def trapezoid(f, a, b, n):
    """Метод трапеций"""
    h = (b - a) / n
    result = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        result += f(a + h * i)
    return result * h

def simpson(f, a, b, n):
    """Метод Симпсона (n должно быть чётным)"""
    if n % 2 == 1:
        n += 1
    h = (b - a) / n
    result = f(a) + f(b)
    for i in range(1, n):
        x = a + h * i
        if i % 2 == 0:
            result += 2 * f(x)
        else:
            result += 4 * f(x)
    return result * h / 3

# 2.2 Тестирование при разных n
print("\n=== Таблица 3: Тестирование методов ===")
print(f"{'n':>5} | {'Прямоугольники':>16} | {'Трапеции':>16} | {'Симпсон':>16} | {'Погр. прям.':>12} | {'Погр. трап.':>12} | {'Погр. Симп.':>12}")
print("-" * 110)
test_n = [1, 2, 4, 10, 30, 50, 75, 100]
for n in test_n:
    r = rectangle_mid(integrand, A_INT, B_INT, n)
    t = trapezoid(integrand, A_INT, B_INT, n)
    s = simpson(integrand, A_INT, B_INT, n)
    er = abs(r - I_exact)
    et = abs(t - I_exact)
    es = abs(s - I_exact)
    print(f"{n:>5} | {r:>16.10f} | {t:>16.10f} | {s:>16.10f} | {er:>12.2e} | {et:>12.2e} | {es:>12.2e}")

# 2.3 Вычисление с заданной точностью (метод трапеций)
def integrate_with_precision(f, a, b, eps, method='trapezoid'):
    """Интегрирование с заданной точностью по правилу Рунге"""
    n = 2
    if method == 'trapezoid':
        I_prev = trapezoid(f, a, b, n)
    else:
        I_prev = simpson(f, a, b, n)
    
    while True:
        n *= 2
        if method == 'trapezoid':
            I_new = trapezoid(f, a, b, n)
        else:
            I_new = simpson(f, a, b, n)
        
        # Оценка погрешности по правилу Рунге
        if method == 'trapezoid':
            err_est = abs(I_new - I_prev) / 3.0
        else:
            err_est = abs(I_new - I_prev) / 15.0
        
        if err_est < eps:
            return I_new, n, err_est
        I_prev = I_new

print("\n=== Таблица 4: Интеграл с заданной точностью (метод трапеций) ===")
print(f"{'Точность ε':>12} | {'Итераций n':>12} | {'Прибл. значение':>18} | {'Оценка погрешности':>18}")
print("-" * 70)
for eps in [1e-1, 1e-3, 1e-5, 1e-8]:
    val, n_iter, err_est = integrate_with_precision(integrand, A_INT, B_INT, eps, 'trapezoid')
    print(f"{eps:>12.0e} | {n_iter:>12} | {val:>18.10f} | {err_est:>18.2e}")

# 2.4 Интеграл от таблично заданной функции (метод трапеций)
# Создаём неравномерную таблицу для f(x) = sin(cos x) на [0, 2π]
n_table = 20
x_tbl = np.sort(np.random.uniform(0, 2*np.pi, n_table))
y_tbl = f(x_tbl)

def trapezoid_table(x, y):
    """Метод трапеций для таблично заданной функции (неравномерный шаг)"""
    result = 0.0
    for i in range(len(x) - 1):
        h_i = x[i+1] - x[i]
        result += 0.5 * (y[i] + y[i+1]) * h_i
    return result

I_table = trapezoid_table(x_tbl, y_tbl)
I_table_exact, _ = integrate.quad(f, 0, 2*np.pi)

print(f"\n=== Табличное интегрирование ===")
print(f"Число узлов: {n_table}")
print(f"x_min={x_tbl.min():.4f}, x_max={x_tbl.max():.4f}")
print(f"Приближённое значение (трапеции): {I_table:.10f}")
print(f"Точное значение (quad):           {I_table_exact:.10f}")
print(f"Погрешность:                      {abs(I_table - I_table_exact):.2e}")

# ===================== 3. ГРАФИКИ ИНТЕГРИРОВАНИЯ =====================

# График сходимости
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ns = [1, 2, 4, 10, 30, 50, 75, 100]
errs_rect = []
errs_trap = []
errs_simp = []

for n in ns:
    errs_rect.append(abs(rectangle_mid(integrand, A_INT, B_INT, n) - I_exact))
    errs_trap.append(abs(trapezoid(integrand, A_INT, B_INT, n) - I_exact))
    errs_simp.append(abs(simpson(integrand, A_INT, B_INT, n) - I_exact))

ax = axes[0]
ax.semilogy(ns, errs_rect, 'b-o', label='Прямоугольники')
ax.semilogy(ns, errs_trap, 'r-s', label='Трапеции')
ax.semilogy(ns, errs_simp, 'g-^', label='Симпсон')
ax.set_xlabel('n (число интервалов)')
ax.set_ylabel('Абс. погрешность')
ax.set_title('Сходимость методов интегрирования')
ax.legend()
ax.grid(True, alpha=0.3)

# График подынтегральной функции
ax = axes[1]
x_plot = np.linspace(A_INT, B_INT, 500)
ax.plot(x_plot, integrand(x_plot), 'k-', lw=2)
ax.fill_between(x_plot, integrand(x_plot), alpha=0.2)
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.set_title(f'Подынтегральная функция, ∫ = {I_exact:.6f}')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/fig3_integration.png", dpi=150, bbox_inches='tight')
plt.show()

# График табличного интегрирования
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x_tbl, y_tbl, 'ro-', label='Табличные точки')
x_fine2 = np.linspace(0, 2*np.pi, 500)
ax.plot(x_fine2, f(x_fine2), 'k--', alpha=0.5, label='Истинная функция')
# Линейная интерполяция между табличными точками
for i in range(len(x_tbl)-1):
    xs = np.linspace(x_tbl[i], x_tbl[i+1], 20)
    ys = y_tbl[i] + (y_tbl[i+1] - y_tbl[i]) * (xs - x_tbl[i]) / (x_tbl[i+1] - x_tbl[i])
    ax.plot(xs, ys, 'b-', alpha=0.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title(f'Табличное интегрирование (трапеции): I ≈ {I_table:.6f}')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/fig4_table_integral.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"\n=== ВСЕ ГРАФИКИ СОХРАНЕНЫ В {SAVE_DIR} ===")