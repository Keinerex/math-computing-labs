import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import polynomial as P
import os

SAVE_DIR = "./img"
os.makedirs(SAVE_DIR, exist_ok=True)

np.random.seed(42)

# ===================== ВАРИАНТ 20 =====================
A, B = 2.0, 10.0

def f(x):
    """Истинная функция: sin(x) / ln(x)"""
    x = np.asarray(x, dtype=float)
    return np.sin(x) / np.log(x)

# ===================== 1. СОЗДАНИЕ ЗАШУМЛЁННЫХ ДАННЫХ =====================

def generate_noisy_data(n_nodes, noise_level=0.05, uniform=True):
    """Генерация зашумлённых данных. noise_level — доля от амплитуды функции."""
    if uniform:
        x_nodes = np.linspace(A, B, n_nodes)
    else:
        x_nodes = np.sort(np.random.uniform(A, B, n_nodes))
    y_true = f(x_nodes)
    amplitude = np.max(y_true) - np.min(y_true)
    noise = np.random.normal(0, noise_level * amplitude, size=n_nodes)
    y_noisy = y_true + noise
    return x_nodes, y_noisy, y_true

# Создаём данные: n1=10 (равномерные), n2=20 (равномерные), n3=15 (случайные)
x10_u, y10_n, y10_t = generate_noisy_data(10, 0.08, uniform=True)
x20_u, y20_n, y20_t = generate_noisy_data(20, 0.08, uniform=True)
x15_r, y15_n, y15_t = generate_noisy_data(15, 0.08, uniform=False)

# ===================== 2. СГЛАЖИВАНИЕ =====================

def smooth_linear_3(y):
    """Линейное сглаживание по 3 точкам (скользящее среднее)."""
    n = len(y)
    ys = np.zeros(n)
    ys[0] = y[0]
    ys[-1] = y[-1]
    for i in range(1, n - 1):
        ys[i] = (y[i-1] + y[i] + y[i+1]) / 3.0
    return ys

def smooth_linear_5(y):
    """Линейное сглаживание по 5 точкам."""
    n = len(y)
    ys = np.zeros(n)
    ys[0], ys[1] = y[0], y[1]
    ys[-1], ys[-2] = y[-1], y[-2]
    for i in range(2, n - 2):
        ys[i] = (y[i-2] + y[i-1] + y[i] + y[i+1] + y[i+2]) / 5.0
    return ys

def smooth_nonlinear_7(y):
    """Нелинейное сглаживание: медианный фильтр по 7 точкам."""
    from scipy.ndimage import median_filter
    return median_filter(y, size=7, mode='nearest')

# Сглаживаем данные n=20
y20_s3 = smooth_linear_3(y20_n)
y20_s5 = smooth_linear_5(y20_n)
try:
    from scipy.ndimage import median_filter
    y20_s7 = smooth_nonlinear_7(y20_n)
    HAS_SCIPY = True
except ImportError:
    y20_s7 = y20_s5.copy()  # fallback
    HAS_SCIPY = False

# ===================== 3. ИНТЕРПОЛЯЦИЯ ЛАГРАНЖА =====================

def lagrange_interp(x_nodes, y_nodes, x_eval):
    """Интерполяция полиномом Лагранжа."""
    x_eval = np.asarray(x_eval)
    n = len(x_nodes)
    result = np.zeros_like(x_eval, dtype=float)
    for i in range(n):
        L = np.ones_like(x_eval, dtype=float)
        for j in range(n):
            if i != j:
                L *= (x_eval - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        result += y_nodes[i] * L
    return result

# ===================== 4. КУБИЧЕСКИЙ СПЛАЙН =====================

def cubic_spline_coeffs(x_nodes, y_nodes):
    """Вычисление коэффициентов кубического сплайна (c_i через трёхдиагональную систему).
    Возвращает (a, b, c, d, h) — коэффициенты для каждого отрезка."""
    n = len(x_nodes) - 1  # число отрезков
    h = np.diff(x_nodes)
    a = y_nodes.copy()
    
    # Система для c: трёхдиагональная
    alpha = np.zeros(n + 1)
    for i in range(1, n):
        alpha[i] = (3.0 / h[i]) * (a[i+1] - a[i]) - (3.0 / h[i-1]) * (a[i] - a[i-1])
    
    l = np.ones(n + 1)
    mu = np.zeros(n + 1)
    z = np.zeros(n + 1)
    l[0] = 1.0
    z[0] = 0.0
    
    for i in range(1, n):
        l[i] = 2.0 * (x_nodes[i+1] - x_nodes[i-1]) - h[i-1] * mu[i-1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i-1] * z[i-1]) / l[i]
    
    l[n] = 1.0
    z[n] = 0.0
    c = np.zeros(n + 1)
    
    for j in range(n - 1, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
    
    b = np.zeros(n)
    d = np.zeros(n)
    for i in range(n):
        b[i] = (a[i+1] - a[i]) / h[i] - h[i] * (c[i+1] + 2.0 * c[i]) / 3.0
        d[i] = (c[i+1] - c[i]) / (3.0 * h[i])
    
    return a, b, c, d, h

def cubic_spline_eval(x_nodes, a, b, c, d, x_eval):
    """Вычисление значения кубического сплайна в точках x_eval."""
    x_eval = np.asarray(x_eval)
    n = len(x_nodes) - 1
    result = np.zeros_like(x_eval, dtype=float)
    for i in range(n):
        mask = (x_eval >= x_nodes[i]) & (x_eval <= x_nodes[i+1])
        if i < n - 1:
            mask = mask | ((x_eval > x_nodes[i]) & (x_eval < x_nodes[i+1]))
        if i == n - 1:
            mask = (x_eval >= x_nodes[i]) & (x_eval <= x_nodes[i+1])
        dx = x_eval[mask] - x_nodes[i]
        result[mask] = a[i] + b[i]*dx + c[i]*dx**2 + d[i]*dx**3
    return result

def cubic_spline(x_nodes, y_nodes, x_eval):
    """Удобная обёртка."""
    a, b, c, d, h = cubic_spline_coeffs(x_nodes, y_nodes)
    return cubic_spline_eval(x_nodes, a, b, c, d, x_eval)

# ===================== 5. МЕТОД НАИМЕНЬШИХ КВАДРАТОВ =====================

def least_squares_poly(x_nodes, y_nodes, degree):
    """Аппроксимация полиномом степени degree методом наименьших квадратов."""
    # Используем np.polyfit (устойчивый алгоритм)
    coeffs = np.polyfit(x_nodes, y_nodes, degree)
    return coeffs  # от старшей степени к младшей

def eval_poly(coeffs, x):
    """Вычисление значения полинома. coeffs от старшей степени к младшей."""
    return np.polyval(coeffs, x)

# ===================== 6. ГРАФИКИ =====================

x_fine = np.linspace(A, B, 500)
y_fine = f(x_fine)

# --- Рис. 1: Зашумлённые данные и сглаживание ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(x_fine, y_fine, 'k-', lw=2, label='Истинная функция')
ax.scatter(x20_u, y20_n, c='red', s=30, zorder=5, label='Зашумлённые данные (n=20)')
ax.scatter(x10_u, y10_n, c='orange', s=40, marker='s', zorder=5, label='Зашумлённые данные (n=10)')
ax.set_title('Равномерная сетка')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(x_fine, y_fine, 'k-', lw=2, label='Истинная функция')
ax.scatter(x15_r, y15_n, c='blue', s=30, zorder=5, label='Случайная сетка (n=15)')
ax.set_title('Случайная сетка')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/fig1_noisy_data.png", dpi=150, bbox_inches='tight')
plt.show()

# --- Рис. 2: Сглаживание ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

ax = axes[0]
ax.plot(x_fine, y_fine, 'k-', lw=2, label='Истинная')
ax.plot(x20_u, y20_n, 'r--', alpha=0.5, label='Зашумлённые')
ax.plot(x20_u, y20_s3, 'g-o', markersize=3, label='Сглаж. по 3 точкам')
ax.set_title('Линейное сглаживание (3 точки)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(x_fine, y_fine, 'k-', lw=2, label='Истинная')
ax.plot(x20_u, y20_n, 'r--', alpha=0.5, label='Зашумлённые')
ax.plot(x20_u, y20_s5, 'b-s', markersize=3, label='Сглаж. по 5 точкам')
ax.set_title('Линейное сглаживание (5 точек)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.plot(x_fine, y_fine, 'k-', lw=2, label='Истинная')
ax.plot(x20_u, y20_n, 'r--', alpha=0.5, label='Зашумлённые')
if HAS_SCIPY:
    ax.plot(x20_u, y20_s7, 'm-^', markersize=3, label='Нелинейное (медиана, 7)')
else:
    ax.plot(x20_u, y20_s5, 'm-^', markersize=3, label='Нелинейное (5 точек)')
ax.set_title('Нелинейное сглаживание')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/fig2_smoothing.png", dpi=150, bbox_inches='tight')
plt.show()

# --- Рис. 3: Лагранж и Сплайн ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Лагранж по сглаженным данным n=10
y_lag = lagrange_interp(x10_u, y20_s3[:10], x_fine)  # используем первые 10 точек
# Better: use x10_u with smoothed data at those points
y_lag10 = lagrange_interp(x10_u, y10_n, x_fine)

ax = axes[0]
ax.plot(x_fine, y_fine, 'k-', lw=2, label='Истинная')
ax.plot(x_fine, y_lag10, 'b--', lw=1.5, label='Лагранж n=10 (зашумл.)')
# Лагранж по сглаженным
y_lag10_s = lagrange_interp(x10_u, smooth_linear_3(y10_n), x_fine)
ax.plot(x_fine, y_lag10_s, 'g-', lw=1.5, label='Лагранж n=10 (сглаж.)')
ax.scatter(x10_u, y10_n, c='red', s=30, zorder=5)
ax.set_title('Интерполяция Лагранжа')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Сплайн
y_spl = cubic_spline(x10_u, y10_n, x_fine)
y_spl_s = cubic_spline(x10_u, smooth_linear_3(y10_n), x_fine)

ax = axes[1]
ax.plot(x_fine, y_fine, 'k-', lw=2, label='Истинная')
ax.plot(x_fine, y_spl, 'b--', lw=1.5, label='Сплайн (зашумл.)')
ax.plot(x_fine, y_spl_s, 'r-', lw=1.5, label='Сплайн (сглаж.)')
ax.scatter(x10_u, y10_n, c='red', s=30, zorder=5)
ax.set_title('Кубический сплайн')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/fig3_lagrange_spline.png", dpi=150, bbox_inches='tight')
plt.show()

# --- Рис. 4: МНК ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
degrees = [2, 3, 4, 5]
colors = ['blue', 'green', 'red', 'purple']

for idx, (deg, color) in enumerate(zip(degrees, colors)):
    ax = axes[idx // 2, idx % 2]
    coeffs = least_squares_poly(x20_u, y20_n, deg)
    y_mnk = eval_poly(coeffs, x_fine)
    ax.plot(x_fine, y_fine, 'k-', lw=2, label='Истинная')
    ax.plot(x_fine, y_mnk, color=color, lw=1.5, label=f'МНК, степень {deg}')
    ax.scatter(x20_u, y20_n, c='red', s=20, alpha=0.6, zorder=5)
    ax.set_title(f'Аппроксимация МНК, степень {deg}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/fig4_mnk.png", dpi=150, bbox_inches='tight')
plt.show()

# --- Рис. 5: Сравнение МНК со сглаженными данными ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# МНК на зашумлённых
ax = axes[0]
ax.plot(x_fine, y_fine, 'k-', lw=2, label='Истинная')
for deg, color in zip([2, 3, 4], ['blue', 'green', 'red']):
    coeffs = least_squares_poly(x20_u, y20_n, deg)
    y_mnk = eval_poly(coeffs, x_fine)
    err = np.max(np.abs(y_mnk - y_fine))
    ax.plot(x_fine, y_mnk, color=color, lw=1.5, label=f'МНК deg={deg}, max_err={err:.4f}')
ax.scatter(x20_u, y20_n, c='red', s=20, alpha=0.6, zorder=5)
ax.set_title('МНК на зашумлённых данных')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# МНК на сглаженных
ax = axes[1]
ax.plot(x_fine, y_fine, 'k-', lw=2, label='Истинная')
for deg, color in zip([2, 3, 4], ['blue', 'green', 'red']):
    coeffs = least_squares_poly(x20_u, y20_s3, deg)
    y_mnk = eval_poly(coeffs, x_fine)
    err = np.max(np.abs(y_mnk - y_fine))
    ax.plot(x_fine, y_mnk, color=color, lw=1.5, label=f'МНК deg={deg}, max_err={err:.4f}')
ax.plot(x20_u, y20_s3, 'c--', alpha=0.5, label='Сглаженные данные')
ax.set_title('МНК на сглаженных данных')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/fig5_mnk_compare.png", dpi=150, bbox_inches='tight')
plt.show()

# ===================== 7. ТАБЛИЦЫ ПОГРЕШНОСТЕЙ =====================

print("\n=== ТАБЛИЦА 1: Погрешности сглаживания (n=20, равномерная сетка) ===")
print(f"{'Метод':<30} {'Средняя абс. погрешность':<25} {'Макс. абс. погрешность':<25}")
print("-" * 80)
for name, y_sm in [("Без сглаживания", y20_n), ("Линейное 3 точки", y20_s3), 
                     ("Линейное 5 точек", y20_s5), ("Нелинейное 7 точек", y20_s7 if HAS_SCIPY else y20_s5)]:
    diff = np.abs(y_sm - y20_t)
    print(f"{name:<30} {np.mean(diff):<25.6f} {np.max(diff):<25.6f}")

print("\n=== ТАБЛИЦА 2: Погрешности интерполяции (n=10, равномерная) ===")
print(f"{'Метод':<35} {'Макс. абс. погрешность':<25}")
print("-" * 60)
y_lag = lagrange_interp(x10_u, y10_n, x_fine)
y_lag_s = lagrange_interp(x10_u, smooth_linear_3(y10_n), x_fine)
y_spl = cubic_spline(x10_u, y10_n, x_fine)
y_spl_s = cubic_spline(x10_u, smooth_linear_3(y10_n), x_fine)
print(f"{'Лагранж (зашумл.)':<35} {np.max(np.abs(y_lag - y_fine)):<25.6f}")
print(f"{'Лагранж (сглаж.)':<35} {np.max(np.abs(y_lag_s - y_fine)):<25.6f}")
print(f"{'Куб. сплайн (зашумл.)':<35} {np.max(np.abs(y_spl - y_fine)):<25.6f}")
print(f"{'Куб. сплайн (сглаж.)':<35} {np.max(np.abs(y_spl_s - y_fine)):<25.6f}")

print("\n=== ТАБЛИЦА 3: Погрешности МНК (n=20, равномерная) ===")
print(f"{'Степень':<10} {'Макс. ошибка (зашумл.)':<25} {'Макс. ошибка (сглаж.)':<25} {'Число обусловленности':<25}")
print("-" * 85)
for deg in [2, 3, 4, 5]:
    coeffs_n = least_squares_poly(x20_u, y20_n, deg)
    coeffs_s = least_squares_poly(x20_u, y20_s3, deg)
    y_mnk_n = eval_poly(coeffs_n, x_fine)
    y_mnk_s = eval_poly(coeffs_s, x_fine)
    err_n = np.max(np.abs(y_mnk_n - y_fine))
    err_s = np.max(np.abs(y_mnk_s - y_fine))
    # Число обусловленности матрицы Вандермонда
    V = np.vander(x20_u, deg + 1, increasing=True)
    cond = np.linalg.cond(V)
    print(f"{deg:<10} {err_n:<25.6f} {err_s:<25.6f} {cond:<25.6e}")

print("\n=== ВСЕ ГРАФИКИ СОХРАНЕНЫ В", SAVE_DIR, "===")

plt.show()
