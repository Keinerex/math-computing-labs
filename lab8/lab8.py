"""
Лабораторная работа №8: Численные методы оптимизации
Вариант 20

Часть 1 (одномерная):
    f(x) = cos(x) -> min, x in [0, pi]
    Методы: Фибоначчи, золотого сечения, глобальный минимум (сканирование + дихотомия)

Часть 2 (многомерная):
    f(x1,x2) = -4*x1^2 - 8*x1 + x2 + 3 -> max
    Ограничение: -x1 - x2 = 2
    Методы: штрафов, градиентный спуск
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

PLOT_DIR = "./img"
os.makedirs(PLOT_DIR, exist_ok=True)

def savefig(name):
    plt.savefig(os.path.join(PLOT_DIR, name), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {name}")

# ============================================================
# ЧАСТЬ 1: Одномерная оптимизация
# ============================================================

def f1(x):
    """Целевая функция одной переменной"""
    return np.cos(x)

def df1(x):
    """Производная f1"""
    return -np.sin(x)

def d2f1(x):
    """Вторая производная f1"""
    return -np.cos(x)

def fibonacci_search(f, a, b, eps=1e-6):
    """Метод Фибоначчи для поиска минимума"""
    # Генерируем числа Фибоначчи
    fib = [1, 1]
    while fib[-1] < (b - a) / eps:
        fib.append(fib[-1] + fib[-2])
    n = len(fib) - 1

    x1 = a + fib[n-2] / fib[n] * (b - a)
    x2 = a + fib[n-1] / fib[n] * (b - a)
    f1_val = f(x1)
    f2_val = f(x2)
    history = [(a, b)]

    for k in range(1, n - 1):
        history.append((a, b))
        if f1_val > f2_val:  # ищем минимум
            a = x1
            x1 = x2
            f1_val = f2_val
            x2 = a + fib[n-k-1] / fib[n-k] * (b - a)
            f2_val = f(x2)
        else:
            b = x2
            x2 = x1
            f2_val = f1_val
            x1 = a + fib[n-k-2] / fib[n-k] * (b - a) if n-k-2 >= 0 else a + 0.382*(b-a)
            f1_val = f(x1)

    history.append((a, b))
    x_min = (a + b) / 2
    return x_min, f(x_min), len(history) - 1, history

def golden_section_search(f, a, b, eps=1e-6):
    """Метод золотого сечения для поиска минимума"""
    phi = (1 + np.sqrt(5)) / 2  # золотое сечение ~1.618
    resphi = 2 - phi            # ~0.382

    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)
    f1_val = f(x1)
    f2_val = f(x2)
    history = [(a, b)]
    iterations = 0

    while abs(b - a) > eps:
        history.append((a, b))
        iterations += 1
        if f1_val > f2_val:  # минимум справа
            a = x1
            x1 = x2
            f1_val = f2_val
            x2 = b - resphi * (b - a)
            f2_val = f(x2)
        else:  # минимум слева
            b = x2
            x2 = x1
            f2_val = f1_val
            x1 = a + resphi * (b - a)
            f1_val = f(x1)

    history.append((a, b))
    x_min = (a + b) / 2
    return x_min, f(x_min), iterations, history

def dichotomy_search(f, a, b, eps=1e-6, delta=None):
    """Метод дихотомии (деления отрезка пополам)"""
    if delta is None:
        delta = eps / 2
    iterations = 0
    history = [(a, b)]
    while abs(b - a) > eps:
        iterations += 1
        mid = (a + b) / 2
        x1 = mid - delta
        x2 = mid + delta
        if f(x1) < f(x2):
            b = x2
        else:
            a = x1
        history.append((a, b))
    x_min = (a + b) / 2
    return x_min, f(x_min), iterations, history

def global_minimum_scan(f, a, b, N=20, local_method=dichotomy_search, eps=1e-6):
    """
    Поиск глобального минимума:
    1. Разбиваем [a,b] на N подынтервалов
    2. На каждом подынтервале ищем локальный минимум методом дихотомии
    3. Выбираем наименьший из локальных минимумов
    """
    sub_intervals = np.linspace(a, b, N + 1)
    local_minima = []

    for i in range(N):
        ai, bi = sub_intervals[i], sub_intervals[i+1]
        # Оцениваем функцию на концах подынтервала
        if f(ai) < f(bi):
            # Возможно минимум на этом отрезке
            x_min, f_min, iters, _ = local_method(f, ai, bi, eps)
            local_minima.append((x_min, f_min, iters, ai, bi))

    if not local_minima:
        # Если не нашли, ищем на всем отрезке
        x_min, f_min, iters, _ = local_method(f, a, b, eps)
        local_minima.append((x_min, f_min, iters, a, b))

    # Выбираем глобальный минимум
    best = min(local_minima, key=lambda t: t[1])
    total_iters = sum(t[2] for t in local_minima)
    return best[0], best[1], total_iters, local_minima

# ============================================================
# ЧАСТЬ 2: Многомерная оптимизация
# ============================================================

def f2(x1, x2):
    """Целевая функция двух переменных"""
    return -4*x1**2 - 8*x1 + x2 + 3

def grad_f2(x1, x2):
    """Градиент f2"""
    return np.array([-8*x1 - 8, 1])

def penalty_method(f, grad_f, constraint, grad_constraint, x0, r0=0.1, C=10,
                   eps=1e-6, max_iter=100, method='max'):
    """
    Метод штрафов для условной оптимизации.
    F(x, r) = f(x) - r * P(x), где P(x) = (constraint(x))^2
    """
    x = np.array(x0, dtype=float)
    r = r0
    k = 0
    history = [x.copy()]

    def penalty_func(x_vec):
        c = constraint(x_vec[0], x_vec[1])
        P = c**2
        if method == 'max':
            return f(x_vec[0], x_vec[1]) - r * P
        else:
            return f(x_vec[0], x_vec[1]) + r * P

    def penalty_grad(x_vec):
        c = constraint(x_vec[0], x_vec[1])
        gc = grad_constraint(x_vec[0], x_vec[1])
        g = grad_f(x_vec[0], x_vec[1])
        if method == 'max':
            return g - r * 2 * c * gc
        else:
            return g + r * 2 * c * gc

    for k in range(max_iter):
        # Градиентный спуск для безусловной оптимизации F(x,r)
        x_k = x.copy()
        alpha = 0.01
        for _ in range(1000):
            g = penalty_grad(x_k)
            x_new = x_k + alpha * g  # для max идем по градиенту
            if penalty_func(x_new) > penalty_func(x_k):
                x_k = x_new
                alpha *= 1.2
            else:
                alpha *= 0.5
                if alpha < 1e-12:
                    break

        x = x_k
        history.append(x.copy())
        P_val = constraint(x[0], x[1])**2
        if P_val < eps:
            break
        r *= C

    return x, f(x[0], x[1]), k+1, history

def gradient_ascent(f, grad_f, x0, alpha=0.01, eps=1e-6, max_iter=5000):
    """Градиентный подъем (для максимизации)"""
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    for k in range(max_iter):
        g = grad_f(x[0], x[1])
        x_new = x + alpha * g
        history.append(x_new.copy())
        if np.linalg.norm(x_new - x) < eps:
            x = x_new
            break
        x = x_new
    return x, f(x[0], x[1]), k+1, history

# ============================================================
# ВЫПОЛНЕНИЕ
# ============================================================

print("=" * 60)
print("Лабораторная работа №8: Численные методы оптимизации")
print("Вариант 20")
print("=" * 60)

# --- Часть 1: Одномерная оптимизация ---
print("\n--- Часть 1: Одномерная оптимизация ---")
print("f(x) = cos(x) -> min, x in [0, pi]")

a, b = 0.0, np.pi

# Точное решение: f'(x) = -sin(x) = 0 => x = 0, pi. f''(x) = -cos(x).
# At x=pi: f''(pi) = -cos(pi) = 1 > 0 => minimum at x=pi
# At x=0: f''(0) = -1 < 0 => maximum at x=0
print(f"Точное решение: x_min = pi = {np.pi:.10f}, f_min = cos(pi) = {np.cos(np.pi):.10f}")

# Метод Фибоначчи
x_fib, f_fib, it_fib, hist_fib = fibonacci_search(f1, a, b, eps=1e-4)
print(f"\n1. Метод Фибоначчи:")
print(f"   x_min = {x_fib:.8f}, f_min = {f_fib:.8f}")
print(f"   Итераций: {it_fib}, |x - pi| = {abs(x_fib - np.pi):.2e}")

# Метод золотого сечения
x_gs, f_gs, it_gs, hist_gs = golden_section_search(f1, a, b, eps=1e-4)
print(f"\n2. Метод золотого сечения:")
print(f"   x_min = {x_gs:.8f}, f_min = {f_gs:.8f}")
print(f"   Итераций: {it_gs}, |x - pi| = {abs(x_gs - np.pi):.2e}")

# Метод дихотомии
x_dich, f_dich, it_dich, hist_dich = dichotomy_search(f1, a, b, eps=1e-4)
print(f"\n3. Метод дихотомии:")
print(f"   x_min = {x_dich:.8f}, f_min = {f_dich:.8f}")
print(f"   Итераций: {it_dich}, |x - pi| = {abs(x_dich - np.pi):.2e}")

# Глобальный минимум
x_glob, f_glob, it_glob, all_local = global_minimum_scan(f1, a, b, N=10, eps=1e-4)
print(f"\n4. Глобальный минимум (сканирование + дихотомия):")
print(f"   x_global = {x_glob:.8f}, f_global = {f_glob:.8f}")
print(f"   Всего итераций: {it_glob}")
print(f"   Найдено локальных минимумов: {len(all_local)}")

# --- Часть 2: Многомерная оптимизация ---
print("\n--- Часть 2: Многомерная оптимизация ---")
print("f(x1,x2) = -4*x1^2 - 8*x1 + x2 + 3 -> max")
print("Ограничение: -x1 - x2 = 2")

# Точное решение: метод Лагранжа
# L = -4x1^2 - 8x1 + x2 + 3 + lambda*(-x1 - x2 - 2)
# dL/dx1 = -8x1 - 8 - lambda = 0  =>  lambda = -8x1 - 8
# dL/dx2 = 1 - lambda = 0  =>  lambda = 1
# => -8x1 - 8 = 1  =>  x1 = -9/8 = -1.125
# -x1 - x2 = 2  =>  x2 = -2 - x1 = -2 + 1.125 = -0.875
x1_exact = -9/8
x2_exact = -2 - x1_exact
f_exact = f2(x1_exact, x2_exact)
print(f"Точное решение (метод Лагранжа):")
print(f"   x1 = {x1_exact:.6f}, x2 = {x2_exact:.6f}")
print(f"   f_max = {f_exact:.6f}")

# Безусловный максимум градиентным методом
x_bg, f_bg, it_bg, hist_bg = gradient_ascent(f2, grad_f2, [0.0, 0.0], alpha=0.01, max_iter=5000)
print(f"\n5. Безусловный максимум (градиентный метод):")
print(f"   x1 = {x_bg[0]:.6f}, x2 = {x_bg[1]:.6f}")
print(f"   f = {f_bg:.6f}")
print(f"   Итераций: {it_bg}")
print("   (Функция не ограничена сверху по x2 - градиент уходит в +inf)")

# Метод штрафов
def constraint(x1, x2):
    return -x1 - x2 - 2

def grad_constraint(x1, x2):
    return np.array([-1, -1])

x_pen, f_pen, it_pen, hist_pen = penalty_method(
    f2, grad_f2, constraint, grad_constraint,
    x0=[0.0, -2.0], r0=0.1, C=10, eps=1e-6, max_iter=50, method='max')
print(f"\n6. Условный максимум (метод штрафов):")
print(f"   x1 = {x_pen[0]:.6f}, x2 = {x_pen[1]:.6f}")
print(f"   f = {f_pen:.6f}")
print(f"   Ограничение: -x1 - x2 = {-x_pen[0] - x_pen[1]:.6f} (должно быть 2)")
print(f"   Итераций: {it_pen}")
print(f"   |x1 - x1_exact| = {abs(x_pen[0] - x1_exact):.6e}")
print(f"   |x2 - x2_exact| = {abs(x_pen[1] - x2_exact):.6e}")

# ============================================================
# ГРАФИКИ
# ============================================================

# График 1: f(x) = cos(x) и точки минимума
x_plot = np.linspace(0, np.pi, 500)
y_plot = np.cos(x_plot)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_plot, y_plot, 'b-', linewidth=2, label='f(x) = cos(x)')
ax.axvline(x=np.pi, color='r', linestyle='--', alpha=0.5, label=f'Точный минимум x = {np.pi:.4f}')
ax.plot(x_fib, f_fib, 'ro', markersize=10, label=f'Фибоначчи: ({x_fib:.4f}, {f_fib:.4f})')
ax.plot(x_gs, f_gs, 'gs', markersize=10, label=f'Золотое сечение: ({x_gs:.4f}, {f_gs:.4f})')
ax.plot(x_dich, f_dich, 'm^', markersize=10, label=f'Дихотомия: ({x_dich:.4f}, {f_dich:.4f})')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.set_title('Минимизация f(x) = cos(x) на [0, pi]')
ax.legend()
ax.grid(True, alpha=0.3)
savefig('1d_optimization.png')

# График 2: Сходимость методов - длина интервала по итерациям
fig, ax = plt.subplots(figsize=(10, 6))
fib_lens = [b-a for a, b in hist_fib]
gs_lens = [b-a for a, b in hist_gs]
dich_lens = [b-a for a, b in hist_dich]
ax.semilogy(range(len(fib_lens)), fib_lens, '-o', label='Фибоначчи', markersize=4)
ax.semilogy(range(len(gs_lens)), gs_lens, '-s', label='Золотое сечение', markersize=4)
ax.semilogy(range(len(dich_lens)), dich_lens, '-^', label='Дихотомия', markersize=4)
ax.set_xlabel('Итерация')
ax.set_ylabel('Длина интервала неопределенности')
ax.set_title('Сходимость методов одномерной оптимизации')
ax.legend()
ax.grid(True, alpha=0.3)
savefig('convergence_1d.png')

# График 3: Глобальный минимум - разбиение на подынтервалы
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_plot, y_plot, 'b-', linewidth=2, label='f(x) = cos(x)')
for i, (xm, fm, it, ai, bi) in enumerate(all_local):
    ax.axvline(x=ai, color='gray', linestyle=':', alpha=0.3)
    ax.plot(xm, fm, 'ro', markersize=6)
ax.axvline(x=b, color='gray', linestyle=':', alpha=0.3)
ax.plot(x_glob, f_glob, 'r*', markersize=15, label=f'Глобальный мин: ({x_glob:.4f}, {f_glob:.4f})')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.set_title('Поиск глобального минимума (разбиение на подынтервалы + дихотомия)')
ax.legend()
ax.grid(True, alpha=0.3)
savefig('global_minimum.png')

# График 4: Целевая функция 2D - поверхность
x1_range = np.linspace(-3, 1, 100)
x2_range = np.linspace(-4, 2, 100)
X1, X2 = np.meshgrid(x1_range, x2_range)
Z = -4*X1**2 - 8*X1 + X2 + 3

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.7)
# Точное решение
ax.plot([x1_exact], [x2_exact], [f_exact], 'r*', markersize=15, label='Точное решение')
# Решение методом штрафов
ax.plot([x_pen[0]], [x_pen[1]], [f_pen], 'go', markersize=10, label='Метод штрафов')
# Ограничение: x2 = -2 - x1
x1_c = np.linspace(-3, 1, 100)
x2_c = -2 - x1_c
z_c = -4*x1_c**2 - 8*x1_c + x2_c + 3
ax.plot(x1_c, x2_c, z_c, 'r-', linewidth=2, label='Ограничение')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1,x2)')
ax.set_title('Целевая функция и условный экстремум')
ax.legend()
savefig('2d_surface.png')

# График 5: Контурный график + траектория метода штрафов
fig, ax = plt.subplots(figsize=(10, 8))
contours = ax.contour(X1, X2, Z, levels=20, cmap='viridis')
plt.colorbar(contours, ax=ax)
# Ограничение
ax.plot(x1_c, x2_c, 'r-', linewidth=2, label='-x1 - x2 = 2')
# Траектория
if len(hist_pen) > 1:
    hist_arr = np.array(hist_pen)
    ax.plot(hist_arr[:, 0], hist_arr[:, 1], 'r.-', markersize=4, linewidth=1, label='Траектория штрафов')
# Точки
ax.plot(x1_exact, x2_exact, 'r*', markersize=15, label='Точное решение')
ax.plot(x_pen[0], x_pen[1], 'go', markersize=10, label='Метод штрафов')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_title('Контурный график: условный максимум методом штрафов')
ax.legend()
ax.grid(True, alpha=0.3)
savefig('2d_contour.png')

# График 6: Сходимость метода штрафов
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
hist_arr = np.array(hist_pen)
# Норма ошибки по ограничению
constraint_errors = [(-h[0] - h[1] - 2)**2 for h in hist_arr]
axes[0].semilogy(constraint_errors, '-o', markersize=4)
axes[0].set_xlabel('Итерация')
axes[0].set_ylabel('P(x) = (-x1 - x2 - 2)^2')
axes[0].set_title('Сходимость ограничения (метод штрафов)')
axes[0].grid(True, alpha=0.3)
# Значение функции
f_values = [f2(h[0], h[1]) for h in hist_arr]
axes[1].plot(f_values, '-o', markersize=4, color='green')
axes[1].axhline(y=f_exact, color='r', linestyle='--', label=f'Точное f = {f_exact:.4f}')
axes[1].set_xlabel('Итерация')
axes[1].set_ylabel('f(x1, x2)')
axes[1].set_title('Значение целевой функции')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
savefig('penalty_convergence.png')

# ============================
# СВОДНАЯ ТАБЛИЦА
# ============================
print("\n" + "=" * 60)
print("СВОДНАЯ ТАБЛИЦА: результаты одномерной оптимизации")
print("=" * 60)
print(f"{'Метод':<25} {'x_min':>12} {'f_min':>12} {'Итераций':>10}")
print("-" * 60)
print(f"{'Точное':<25} {np.pi:>12.8f} {np.cos(np.pi):>12.8f} {'-':>10}")
print(f"{'Фибоначчи':<25} {x_fib:>12.8f} {f_fib:>12.8f} {it_fib:>10}")
print(f"{'Золотое сечение':<25} {x_gs:>12.8f} {f_gs:>12.8f} {it_gs:>10}")
print(f"{'Дихотомия':<25} {x_dich:>12.8f} {f_dich:>12.8f} {it_dich:>10}")
print(f"{'Глобальный (скан+дих)':<25} {x_glob:>12.8f} {f_glob:>12.8f} {it_glob:>10}")

print("\n" + "=" * 60)
print("СВОДНАЯ ТАБЛИЦА: результаты многомерной оптимизации")
print("=" * 60)
print(f"{'Параметр':<20} {'Точное':>12} {'Метод штрафов':>15}")
print("-" * 50)
print(f"{'x1':<20} {x1_exact:>12.6f} {x_pen[0]:>15.6f}")
print(f"{'x2':<20} {x2_exact:>12.6f} {x_pen[1]:>15.6f}")
print(f"{'f(x1,x2)':<20} {f_exact:>12.6f} {f_pen:>15.6f}")
print(f"{'Ограничение':<20} {-x1_exact-x2_exact:>12.6f} {-x_pen[0]-x_pen[1]:>15.6f}")

print("\nВсе графики сохранены в папке:", PLOT_DIR)