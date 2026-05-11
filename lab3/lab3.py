"""
=============================================================
Решение системы нелинейных уравнений тремя методами:
  - Метод простых итераций
  - Метод Ньютона
  - Метод Зейделя

Система:
  sin(x - y) - x*y = -1   [F1]
  x^2 - y^2  = 0.75       [F2]
=============================================================
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



# ──────────────────────────────────────────────────────────
#  Функции системы и матрица Якоби
# ──────────────────────────────────────────────────────────

def F1(x, y):
    return np.sin(x - y) - x * y + 1.0          # F1 = 0

def F2(x, y):
    return x**2 - y**2 - 0.75                    # F2 = 0

def F(x, y):
    return np.array([F1(x, y), F2(x, y)])

def jacobian(x, y):
    """Матрица Якоби системы F1, F2"""
    dF1dx =  np.cos(x - y) - y
    dF1dy = -np.cos(x - y) - x
    dF2dx =  2.0 * x
    dF2dy = -2.0 * y
    return np.array([[dF1dx, dF1dy],
                     [dF2dx, dF2dy]])


# ──────────────────────────────────────────────────────────
#  Итерационная схема для методов простых итераций и Зейделя
#
#  g1: из F2  =>  x = sqrt(y^2 + 0.75)
#  g2: из F1  =>  y = (sin(x - y) + 1) / x
#  (g2 задаётся неявно, но при итерациях y берётся со старой итерации)
# ──────────────────────────────────────────────────────────

def g1(x, y):
    return np.sqrt(y**2 + 0.75)

def g2(x, y):
    return (np.sin(x - y) + 1.0) / x


# ══════════════════════════════════════════════════════════
#  1. МЕТОД ПРОСТЫХ ИТЕРАЦИЙ
# ══════════════════════════════════════════════════════════

def simple_iteration(x0, y0, tol=1e-6, max_iter=1000):
    """
    Итерационная схема:
      x_{n+1} = g1(x_n, y_n)
      y_{n+1} = g2(x_n, y_n)   <- используются СТАРЫЕ значения
    """
    x, y = float(x0), float(y0)
    history = [(0, x, y, np.nan, np.nan)]

    for k in range(1, max_iter + 1):
        x_new = g1(x, y)
        y_new = g2(x, y)          # оба аргумента — предыдущие

        err_x = abs(x_new - x)
        err_y = abs(y_new - y)
        history.append((k, x_new, y_new, err_x, err_y))

        x, y = x_new, y_new
        if err_x < tol and err_y < tol:
            break

    return x, y, history


# ══════════════════════════════════════════════════════════
#  2. МЕТОД НЬЮТОНА
# ══════════════════════════════════════════════════════════

def newton(x0, y0, tol=1e-6, max_iter=100):
    """
    J(x_n) * Δ = -F(x_n)
    x_{n+1} = x_n + Δ
    """
    x, y = float(x0), float(y0)
    history = [(0, x, y, np.nan, np.nan)]

    for k in range(1, max_iter + 1):
        f   = F(x, y)
        jac = jacobian(x, y)

        det = np.linalg.det(jac)
        if abs(det) < 1e-15:
            print(f"  [Ньютон] Вырожденный Якоби на шаге {k}, det = {det:.2e}")
            break

        delta = np.linalg.solve(jac, -f)
        x_new = x + delta[0]
        y_new = y + delta[1]

        err_x = abs(delta[0])
        err_y = abs(delta[1])
        history.append((k, x_new, y_new, err_x, err_y))

        x, y = x_new, y_new
        if np.linalg.norm(delta) < tol:
            break

    return x, y, history


# ══════════════════════════════════════════════════════════
#  3. МЕТОД ЗЕЙДЕЛЯ (нелинейный)
# ══════════════════════════════════════════════════════════

def seidel(x0, y0, tol=1e-6, max_iter=1000):
    """
    Итерационная схема:
      x_{n+1} = g1(x_n,   y_n)
      y_{n+1} = g2(x_{n+1}, y_n)   <- используется НОВЫЙ x_{n+1}
    """
    x, y = float(x0), float(y0)
    history = [(0, x, y, np.nan, np.nan)]

    for k in range(1, max_iter + 1):
        x_new = g1(x,     y)
        y_new = g2(x_new, y)        # обновлённый x_new подставляем сразу

        err_x = abs(x_new - x)
        err_y = abs(y_new - y)
        history.append((k, x_new, y_new, err_x, err_y))

        x, y = x_new, y_new
        if err_x < tol and err_y < tol:
            break

    return x, y, history


# ──────────────────────────────────────────────────────────
#  Вспомогательные функции вывода
# ──────────────────────────────────────────────────────────

HEADER = (
    f"  {'Итер':>5}  {'x':>14}  {'y':>14}  "
    f"{'|Δx|':>12}  {'|Δy|':>12}"
)
SEP = "  " + "─" * 63


def print_table(history, max_rows=20):
    print(HEADER)
    print(SEP)
    rows = history if len(history) <= max_rows else (
        history[:5] + [None] + history[-5:]
    )
    for row in rows:
        if row is None:
            print("  " + " " * 5 + "  " + "   ...  " + "   (промежуточные строки скрыты)")
            continue
        k, xv, yv, ex, ey = row
        if k == 0:
            print(f"  {k:>5}  {xv:>14.8f}  {yv:>14.8f}  {'—':>12}  {'—':>12}")
        else:
            print(f"  {k:>5}  {xv:>14.8f}  {yv:>14.8f}  {ex:>12.2e}  {ey:>12.2e}")


def print_method_result(name, x, y, history):
    print(f"\n{'═' * 65}")
    print(f"  {name}")
    print(f"{'═' * 65}")
    print_table(history)
    f_val = F(x, y)
    n_iter = len(history) - 1
    print()
    print(f"  Результат   :  x = {x:.10f}   y = {y:.10f}")
    print(f"  Итераций    :  {n_iter}")
    print(f"  Невязка F1  :  {f_val[0]:.4e}")
    print(f"  Невязка F2  :  {f_val[1]:.4e}")


# ──────────────────────────────────────────────────────────
#  Основная программа
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Начальное приближение
    x0, y0 = 1.3, 1.0
    tol    = 1e-6

    print("=" * 65)
    print("  Решение системы нелинейных уравнений")
    print()
    print("    sin(x - y) - xy  = -1")
    print("    x²  -  y²       =  0.75")
    print()
    print(f"  Начальное приближение: x₀ = {x0},  y₀ = {y0}")
    print(f"  Точность:              ε  = {tol}")
    print("=" * 65)

    # ── Метод 1 ──
    x_si, y_si, hist_si = simple_iteration(x0, y0, tol)
    print_method_result("МЕТОД ПРОСТЫХ ИТЕРАЦИЙ", x_si, y_si, hist_si)

    # ── Метод 2 ──
    x_n, y_n, hist_n = newton(x0, y0, tol)
    print_method_result("МЕТОД НЬЮТОНА", x_n, y_n, hist_n)

    # ── Метод 3 ──
    x_z, y_z, hist_z = seidel(x0, y0, tol)
    print_method_result("МЕТОД ЗЕЙДЕЛЯ (нелинейный)", x_z, y_z, hist_z)

    # ── Сводная таблица ──
    print()
    print("═" * 65)
    print("  СВОДНАЯ ТАБЛИЦА СРАВНЕНИЯ МЕТОДОВ")
    print("═" * 65)
    fmt = "  {:<28}  {:>12.10f}  {:>12.10f}  {:>8}"
    hdr = "  {:<28}  {:>12}  {:>12}  {:>8}"
    print(hdr.format("Метод", "x", "y", "Итераций"))
    print("  " + "─" * 63)
    print(fmt.format("Простые итерации",  x_si, y_si, len(hist_si) - 1))
    print(fmt.format("Ньютона",           x_n,  y_n,  len(hist_n)  - 1))
    print(fmt.format("Зейделя",           x_z,  y_z,  len(hist_z)  - 1))
    print()

    # ════════════════════════════════════════════════════════
    #  Графики
    # ════════════════════════════════════════════════════════

    fig = plt.figure(figsize=(15, 6))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    # ── График 1: Кривые пересечения ──
    ax1 = fig.add_subplot(gs[0])
    xm  = np.linspace(-0.5, 2.5, 600)
    ym  = np.linspace(-0.5, 2.5, 600)
    X, Y = np.meshgrid(xm, ym)

    Z1 = np.sin(X - Y) - X * Y + 1.0
    Z2 = X**2 - Y**2 - 0.75

    c1 = ax1.contour(X, Y, Z1, levels=[0], colors='royalblue',   linewidths=2)
    c2 = ax1.contour(X, Y, Z2, levels=[0], colors='crimson',     linewidths=2)

    ax1.plot(x_n, y_n, 'k*', markersize=16, zorder=5,
             label=f'Решение\nx ≈ {x_n:.6f}\ny ≈ {y_n:.6f}')

    h1, _ = c1.legend_elements()
    h2, _ = c2.legend_elements()
    ax1.legend(handles=[h1[0], h2[0], ax1.lines[0]],
               labels=['F₁(x,y) = 0', 'F₂(x,y) = 0', f'Решение\nx≈{x_n:.5f}, y≈{y_n:.5f}'],
               loc='upper left', fontsize=9)
    ax1.set_title('Графическое решение системы', fontsize=13)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.5, 2.5)
    ax1.set_ylim(-0.5, 2.5)

    # ── График 2: Сходимость ──
    ax2 = fig.add_subplot(gs[1])

    def err_seq(hist):
        return [row[3] + row[4] for row in hist[1:]
                if not (np.isnan(row[3]) or np.isnan(row[4]))]

    e_si = err_seq(hist_si)
    e_n  = err_seq(hist_n)
    e_z  = err_seq(hist_z)

    if e_si:
        ax2.semilogy(range(1, len(e_si) + 1), e_si, 'b-o',
                     markersize=4, label=f'Простые итерации ({len(hist_si)-1} ит.)')
    if e_n:
        ax2.semilogy(range(1, len(e_n)  + 1), e_n,  'r-s',
                     markersize=7, label=f'Ньютона ({len(hist_n)-1} ит.)')
    if e_z:
        ax2.semilogy(range(1, len(e_z)  + 1), e_z,  'g-^',
                     markersize=5, label=f'Зейделя ({len(hist_z)-1} ит.)')

    ax2.axhline(tol, color='gray', linestyle='--', linewidth=1, label=f'ε = {tol}')
    ax2.set_title('Сходимость методов', fontsize=13)
    ax2.set_xlabel('Номер итерации')
    ax2.set_ylabel('Суммарная погрешность |Δx| + |Δy|  (лог. шкала)')
    ax2.legend(fontsize=9)
    ax2.grid(True, which='both', alpha=0.3)

    plt.suptitle('Решение нелинейной системы: sin(x−y) − xy = −1,  x²−y² = 0.75',
                 fontsize=12, y=1.01)
    save_path = './convergence.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  График сохранён: {save_path}")
    plt.show()
