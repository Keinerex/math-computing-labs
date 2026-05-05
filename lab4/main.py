"""
=============================================================
Лабораторная работа: Аппроксимация и интерполяция функции

  f(x) = lg(x) - 5 / (2x + 3)

Цели:
  1. Аппроксимация многочленом Тейлора: зависимость ошибки от
     степени n и от положения точки разложения x0
  2. Интерполяция многочленом Лагранжа (равноотстоящие узлы)
  3. Интерполяция многочленом Ньютона (разделённые разности)
  4. «Скользящая» интерполяция
  5. Сравнительный анализ и выводы
=============================================================
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import sympy as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

# ──────────────────────────────────────────────────────────
#  Функция и область определения
# ──────────────────────────────────────────────────────────
x_sym = sp.Symbol("x")
f_sym = sp.log(x_sym, 10) - sp.Rational(5, 1) / (2 * x_sym + 3)

def f(x):
    return np.log10(x) - 5.0 / (2.0 * x + 3.0)

A, B = 1.0, 5.0           # интервал
N_PLOT = 800               # точки для графиков
x_plot = np.linspace(A, B, N_PLOT)
y_exact = f(x_plot)


# ══════════════════════════════════════════════════════════
# 1. АППРОКСИМАЦИЯ МНОГОЧЛЕНОМ ТЕЙЛОРА
# ══════════════════════════════════════════════════════════

def taylor_coeffs(x0_val: float, n: int) -> list:
    """Коэффициенты Тейлора c_k = f^(k)(x0) / k!"""
    coeffs = []
    deriv = f_sym
    for k in range(n + 1):
        val = float(deriv.subs(x_sym, x0_val))
        coeffs.append(val / float(sp.factorial(k)))
        deriv = sp.diff(deriv, x_sym)
    return coeffs

def eval_taylor(x_arr: np.ndarray, x0_val: float, coeffs: list) -> np.ndarray:
    """P_n(x) = Σ c_k * (x - x0)^k"""
    result = np.zeros_like(x_arr, dtype=float)
    for k, c in enumerate(coeffs):
        result += c * (x_arr - x0_val) ** k
    return result

def max_abs_error(y_approx: np.ndarray, y_ref: np.ndarray) -> float:
    return float(np.max(np.abs(y_approx - y_ref)))


# ──── Таблица 1: ошибка Тейлора vs степень n (фиксир. x0) ────
x0_fixed = 3.0
degrees  = [1, 2, 3, 4, 5, 6, 7, 8]

print("=" * 65)
print("  f(x) = lg(x) - 5/(2x+3)     Интервал: [{}, {}]".format(A, B))
print("=" * 65)

print("\n── Таблица 1. Ошибка аппроксимации Тейлора vs степень n")
print(f"   (точка разложения x₀ = {x0_fixed})")
print(f"   {'n':>4}   {'Макс. абс. погрешность':>24}   {'Макс. отн. погрешность':>24}")
print("   " + "─" * 58)

taylor_errors_n = []
for n in degrees:
    c = taylor_coeffs(x0_fixed, n)
    yp = eval_taylor(x_plot, x0_fixed, c)
    err_abs = max_abs_error(yp, y_exact)
    err_rel = float(np.max(np.abs((yp - y_exact) / y_exact)))
    taylor_errors_n.append((n, err_abs, err_rel))
    print(f"   {n:>4}   {err_abs:>24.6e}   {err_rel:>24.6e}")

# ──── Таблица 2: ошибка Тейлора vs положение x0 (фиксир. n) ────
n_fixed  = 5
x0_vals  = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]

print(f"\n── Таблица 2. Ошибка аппроксимации Тейлора vs точка x₀")
print(f"   (степень многочлена n = {n_fixed})")
print(f"   {'x₀':>6}   {'Макс. абс. погрешность':>24}   {'Макс. отн. погрешность':>24}")
print("   " + "─" * 62)

taylor_errors_x0 = []
for x0v in x0_vals:
    c = taylor_coeffs(x0v, n_fixed)
    yp = eval_taylor(x_plot, x0v, c)
    err_abs = max_abs_error(yp, y_exact)
    err_rel = float(np.max(np.abs((yp - y_exact) / y_exact)))
    taylor_errors_x0.append((x0v, err_abs, err_rel))
    print(f"   {x0v:>6.2f}   {err_abs:>24.6e}   {err_rel:>24.6e}")


# ══════════════════════════════════════════════════════════
# 2. ИНТЕРПОЛЯЦИЯ МНОГОЧЛЕНОМ ЛАГРАНЖА
# ══════════════════════════════════════════════════════════

def lagrange_interp(x_nodes: np.ndarray, y_nodes: np.ndarray,
                    x_eval: np.ndarray) -> np.ndarray:
    """Вычисляет многочлен Лагранжа в точках x_eval"""
    n = len(x_nodes)
    result = np.zeros_like(x_eval, dtype=float)
    for i in range(n):
        L_i = np.ones_like(x_eval, dtype=float)
        for j in range(n):
            if i != j:
                L_i *= (x_eval - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        result += y_nodes[i] * L_i
    return result

# ──── Таблица 3: ошибка Лагранжа vs число узлов ────
node_counts = [3, 4, 5, 6, 7, 8, 10, 12]

print("\n── Таблица 3. Ошибка интерполяции Лагранжа vs число узлов n+1")
print("   (равноотстоящие узлы на [{}, {}])".format(A, B))
print(f"   {'Узлов':>6}   {'Макс. абс. погрешность':>24}   {'Макс. отн. погрешность':>24}")
print("   " + "─" * 62)

lagrange_errors = []
for nc in node_counts:
    x_nodes = np.linspace(A, B, nc)
    y_nodes = f(x_nodes)
    yp = lagrange_interp(x_nodes, y_nodes, x_plot)
    err_abs = max_abs_error(yp, y_exact)
    err_rel = float(np.max(np.abs((yp - y_exact) / y_exact)))
    lagrange_errors.append((nc, err_abs, err_rel))
    print(f"   {nc:>6}   {err_abs:>24.6e}   {err_rel:>24.6e}")


# ══════════════════════════════════════════════════════════
# 3. ИНТЕРПОЛЯЦИЯ МНОГОЧЛЕНОМ НЬЮТОНА
# ══════════════════════════════════════════════════════════

def divided_differences(x_nodes: np.ndarray, y_nodes: np.ndarray) -> np.ndarray:
    """Таблица разделённых разностей"""
    n = len(x_nodes)
    dd = y_nodes.copy().astype(float)
    table = [dd.copy()]
    for order in range(1, n):
        dd_new = np.zeros(n - order)
        for i in range(n - order):
            dd_new[i] = (dd[i + 1] - dd[i]) / (x_nodes[i + order] - x_nodes[i])
        dd = dd_new
        table.append(dd.copy())
    return table   # table[k][0] — k-я разделённая разность от x_0

def newton_interp(x_nodes: np.ndarray, y_nodes: np.ndarray,
                  x_eval: np.ndarray) -> np.ndarray:
    """Многочлен Ньютона через разделённые разности"""
    table = divided_differences(x_nodes, y_nodes)
    n = len(x_nodes)
    result = np.full_like(x_eval, table[0][0], dtype=float)
    omega = np.ones_like(x_eval, dtype=float)
    for k in range(1, n):
        omega *= (x_eval - x_nodes[k - 1])
        result += table[k][0] * omega
    return result

# ──── Таблица 4: Ньютон vs Лагранж (сравнение) ────
print("\n── Таблица 4. Сравнение Лагранжа и Ньютона (макс. абс. погрешность)")
print("   (равноотстоящие узлы на [{}, {}])".format(A, B))
print(f"   {'Узлов':>6}   {'Лагранж':>18}   {'Ньютон':>18}   {'|Δ|':>14}")
print("   " + "─" * 64)

newton_errors = []
for nc in node_counts:
    x_nodes = np.linspace(A, B, nc)
    y_nodes = f(x_nodes)
    yL = lagrange_interp(x_nodes, y_nodes, x_plot)
    yN = newton_interp(x_nodes, y_nodes, x_plot)
    eL = max_abs_error(yL, y_exact)
    eN = max_abs_error(yN, y_exact)
    newton_errors.append((nc, eN))
    print(f"   {nc:>6}   {eL:>18.6e}   {eN:>18.6e}   {abs(eL-eN):>14.2e}")


# ══════════════════════════════════════════════════════════
# 4. «СКОЛЬЗЯЩАЯ» ИНТЕРПОЛЯЦИЯ
# ══════════════════════════════════════════════════════════

def sliding_interp(x_table: np.ndarray, y_table: np.ndarray,
                   x_eval: np.ndarray, deg: int) -> np.ndarray:
    """
    Для каждой точки x_eval выбирает deg+1 ближайших узлов
    и строит локальный многочлен Ньютона.
    """
    result = np.zeros_like(x_eval, dtype=float)
    m = len(x_table)
    k = deg + 1   # число узлов

    for idx, xv in enumerate(x_eval):
        # Найти k ближайших узлов
        dists = np.abs(x_table - xv)
        order = np.argsort(dists)[:k]
        order = np.sort(order)               # сохранить порядок
        xn = x_table[order]
        yn = y_table[order]
        result[idx] = newton_interp(xn, yn, np.array([xv]))[0]
    return result

# Таблица узлов для скользящей интерполяции
N_TABLE = 20
x_table = np.linspace(A, B, N_TABLE)
y_table = f(x_table)

sliding_degrees = [1, 2, 3, 4, 5]

print("\n── Таблица 5. «Скользящая» интерполяция vs степень многочлена")
print(f"   (таблица {N_TABLE} узлов на [{A}, {B}])")
print(f"   {'Степень':>8}   {'Макс. абс. погрешность':>24}   {'Макс. отн. погрешность':>24}")
print("   " + "─" * 64)

sliding_errors = []
for deg in sliding_degrees:
    yp = sliding_interp(x_table, y_table, x_plot, deg)
    err_abs = max_abs_error(yp, y_exact)
    err_rel = float(np.max(np.abs((yp - y_exact) / y_exact)))
    sliding_errors.append((deg, err_abs, err_rel))
    print(f"   {deg:>8}   {err_abs:>24.6e}   {err_rel:>24.6e}")


# ══════════════════════════════════════════════════════════
# 5. ГРАФИКИ
# ══════════════════════════════════════════════════════════

COLORS = plt.cm.tab10.colors

# ════════════════ Рисунок 1: Тейлор ════════════════
fig1, axes1 = plt.subplots(1, 2, figsize=(15, 5))
fig1.suptitle("Аппроксимация многочленом Тейлора   f(x) = lg(x) − 5/(2x+3)", fontsize=13)

ax = axes1[0]
ax.plot(x_plot, y_exact, "k-", lw=2.5, label="f(x) точная", zorder=5)
for i, n in enumerate([2, 4, 6, 8]):
    c = taylor_coeffs(x0_fixed, n)
    yp = eval_taylor(x_plot, x0_fixed, c)
    ax.plot(x_plot, yp, lw=1.5, color=COLORS[i], label=f"T_{n}(x), x₀={x0_fixed}")
ax.axvline(x0_fixed, color="gray", ls="--", lw=1, alpha=0.7, label=f"x₀ = {x0_fixed}")
ax.set_xlim(A, B); ax.set_ylim(-0.5, 1.5)
ax.set_title(f"Многочлены Тейлора разных степеней (x₀ = {x0_fixed})")
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes1[1]
ns   = [r[0] for r in taylor_errors_n]
errs = [r[1] for r in taylor_errors_n]
ax.semilogy(ns, errs, "b-o", markersize=7, lw=2)
ax.set_title(f"Ошибка Тейлора vs степень n  (x₀ = {x0_fixed})")
ax.set_xlabel("Степень n"); ax.set_ylabel("Макс. абс. погрешность (лог. шкала)")
ax.grid(True, which="both", alpha=0.3)
for n, e in zip(ns, errs):
    ax.annotate(f"{e:.1e}", (n, e), textcoords="offset points",
                xytext=(4, 4), fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "fig1_taylor.png"), dpi=150, bbox_inches="tight")
print("\n  Сохранён: fig1_taylor.png")
plt.close()

# ════════════════ Рисунок 2: Тейлор vs x0 ════════════════
fig2, axes2 = plt.subplots(1, 2, figsize=(15, 5))
fig2.suptitle(f"Тейлор: влияние положения точки разложения  (n = {n_fixed})", fontsize=13)

ax = axes2[0]
ax.plot(x_plot, y_exact, "k-", lw=2.5, label="f(x) точная", zorder=5)
for i, x0v in enumerate(x0_vals):
    c = taylor_coeffs(x0v, n_fixed)
    yp = eval_taylor(x_plot, x0v, c)
    ax.plot(x_plot, yp, lw=1.5, color=COLORS[i % 10], label=f"x₀={x0v}")
ax.set_xlim(A, B); ax.set_ylim(-0.5, 1.5)
ax.set_title(f"Многочлены Тейлора степени {n_fixed} при разных x₀")
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)

ax = axes2[1]
x0s  = [r[0] for r in taylor_errors_x0]
errs = [r[1] for r in taylor_errors_x0]
ax.plot(x0s, errs, "r-s", markersize=8, lw=2)
ax.set_title(f"Ошибка Тейлора vs точка разложения x₀  (n = {n_fixed})")
ax.set_xlabel("x₀"); ax.set_ylabel("Макс. абс. погрешность")
ax.grid(True, alpha=0.3)
for x0v, e in zip(x0s, errs):
    ax.annotate(f"{e:.2e}", (x0v, e), textcoords="offset points",
                xytext=(4, 4), fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "fig2_taylor_x0.png"), dpi=150, bbox_inches="tight")
print("  Сохранён: fig2_taylor_x0.png")
plt.close()

# ════════════════ Рисунок 3: Лагранж и Ньютон ════════════════
fig3, axes3 = plt.subplots(1, 2, figsize=(15, 5))
fig3.suptitle("Интерполяция многочленами Лагранжа и Ньютона", fontsize=13)

ax = axes3[0]
ax.plot(x_plot, y_exact, "k-", lw=2.5, label="f(x) точная", zorder=5)
for i, nc in enumerate([3, 5, 8, 10]):
    x_nodes = np.linspace(A, B, nc)
    y_nodes = f(x_nodes)
    yL = lagrange_interp(x_nodes, y_nodes, x_plot)
    ax.plot(x_plot, yL, lw=1.5, color=COLORS[i], label=f"Лагранж, {nc} узла")
    ax.plot(x_nodes, y_nodes, "o", color=COLORS[i], markersize=5)
ax.set_xlim(A, B); ax.set_ylim(-0.3, 1.4)
ax.set_title("Многочлены Лагранжа")
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes3[1]
ncs  = [r[0] for r in lagrange_errors]
erL  = [r[1] for r in lagrange_errors]
erN  = [r[1] for r in newton_errors]
ax.semilogy(ncs, erL, "b-o", markersize=7, lw=2, label="Лагранж")
ax.semilogy(ncs, erN, "r--s", markersize=6, lw=1.8, label="Ньютон")
ax.set_title("Ошибка интерполяции vs число узлов")
ax.set_xlabel("Число узлов"); ax.set_ylabel("Макс. абс. погрешность (лог. шкала)")
ax.legend(); ax.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "fig3_lagrange_newton.png"), dpi=150, bbox_inches="tight")
print("  Сохранён: fig3_lagrange_newton.png")
plt.close()

# ════════════════ Рисунок 4: Скользящая интерполяция ════════════════
fig4, axes4 = plt.subplots(1, 2, figsize=(15, 5))
fig4.suptitle(f"«Скользящая» интерполяция  (таблица {N_TABLE} узлов)", fontsize=13)

ax = axes4[0]
ax.plot(x_plot, y_exact, "k-", lw=2.5, label="f(x) точная", zorder=5)
ax.plot(x_table, y_table, "k|", markersize=10, label="Узлы таблицы")
for i, deg in enumerate([1, 2, 3, 5]):
    yp = sliding_interp(x_table, y_table, x_plot, deg)
    ax.plot(x_plot, yp, lw=1.5, color=COLORS[i], label=f"Степень {deg}")
ax.set_xlim(A, B); ax.set_ylim(-0.3, 1.4)
ax.set_title("Кривые скользящей интерполяции")
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes4[1]
degs = [r[0] for r in sliding_errors]
errs = [r[1] for r in sliding_errors]
ax.semilogy(degs, errs, "g-^", markersize=9, lw=2)
ax.set_title("Ошибка скользящей интерполяции vs степень")
ax.set_xlabel("Степень многочлена"); ax.set_ylabel("Макс. абс. погрешность (лог. шкала)")
ax.grid(True, which="both", alpha=0.3)
for d, e in zip(degs, errs):
    ax.annotate(f"{e:.1e}", (d, e), textcoords="offset points",
                xytext=(4, 4), fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "fig4_sliding.png"), dpi=150, bbox_inches="tight")
print("  Сохранён: fig4_sliding.png")
plt.close()

# ════════════════ Рисунок 5: Сводное сравнение ════════════════
fig5, ax5 = plt.subplots(figsize=(10, 6))
fig5.suptitle("Сводное сравнение методов аппроксимации и интерполяции", fontsize=13)

# Тейлор (n: 1..8, x0=3)
ax5.semilogy([r[0] for r in taylor_errors_n],
             [r[1] for r in taylor_errors_n],
             "b-o", markersize=6, lw=1.8, label=f"Тейлор (x₀={x0_fixed})")
# Лагранж (узлы 3..12)
ax5.semilogy([r[0] for r in lagrange_errors],
             [r[1] for r in lagrange_errors],
             "r-s", markersize=6, lw=1.8, label="Лагранж (равноотст. узлы)")
# Ньютон
ax5.semilogy([r[0] for r in newton_errors],
             [r[1] for r in newton_errors],
             "r--^", markersize=5, lw=1.3, label="Ньютон (равноотст. узлы)")
# Скользящая (узлы фиксированы=20, степени 1..5)
ax5.semilogy([r[0] for r in sliding_errors],
             [r[1] for r in sliding_errors],
             "g-D", markersize=7, lw=1.8, label=f"Скользящая (таблица {N_TABLE} узлов)")

ax5.set_xlabel("Степень многочлена / число узлов")
ax5.set_ylabel("Макс. абс. погрешность (лог. шкала)")
ax5.legend(fontsize=10)
ax5.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "fig5_comparison.png"), dpi=150, bbox_inches="tight")
print("  Сохранён: fig5_comparison.png")
plt.close()