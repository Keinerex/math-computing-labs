import copy
import numpy as np
from rich.box import ROUNDED
from rich.table import Table
from rich.console import Console
from rich.panel import Panel
from rich.prompt import IntPrompt

console = Console()


# ── Метод Крамера ─────────────────────────────────────────────────────────────

def det(M):
    n = len(M)
    if n == 1:
        return M[0][0]
    if n == 2:
        return M[0][0] * M[1][1] - M[0][1] * M[1][0]
    return sum(
        (-1) ** j * M[0][j] * det([[M[i][k] for k in range(n) if k != j] for i in range(1, n)])
        for j in range(n)
    )


def cramer(A, b):
    d = det(A)
    assert abs(d) > 1e-12, "det(A) = 0"
    result = []
    for i in range(len(b)):
        Ai = copy.deepcopy(A)
        for r in range(len(b)):
            Ai[r][i] = b[r]
        result.append(det(Ai) / d)
    return result


# ── Метод прогонки (Гаусс) ────────────────────────────────────────────────────

def sweep(A, b):
    n = len(b)
    aug = [A[i][:] + [b[i]] for i in range(n)]

    for k in range(n):
        for i in range(k + 1, n):
            m = aug[i][k] / aug[k][k]
            for j in range(k, n + 1):
                aug[i][j] -= m * aug[k][j]

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (aug[i][n] - sum(aug[i][j] * x[j] for j in range(i + 1, n))) / aug[i][i]
    return x


# ── Метод простой итерации ────────────────────────────────────────────────────

def iteration(A, b, eps=1e-5, max_iter=10000):
    n = len(b)
    beta = [b[i] / A[i][i] for i in range(n)]
    alpha = [
        [-A[i][j] / A[i][i] if i != j else 0.0 for j in range(n)]
        for i in range(n)
    ]
    x = beta[:]

    for k in range(max_iter):
        xn = [beta[i] + sum(alpha[i][j] * x[j] for j in range(n)) for i in range(n)]
        diff = max(abs(xn[i] - x[i]) for i in range(n))

        if diff < eps:
            return xn, k + 1
        if diff > 1e15:
            return None, -(k + 1)
        x = xn

    return x, -max_iter


# ── Главная ───────────────────────────────────────────────────────────────────

def main():
    console.print(Panel("[bold cyan]Лабораторная работа №2 — СЛАУ[/]", expand=False))

    a, b, c, d = (IntPrompt.ask(f"  [yellow]{v}[/]") for v in "abcd")

    A = [
        [1 + a, 14, -15, 23],
        [16, 1 + b, -22, 29],
        [18, 20, -(1 + c), 32],
        [10, 12, -16, 1 + d],
    ]
    B = [5, 8, 9, 4]

    x1 = cramer(A, B)
    x2 = sweep(A, B)
    x3, n_it = iteration(A, B)
    ref = np.linalg.solve(np.array(A, float), np.array(B, float))

    t = Table(show_header=True, header_style="bold magenta", box=ROUNDED, show_edge=True)
    t.add_column("xi", style="bold", justify="center")
    for col in ("Крамер", "Прогонка", "Итерации", "NumPy"):
        t.add_column(col, justify="right")

    for i in range(4):
        it_val = f"{x3[i]:.8f}" if x3 else "[red]расходится[/]"
        t.add_row(f"x{i + 1}", f"{x1[i]:.8f}", f"{x2[i]:.8f}", it_val, f"{ref[i]:.8f}")

    console.print()
    console.print(t)

    status = (
        f"[green]сошлось за {n_it} ит.[/]"
        if n_it > 0 else
        f"[red]расходится ({abs(n_it)} ит.)[/]"
    )
    cond = np.linalg.cond(np.array(A, float))
    console.print(f"\n  Итерации: {status}   cond(A) = [cyan]{cond:.2f}[/]")


if __name__ == "__main__":
    main()
