import math


def f(x):
    return math.log10(x) - 5 / (2 * x + 3)


def df(x):
    return 1 / (x * math.log(10)) + 10 / (2 * x + 3) ** 2


def bisection(a, b, epsilon=1e-4, max_iter=1000):
    iteration = 0
    history = []
    fa = f(a)
    fb = f(b)

    if fa * fb > 0:
        raise ValueError("Для метода бисекции f(a) и f(b) должны иметь разные знаки.")

    while abs(b - a) > epsilon:
        if iteration > max_iter:
            raise RuntimeError("Превышено число итераций (bisection)")

        x = (a + b) / 2
        fx = f(x)

        history.append((iteration, a, b, x, fx))

        if fa * fx < 0:
            b = x
            fb = fx
        else:
            a = x
            fa = fx

        iteration += 1

    return (a + b) / 2, iteration, history


def chord_method(a, b, epsilon=1e-4, max_iter=1000):
    x = a
    iteration = 0
    history = []

    fb = f(b)

    while True:
        if iteration > max_iter:
            raise RuntimeError("Превышено число итераций (chord)")

        fx = f(x)

        denominator = fb - fx
        if abs(denominator) < 1e-12:
            raise ValueError("Деление на ноль в методе хорд")

        x_next = x - fx * (b - x) / denominator

        history.append((iteration, x, x_next, abs(x_next - x)))

        if abs(x_next - x) <= epsilon:
            return x_next, iteration, history

        x = x_next
        iteration += 1


def newton_method(x0, epsilon=1e-4, max_iter=1000):
    x = x0
    iteration = 0
    history = []

    while True:
        if iteration > max_iter:
            raise RuntimeError("Превышено число итераций (Newton)")

        fx = f(x)
        dfx = df(x)

        if abs(dfx) < 1e-12:
            raise ValueError("Производная близка к нулю")

        x_next = x - fx / dfx

        history.append((iteration, x, x_next, abs(x_next - x)))

        if abs(x_next - x) <= epsilon:
            return x_next, iteration, history

        x = x_next
        iteration += 1


def format_number(value, digits=6):
    return f"{value:.{digits}f}"


def select_rows(history, max_rows=20):
    if len(history) <= max_rows:
        return history, False
    half = max_rows // 2
    rows = history[:half] + history[-(max_rows - half):]
    return rows, True


def print_result(title, root, iterations, history, columns, epsilon):
    print(f"\n{title}")
    print("-" * len(title))
    print(f"Точность epsilon: {epsilon}")
    print(f"Найденный корень x: {format_number(root)}")
    print(f"Проверка f(x): {f(root):.6e}")
    print(f"Число итераций: {iterations}\n")

    rows, is_truncated = select_rows(history)

    print("История итераций:")
    print(" | ".join(columns))
    print("-" * (len(" | ".join(columns)) + 4))

    for row in rows:
        formatted = [str(row[0])] + [format_number(v) for v in row[1:]]
        print(" | ".join(formatted))

    if is_truncated:
        print("... (показаны первые и последние итерации)")


if __name__ == "__main__":
    epsilon = 1e-4
    a, b = 3, 4

    # Метод половинного деления
    root_bis, iter_bis, hist_bis = bisection(a, b, epsilon)
    print_result(
        "Метод половинного деления",
        root_bis,
        iter_bis,
        hist_bis,
        columns=("Итерация", "a", "b", "x", "f(x)"),
        epsilon=epsilon,
    )

    # Метод хорд
    root_chord, iter_chord, hist_chord = chord_method(a, b, epsilon)
    print_result(
        "Метод хорд",
        root_chord,
        iter_chord,
        hist_chord,
        columns=("Итерация", "x_k", "x_k+1", "|Δ|"),
        epsilon=epsilon,
    )

    root_newton, iter_newton, hist_newton = newton_method(4, epsilon)
    print_result(
        "Метод Ньютона",
        root_newton,
        iter_newton,
        hist_newton,
        columns=("Итерация", "x_k", "x_k+1", "|Δ|"),
        epsilon=epsilon,
    )

    print("\nКраткий итог:")
    print(
        f"Бисекция: x={format_number(root_bis)}, итераций={iter_bis}, f(x)={f(root_bis):.2e}"
    )
    print(
        f"Хорды:    x={format_number(root_chord)}, итераций={iter_chord}, f(x)={f(root_chord):.2e}"
    )
    print(
        f"Ньютон:   x={format_number(root_newton)}, итераций={iter_newton}, f(x)={f(root_newton):.2e}"
    )
