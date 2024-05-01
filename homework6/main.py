import numpy as np
import matplotlib.pyplot as plt


def abs_precision(x, y, epsilon):
    if abs(x - y) <= epsilon:
        return 0
    return abs(x - y)


def f(x):
    return x ** 4 - 12 * x ** 3 + 30 * x ** 2 + 12


def newton_forward_interpolation(x, y, x_bar):
    n = len(y)
    pyramid = np.zeros([n, n])
    pyramid[::, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            pyramid[i][j] = (pyramid[i + 1][j - 1] - pyramid[i][j - 1]) / (x[i + j] - x[i])
    prod = 1
    result = pyramid[0][0]
    for i in range(1, n):
        prod *= (x_bar - x[i - 1])
        result += (prod * pyramid[0][i])
    return result


def least_squares_approximation(x, y, x_bar, m):
    A = np.vander(x, m + 1)
    coeffs = np.linalg.lstsq(A, y, rcond=None)[0]
    Pm_x_bar = np.polyval(coeffs, x_bar)
    return Pm_x_bar


def plot_f(x, y, x_bar):
    x_vals = np.linspace(min(x), max(x), 1000)
    y_vals = [f(xi) for xi in x_vals]
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, label='f(x)')
    plt.scatter(x_bar, f(x_bar), color='red')  # punctul pentru f(x_bar)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_Ln(x, y, x_bar):
    Ln_x_bar = newton_forward_interpolation(x, y, x_bar)
    x_vals = np.linspace(min(x), max(x), 1000)
    Ln_vals = [newton_forward_interpolation(x, y, xi) for xi in x_vals]
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, Ln_vals, label='Ln(x)')
    plt.scatter(x_bar, Ln_x_bar, color='red')  # punctul pentru Ln(x_bar)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_Pm(x, y, x_bar, m):
    coeffs = np.polyfit(x, y, m)
    Pm_x_bar = np.polyval(coeffs, x_bar)
    x_vals = np.linspace(min(x), max(x), 1000)
    Pm_vals = [np.polyval(coeffs, xi) for xi in x_vals]
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, Pm_vals, label='Pm(x)')
    plt.scatter(x_bar, Pm_x_bar, color='red')  # punctul pentru Pm(x_bar)
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    # Setare precizie
    t = 5
    epsilon = 10 ** (-t)

    # n = 6, x0 = 1, xn = 5
    n = int(input("Introduceti n: "))
    x0 = float(input("Introduceti x0: "))
    xn = float(input("Introduceti xn: "))
    h = (xn - x0) / n
    x = [x0 + i * h for i in range(n + 1)]
    y = [f(xi) for xi in x]
    # x_bar = 1.5
    x_bar = float(input("Introduceti x_bar: "))

    Ln_x_bar = newton_forward_interpolation(x, y, x_bar)
    print(f"Ln({x_bar}) = {Ln_x_bar}")
    print(f"|Ln({x_bar}) - f({x_bar})| = {abs_precision(Ln_x_bar, f(x_bar), epsilon)}")

    m = int(input("Introduceti m (mai mic decat 6): "))
    Pm_x_bar = least_squares_approximation(x, y, x_bar, m)
    print(f"Pm({x_bar}) = {Pm_x_bar}")
    print(f"|Pm({x_bar}) - f({x_bar})| = {abs_precision(Pm_x_bar, f(x_bar), epsilon)}")
    print(
        f"Suma(|Pm(xi) - yi|) = {sum(abs_precision(np.polyval(np.polyfit(x, y, m), xi), yi, epsilon) for xi, yi in zip(x, y))}")

    plot_f(x, y, x_bar)
    plot_Ln(x, y, x_bar)
    plot_Pm(x, y, x_bar, m)


main()
