import numpy as np


# Functia pentru descompunerea LU
def lu_decomposition(A):
    n = len(A)
    l = np.zeros(n * (n + 1) // 2)
    u = np.zeros(n * (n + 1) // 2)
    for i in range(n):
        for j in range(i + 1):
            u[j * (j + 1) // 2 + i] = A[j][i] - sum(u[k * (k + 1) // 2 + i] * l[j * (j + 1) // 2 + k] for k in range(j))
        for j in range(i, n):
            l[j * (j + 1) // 2 + i] = (A[j][i] - sum(
                u[k * (k + 1) // 2 + i] * l[j * (j + 1) // 2 + k] for k in range(i))) / u[i * (i + 1) // 2 + i]
    return l, u


# Functia pentru rezolvarea sistemului
def solve_system(l, u, b, epsilon):
    n = len(b)
    y = np.zeros(n)
    x = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - sum(l[i * (i + 1) // 2 + j] * y
        [j] for j in range(i))
    for i in range(n - 1, -1, -1):
        if is_zero_with_precision(u[i * (i + 1) // 2 + i], epsilon):
            raise ValueError("Nu se poate face impartirea la zero!")
        else:
            x[i] = (y[i] - sum(u[i * (i + 1) // 2 + j] * x[j] for j in range(i + 1, n))) / u[i * (i + 1) // 2 + i]
    return x


# Functia pentru calculul determinantului
def determinant(A, m):
    l, u = lu_decomposition(A)
    det_l = np.prod([l[i * (i + 1) // 2 + i] for i in range(m)])
    det_u = np.prod([u[i * (i + 1) // 2 + i] for i in range(m)])
    return det_l * det_u


# Functia pentru calculul normei
def calculate_norm(A, x, b):
    return np.linalg.norm(np.dot(A, x) - b)


def is_zero_with_precision(value, eps):
    return abs(value) < eps


def solve_and_print(A, b, epsilon):
    l, u = lu_decomposition(A)
    x = solve_system(l, u, b, epsilon)

    # Calculul solutiei sistemului Ax = b folosind biblioteca NumPy
    x_numpy = np.linalg.solve(A, b)

    # Calculul inversa matricei A folosind biblioteca NumPy
    A_inv_numpy = np.linalg.inv(A)

    # Norma euclidiană ||x_LU − x_numpy||
    norm1 = np.linalg.norm(x - x_numpy)

    # Calculul produsului matricial [(A_numpy)-1]*b_init
    A_inv_b_init = np.dot(A_inv_numpy, b)

    # Calculul normei
    norm = calculate_norm(A, x, b)

    # Norma euclidiană ||x_LU − [(A_numpy)-1]*b_init||
    norm2 = np.linalg.norm(x - A_inv_b_init)

    return {
        "Valorile corespunzatoare matricei L sunt": l,
        "Valorile corespunzatoare matricei U sunt": u,
        "Determinantul matricei A": determinant(A, len(b)),
        "Norma": norm,
        "Solutia sistemului Ax = b folosind descompunerea LU": x,
        "Solutia sistemului Ax = b folosind biblioteca NumPy": x_numpy,
        "Inversa matricei A folosind biblioteca NumPy": A_inv_numpy,
        "Norma euclidiana ||x_LU − x_numpy||": norm1,
        "Norma euclidiana ||x_LU − [(A_numpy)-1]*b_init||": norm2
    }


def main():
    t = int(input("Introduceti precizia pentru verificarea impartirii la zero: "))
    epsilon = 10 ** (-t)

    # Testare pe o matrice de dimensiune mai mare
    n = 100
    A_large = np.random.rand(n, n)
    b_large = np.random.rand(n)

    # Testarea pe exemplul din tema
    A = np.array([[2.5, 2, 2], [5, 6, 5], [5, 6, 6.5]], dtype=float)
    b = np.array([2, 2, 2], dtype=float)

    result = solve_and_print(A, b, epsilon)

    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
