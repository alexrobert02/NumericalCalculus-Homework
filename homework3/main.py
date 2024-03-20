import numpy as np


def is_zero_with_precision(value, eps):
    return abs(value) < eps


# Ex 1
def calcul_vector_b(A,s):
    n = len(s)
    b = np.zeros(n)
    for i in range(n):
        for j in range(n):
            b[i] += s[j] * A[i][j]
    return b


def transposition(A):
    n = len(A)
    B = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            B[i][j] = A[j][i]
    return B


# Ex 2
def householder_QR(A, epsilon):
    n = len(A)
    Q = np.identity(n)
    for i in range(n):
        x = A[i:, i]
        e = np.zeros(len(x))
        e[0] = 1
        v = x - np.linalg.norm(x) * e
        H = np.identity(n)
        for j in range(len(v)):
            for k in range(len(v)):
                if not is_zero_with_precision(np.dot(v, v), epsilon) :
                    H[i + j][i + k] -= 2 * v[j] * v[k] / np.dot(v, v)
        Q = np.dot(Q, H)
        A = np.dot(H, A)
    return Q, A


# Ex 3
def solve_linear_system_numpy(A, b):
    Q, R = np.linalg.qr(A)
    Qt_b = np.dot(Q.T, b)
    x = np.linalg.solve(R, Qt_b)
    return x


def solve_linear_system(Q, R, b):
    Qt = transposition(Q)
    Qt_b = np.dot(Qt, b)
    x = np.linalg.solve(R, Qt_b)
    return x


def norm(x, y):
    return np.linalg.norm(x-y)


# Ex 5
def inverse_using_qr(Q, R):
    n = Q.shape[0]
    inverse = np.zeros((n, n))

    # Rezolvăm sistemul pentru fiecare coloană a matricei inverse
    for i in range(n):
        Qt = transposition(Q)
        b = np.zeros(n)
        b[i] = 1  # Vectorul de pe coloana i a matricei identitate
        Qt_b = np.dot(Qt, b)  # Q^T * b
        x = np.linalg.solve(R, Qt_b)  # Rezolvăm sistemul Rx = Q^T * b pentru fiecare coloană
        inverse[:, i] = x

    return inverse


def afisare(A,s,epsilon):
    A_init = A
    print("Ex 1")
    print("vectorul b:")
    b = calcul_vector_b(A, s)
    b_init = b
    print(b)
    print()
    print("Ex 2")
    Q, R = householder_QR(A, epsilon)
    print("Q:")
    print(Q)
    print("R:")
    print(R)
    print()
    print("Ex 3")
    x_numpy = solve_linear_system_numpy(A, b)
    print("x_numpy :", x_numpy)
    x_householder = solve_linear_system(Q, R, b)
    print("x_householder:", x_householder)
    print("Norma euclidiana : ", norm(x_numpy, x_householder))
    print()
    # Ex 4
    print("Ex 4")
    print("|| A_init * x_householder - b_init || = ", norm(np.dot(A_init, x_householder), b_init))
    print("|| A_init * x_numpy - b_init || = ", norm(np.dot(A_init, x_numpy), b_init))
    print("|| x_householder - s || / || s || = ", norm(x_householder, s) / np.linalg.norm(s))
    print("|| x_numpy - s || / || s || = ", norm(x_numpy, s) / np.linalg.norm(s))
    print()

    # Ex 5
    print("Ex 5")
    A_inverse = inverse_using_qr(Q, R)
    print("A_inverse:")
    print(A_inverse)
    A_inverse_numpy = np.linalg.inv(A)
    print("A_inverse_numpy:")
    print(A_inverse_numpy)
    print("|| A_inverse - A_inverse_numpy || = ", norm(A_inverse, A_inverse_numpy))


def main():

    # t = int(input("Introduceti precizia pentru verificarea impartirii la zero: "))
    t=5
    epsilon = 10 ** (-t)

    # Testare pe matricea data in enunt
    A = np.array([[0, 0, 4], [1, 2, 3], [0, 1, 2]])
    s = np.array([3, 2, 1])

    # Testare pe o matrice de dimensiune mai mare
    n = 100
    A_large = np.random.rand(n, n)
    s_large = np.random.rand(n)

    afisare(A, s, epsilon)

main()
