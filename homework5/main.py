import numpy as np


def jacobi_eigenvalue(A, epsilon):
    n = len(A)
    V = np.eye(n)
    while True:
        # find the pivot element
        p, q = find_pivot(A)
        if abs(A[p, q]) < epsilon:
            break

        # calculate the rotation angle
        theta = 0.5 * np.arctan2(2 * A[p, q], A[q, q] - A[p, p])

        # create the rotation matrix
        U = np.eye(n)
        U[p, p] = np.cos(theta)
        U[q, q] = np.cos(theta)
        U[p, q] = -np.sin(theta)
        U[q, p] = np.sin(theta)

        # rotate the matrix A
        A = np.dot(np.dot(U.T, A), U)

        # update the matrix V
        V = np.dot(V, U)

    eigen_values = np.diag(A)
    return eigen_values, V


def find_pivot(A):
    # find the pivot element with the largest absolute value
    n = len(A)
    max_val = 0.0
    p = 0
    q = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(A[i, j]) > max_val:
                max_val = abs(A[i, j])
                p = i
                q = j
    return p, q


def norm(A, B, epsilon):
    if np.linalg.norm(A - B) <= epsilon:
        return 0
    return np.linalg.norm(A - B)


def cholesky_factorization(A, epsilon):
    n = len(A)
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                if A[i][i] - s <= 0:
                    raise ValueError("Matrix A is not positive definite")
                L[i][j] = np.sqrt(A[i][i] - s)
            else:
                if L[j][j] == 0:
                    raise ValueError("Division by zero encountered")
                L[i][j] = (1.0 / L[j][j] * (A[i][j] - s))
    return L


def compute_Ak(A, epsilon):
    Ak = A.copy()
    max_iter = 500

    while True:
        max_iter -= 1
        L = np.linalg.cholesky(Ak)
        L_T = np.transpose(L)
        Ak_next = np.dot(L_T, L)
        if norm(Ak, Ak_next, epsilon) == 0:
            break
        Ak = Ak_next

        if max_iter == 0:
            break

    return Ak_next


def main():
    # t = int(input("Enter the number of decimal places for epsilon: "))
    t = 6
    epsilon = 10 ** (-t)

    A = np.array([[1, 0, 1, 0],
                  [0, 1, 0, 1],
                  [1, 0, 1, 0],
                  [0, 1, 0, 1]])

    A_init = A.copy()

    eigen_values, eigen_vectors = jacobi_eigenvalue(A, epsilon)

    # Ex 1
    print("Ex 1:")
    print()
    print("Eigenvalues:")
    V = eigen_values
    print(np.round(V))

    print("\nEigenvectors:")
    U = eigen_vectors
    print(U)

    print("\nNorma || A_init * U - U * V || = ", norm(np.dot(A_init, U), np.dot(U, np.diag(V)), epsilon))

    # Ex 2
    print("\nEx 2:")
    print()
    # for testing the Cholesky factorization i chose a positive definite matrix
    B = np.array([[4, 12, -16],
                  [12, 37, -43],
                  [-16, -43, 98]])
    L = cholesky_factorization(B, epsilon)
    L_np = np.linalg.cholesky(B)
    print("L:")
    print(L)
    print("L_numpy: ")
    print(L_np)
    print("\nNorma || L * L.T - B || = ", norm(np.dot(L, L.T), B, epsilon))
    Ak = compute_Ak(B, epsilon)
    print("\nA_k:")
    print(Ak)

    # Ex 3
    n = 3
    p = 7
    C = np.random.rand(p, n)
    print("\nEx 3:")
    print()
    U, S, V = np.linalg.svd(C)
    print("Singular values:")
    print(S)

    rank_C = np.linalg.matrix_rank(C)
    print("\nRank of C: ", rank_C)

    condition_number = np.linalg.cond(C)
    print("\nCondition number of C: ", condition_number)

    S_plus = np.zeros_like(C.T)
    S_plus[:S.shape[0], :S.shape[0]] = np.diag(1.0 / S)
    C_pseudo_inv = np.dot(np.dot(V.T, S_plus), U.T)
    print("\nC_pseudo_inv:")
    print(C_pseudo_inv)

    C_least_squares = np.dot(np.linalg.inv(np.dot(C.T, C)), C.T)
    print("\nC_least_squares:")
    print(C_least_squares)

    print("\nNorma || C_pseudo_inv - C_least_squares || = ", np.linalg.norm(C_pseudo_inv - C_least_squares))


main()
