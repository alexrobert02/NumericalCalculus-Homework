import numpy as np


# Citirea datelor din fișiere
def read_sparse_matrix(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        n = int(lines[0])
        elements = []
        for idx, line in enumerate(lines[1:], start=1):
            if line.strip() == "":
                continue
            values = line.split(',')
            try:
                val = float(values[0].strip())
                i = int(values[1].strip())
                j = int(values[2].strip())
                elements.append((val, i, j))
            except ValueError:
                print(f"Error reading line {idx}: {line}")
                raise
        return n, elements


def read_vector(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        n = int(lines[0])
        b = []
        for line in lines[1:]:
            if line.strip():  # Verificăm dacă linia nu este goală
                try:
                    b.append(float(line.strip()))
                except ValueError:
                    print("Eroare la citirea datelor. Verificați formatul fișierului.")
                    exit()
        if len(b) != n:
            print("Dimensiunea vectorului termenilor liberi nu este compatibilă cu dimensiunea sistemului.")
            exit()
        return n, b


# Verificare elemente nenule pe diagonală
def check_diagonal(elements):
    diagonal_elements = [val for val, i, j in elements if i == j]
    for val in diagonal_elements:
        if val == 0:
            return False
    return True


# Metoda Gauss-Seidel pentru aproximarea soluției sistemului liniar
def gauss_seidel(A, b, epsilon=1e-10, max_iterations=1000):
    n = len(b)
    x = np.zeros(n)
    x_new = np.zeros(n)
    iterations = 0
    print("\n")
    while iterations < max_iterations:
        for i in range(n):
            s1 = sum(val * x_new[j] for j, val in A[i] if j < i)
            s2 = sum(val * x[j] for j, val in A[i] if j > i)
            diag_val = next((val for j, val in A[i] if j == i), None)
            if diag_val is None or abs(diag_val) < epsilon:
                print("Zero division or small diagonal encountered. Adjust your algorithm.")
                return None, iterations
            # Evitarea overflow-ului și a valorilor invalide
            temp_result = (b[i] - s1 - s2) / diag_val
            if abs(temp_result) > 1e100:  # Limitare pentru a evita overflow
                x_new[i] = np.sign(temp_result) * 1e100
            else:
                x_new[i] = temp_result
        if np.any(np.isnan(x_new)):
            print("Solution became NaN. Adjust your algorithm or input.")
            return None, iterations
        if np.linalg.norm(x_new - x, np.inf) < epsilon:
            break
        print(f"Iteration {iterations + 1}: {x_new}")
        x = np.copy(x_new)
        iterations += 1
    else:
        x_new = None
    return x_new, iterations


# Calcularea produsului rar între matricea rară și un vector
def sparse_matrix_vector_product(A, x):
    result = np.zeros(len(x))
    for i in range(len(A)):
        for j, val in A[i]:
            result[i] += val * x[j]
    return result


# Calcularea normei diferenței între doi vectori
def calculate_norm_difference(vector1, vector2):
    norm_diff = 0
    for i in range(len(vector1)):
        norm_diff += abs(vector1[i] - vector2[i])
    return norm_diff


# Implementarea metodei Gauss-Seidel folosind memorarea rară a matricelor
def sparse_matrix_to_array(elements, n):
    A = [[] for _ in range(n)]  # Inițializăm un vector rar pentru fiecare linie a matricei
    for val, i, j in elements:
        A[i].append([j, val])  # Stocăm perechile (col_index, value) pentru fiecare element nenul
    return A


def check_solutions():
    # Rezolvarea pentru fiecare pereche de fișiere
    for i in range(1, 6):
        # Citirea datelor din fișiere
        n, elements = read_sparse_matrix(f"a_{i}.txt")
        _, b = read_vector(f"b_{i}.txt")

        # Verificare elemente nenule pe diagonală
        if not check_diagonal(elements):
            print(f"Elementele de pe diagonală ale matricei A din fisierul a_{i}.txt sunt nule.")
            continue

        A = sparse_matrix_to_array(elements, n)

        # Calcularea soluției aproximative a sistemului liniar și numărul de iterații
        x_approx, iterations = gauss_seidel(A, b)

        if x_approx is not None:
            # Verificarea soluției calculată
            product = sparse_matrix_vector_product(A, x_approx)
            norm = calculate_norm_difference(product, b)
            print(f"\nSolutia aproximativa a sistemului liniar pentru fisierele a_{i}.txt si b_{i}.txt:")
            print("Solutia aproximativa:", x_approx)
            print("Numarul de iteratii:", iterations)
            print("Norma diferenței:", norm)
        else:
            print(f"\nSolutia sistemului liniar pentru fisierele a_{i}.txt si b_{i}.txt a întâmpinat probleme.")


# Funcție pentru verificarea sumei matricilor
def check_sum_of_matrices():
    # Citirea matricilor a, b și aplusb
    n_a, elements_a = read_sparse_matrix("a.txt")
    n_b, elements_b = read_sparse_matrix("b.txt")

    n_aplusb, elements_aplusb = read_sparse_matrix("aplusb.txt")

    # Verificăm dacă dimensiunile matricilor sunt compatibile
    if n_a != n_b or n_b != n_aplusb:
        print("Dimensiunile matricilor nu sunt compatibile.")
        return False

    # Inițializăm matricile a, b și aplusb
    matrix_a = np.zeros((n_a, n_a))
    matrix_b = np.zeros((n_b, n_b))
    matrix_aplusb = np.zeros((n_aplusb, n_aplusb))

    # Umplem matricile a, b și aplusb cu elementele corespunzătoare
    for val, i, j in elements_a:
        matrix_a[i][j] = val
    for val, i, j in elements_b:
        matrix_b[i][j] = val
    for val, i, j in elements_aplusb:
        matrix_aplusb[i][j] = val

    # Verificarea condițiilor pentru fiecare element al matricilor
    for i in range(n_a):
        for j in range(n_a):
            if matrix_a[i][j] != 0 and matrix_b[i][j] != 0 and matrix_aplusb[i][j] != 0:
                if ((matrix_a[i][j] + matrix_b[i][j] != 0 or matrix_aplusb[i][j] != 0) and
                        (matrix_a[j][i] + matrix_b[j][i] != 0 or matrix_aplusb[j][i] != 0)):
                    if (matrix_a[i][j] + matrix_b[i][j] != matrix_aplusb[i][j] or
                            matrix_a[j][i] + matrix_b[j][i] != matrix_aplusb[j][i]):
                        print(f"Elementele care îndeplinesc condiția: a[{i}][{j}] = {matrix_a[i][j]}, s[{i}][{j}] = {matrix_aplusb[i][j]}")
                        return False

    print("\nSuma matricilor a și b este egală cu matricea aplusb.")
    return True


def main():
    check_solutions()
    check_sum_of_matrices()


if __name__ == "__main__":
    main()
