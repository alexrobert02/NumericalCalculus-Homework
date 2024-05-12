import random
import cmath

LIMIT = 5000
epsilon = 10 ** (-7)


def horner_method(p, x):
    n = len(p)
    b = p[0]
    for i in range(1, n):
        b_next = b * x + p[i]
        b = b_next
    return b


def muller(p):
    x0 = random.uniform(-1, 1)
    x1 = x0 + random.uniform(-1, 1)
    x2 = x1 + random.uniform(-1, 1)
    for _ in range(LIMIT):
        f0 = horner_method(p, x0)
        f1 = horner_method(p, x1)
        f2 = horner_method(p, x2)
        h1 = x1 - x0
        h2 = x2 - x1
        d1 = (f1 - f0) / h1
        d2 = (f2 - f1) / h2
        a = (d2 - d1) / (h2 + h1)
        b = a * h2 + d2
        D = cmath.sqrt(b**2 - 4*f2*a)
        if abs(b - D) < abs(b + D):
            E = b + D
        else:
            E = b - D
        h = -2 * f2 / E
        x = x2 + h
        if abs(h) < epsilon:
            return x
        x0, x1, x2 = x1, x2, x
    return None


def write_solutions(p, solutions):
    f = open('solutions.txt', 'a')
    f.write('Solutiile reale ale polinomului ')
    n = len(p) - 1
    for i in range(len(p) - 1):
        f.write(f'{p[i]}x^{n}')
        if p[i + 1] > 0:
            f.write('+')
        n -= 1
    f.write(f'{p[len(p) - 1]} sunt:\n')
    i = 0
    for s in solutions:
        if abs(horner_method(p, s)) < epsilon and s.imag == 0:
            f.write(f'x{i + 1} = {s}\n')
            i += 1
    f.write('\n')


def solve_polynomial(p):
    solutions = []
    for i in range(50):
        x = muller(p)
        if x is not None:
            if x + epsilon not in solutions and x - epsilon not in solutions:
                solutions.append(x)
    write_solutions(p, solutions)


def main():
    p = [1, -6, 11, -6]
    solve_polynomial(p)
    p = [42, -55, -42, 49, -6]
    solve_polynomial(p)
    p = [8, -38, 49, -22, 3]
    solve_polynomial(p)
    p = [1, -6, 13, -12, 4]
    solve_polynomial(p)


main()