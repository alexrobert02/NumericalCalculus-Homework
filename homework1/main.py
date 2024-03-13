import math
import random


def find_machine_precision():
    m = 0
    u = pow(10, -m)
    while 1 + u != 1:
        m += 1
        u = pow(10, -m)
    return u * 10


def ex1():
    precision = find_machine_precision()
    print("Machine precision is:", precision)


def addition_not_associative(x, y, z):
    left_side = (x + y) + z
    right_side = x + (y + z)
    return left_side != right_side, left_side, right_side


def multiplication_not_associative(x, y, z):
    left_side = (x * y) * z
    right_side = x * (y * z)
    return left_side != right_side, left_side, right_side


def ex2():
    precision = find_machine_precision()

    x1 = 1.0
    y1 = precision / 10
    z1 = precision / 10

    # Test addition associativity
    addition_result, addition_left_side, addition_right_side = addition_not_associative(x1, y1, z1)
    if addition_result:
        print("{:.17g} != {:.17g}, so addition is not associative."
              .format(addition_left_side, addition_right_side))
    else:
        print("{:.17g} == {:.17g}, so addition is associative.".format(addition_left_side, addition_right_side))

    x2 = 1/3
    y2 = precision
    z2 = precision

    # Test multiplication associativity
    (multiplication_result,
     multiplication_left_side,
     multiplication_right_side) = multiplication_not_associative(x2, y2, z2)
    if multiplication_result:
        print("{:.17g} != {:.17g}, so multiplication is not associative."
              .format(multiplication_left_side, multiplication_right_side))
    else:
        print("{:.17g} == {:.17g}, so multiplication is associative."
              .format(multiplication_left_side, multiplication_right_side))


def t(i, a):
    if i == 1:
        return a
    elif i == 2:
        return 3 * a / (3 - a ** 2)
    elif i == 3:
        return (15 * a - a ** 3) / (15 - 6 * a ** 2)
    elif i == 4:
        return (105 * a - 10 * a ** 3) / (105 - 45 * a ** 2 + a ** 4)
    elif i == 5:
        return (945 * a - 105 * a ** 3 + a ** 5) / (945 - 420 * a ** 2 + 15 * a ** 4)
    elif i == 6:
        return (10395 * a - 1260 * a ** 3 + 21 * a ** 5) / (10395 - 4725 * a ** 2 + 210 * a ** 4 - a ** 6)
    elif i == 7:
        return (135135 * a - 17325 * a ** 3 + 378 * a ** 5 - a ** 7) / (135135 - 62370 * a ** 2 + 3150 * a ** 4 - 28 * a ** 6)
    elif i == 8:
        return (2027025 * a - 270270 * a**3 + 6930 * a**5 - 36 * a**7) / (2027025 - 945945 * a**2 + 51975 * a**4 - 630 * a**6 + a**8)
    elif i == 9:
        return (34459425 * a - 4729725 * a**3 + 135135 * a**5 - 990 * a**7 + a**9) / (34459425 - 16216200 * a**2 + 945945 * a**4 - 13860 * a**6 + 45 * a**8)


def s(i, a):
    new_a = (2 * a - math.pi) / 4
    return (1 - t(i, new_a) ** 2) / (1 + t(i, new_a) ** 2)


def c(i, a):
    return (1 - t(i, a / 2) ** 2) / (1 + t(i, a / 2) ** 2)


def ex3():
    nr_values = 10000
    random_numbers = [random.uniform(-math.pi / 2, math.pi / 2) for _ in range(nr_values)]

    tan_approx_values = {}
    sin_approx_values = {}
    cos_approx_values = {}

    # Calculate t(i,a) for i = 4 to 9 and a in random_numbers
    for i in range(4, 10):
        tan_approx_values[i] = [t(i, a) for a in random_numbers]
        sin_approx_values[i] = [s(i, a) for a in random_numbers]
        cos_approx_values[i] = [c(i, a) for a in random_numbers]
    # print("Approx values for tangent: ", tan_approx_values)
    # print("Approx values for sine: ", sin_approx_values)
    # print("Approx values for cosine: ", cos_approx_values)

    tan_exact_values = [math.tan(a) for a in random_numbers]

    # Calculate the error for each approximation
    errors = {}
    for i in range(4, 10):
        errors[i] = [abs(tan_exact_values[j] - tan_approx_values[i][j]) for j in range(nr_values)]
    # print("Errors: ", errors)

    # Calculate the average error for each approximation
    average_errors = {}
    for i in range(4, 10):
        average_errors[i] = sum(errors[i]) / nr_values
    print("Average errors: ", average_errors)

    # Hierarchical approximation
    hierarchical_approx = sorted(average_errors, key=lambda x: average_errors[x])
    print(f"Best 3 functions that approximate tan(x) with the smallest average error: T({hierarchical_approx[0]}, a), T({hierarchical_approx[1]}, a), T({hierarchical_approx[2]}, a)")

    print("Hierarchy of the 6 functions:", end=" ")
    for i in range(6):
        print(f"T({hierarchical_approx[i]}, a)", end=" ")


def main():
    print("Exercise 1:")
    ex1()
    print()
    print("Exercise 2:")
    ex2()
    print()
    print("Exercise 3:")
    ex3()


if __name__ == '__main__':
    main()
