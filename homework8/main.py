import numpy as np

epsilon = 1e-6


# Definim functia F: RXR -> R
def F(x, y):
    return x**2 + y**2 - 2*x - 4*y - 1


# Calculul gradientului folosind formula analitica
def gradient_analytic(x, y):
    grad_x = 2 * x - 2
    grad_y = 2 * y - 4
    return np.array([grad_x, grad_y])


def gradient_approx(x, y, h=1e-5):
    G1 = (3*F(x,y) - 4*F(x-h,y) + F(x-2*h,y)) / (2*h)
    G2 = (3*F(x,y) - 4*F(x,y-h) + F(x,y-2*h)) / (2*h)
    return np.array([G1, G2])


# Metoda gradientului descendent cu rata de invatare constanta
def gradient_descent_constant_learning_rate(starting_point, learning_rate, max_iter=5000):
    current_point = starting_point
    iterations = 0
    while iterations < max_iter:
        gradient = gradient_analytic(current_point[0], current_point[1])
        new_point = current_point - learning_rate * gradient
        if np.linalg.norm(new_point - current_point) < epsilon:
            break
        current_point = new_point
        iterations += 1
    return current_point, iterations


# Metoda gradientului descendent cu backtracking line search
def gradient_descent_backtracking(starting_point, beta=0.8, max_iter=5000):
    current_point = starting_point
    iterations = 0
    while iterations < max_iter:
        gradient = gradient_analytic(current_point[0], current_point[1])
        learning_rate = 1
        p = 1
        while (F(current_point[0]-gradient[0], current_point[1]-gradient[1])) > (F(current_point[0], current_point[1]) - ((learning_rate / 2) * np.linalg.norm(gradient))) and p < 8:
            learning_rate = learning_rate * beta
            p += 1
        new_point = current_point - learning_rate * gradient
        if np.linalg.norm(new_point - current_point) < epsilon:
            break
        current_point = new_point
        iterations += 1
    return current_point, iterations


def main():
    # Punctul de pornire pentru metoda gradientului descendent
    starting_point = np.array([3.0, 5.0])

    # Aplicam metoda gradientului descendent cu rata de invatare constanta
    min_point_constant_lr, iterations_constant_lr = gradient_descent_constant_learning_rate(starting_point, learning_rate=0.1)
    print("Minim folosind rata de învățare constantă:", min_point_constant_lr.round())
    print("Numărul de iterații cu rata de învățare constantă:", iterations_constant_lr)

    # Aplicam metoda gradientului descendent cu backtracking line search
    min_point_backtracking, iterations_backtracking = gradient_descent_backtracking(starting_point)
    print("Minim folosind backtracking line search:", min_point_backtracking.round())
    print("Numărul de iterații cu backtracking line search:", iterations_backtracking)


main()