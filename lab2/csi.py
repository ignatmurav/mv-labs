import numpy as np
import matplotlib.pyplot as plt

def f1(x):
    return np.sin(2*x) * np.log(x + 5)

def f1_prime(x):
    # Производная f1(x)
    return 2*np.cos(2*x)*np.log(x+5) + np.sin(2*x)/(x+5)

def f2(x):
    return np.sqrt(2*np.abs(x) + x**2)

def f2_prime(x):
    # Производная f2(x)
    if x < 0:
        return (x - 1) / np.sqrt(x**2 - 2*x)
    elif x > 0:
        return (x + 1) / np.sqrt(x**2 + 2*x)
    else:
        return np.inf


def cubic_spline_coeffs(x, y, yp_left, yp_right):
    n = len(x) - 1          # количество интервалов
    h = x[1] - x[0]

    d = np.zeros(n + 1)
    d[0] = 6/h**2 * (y[1] - y[0]) - 6/h * yp_left
    for i in range(1, n):
        d[i] = 6/h**2 * (y[i+1] - 2*y[i] + y[i-1])
    d[n] = 6/h * yp_right - 6/h**2 * (y[n] - y[n-1])

    a = np.zeros(n + 1)
    b = np.zeros(n + 1)
    c = np.zeros(n + 1)

    b[0] = 2.0
    c[0] = 1.0
    for i in range(1, n):
        a[i] = 1.0
        b[i] = 4.0
        c[i] = 1.0
    a[n] = 1.0
    b[n] = 2.0

    #метод прогонки, прямой ход
    for i in range(1, n + 1):
        w = a[i] / b[i-1]
        b[i] -= w * c[i-1]
        d[i] -= w * d[i-1]

    #обратный ход
    m = np.zeros(n + 1)
    m[n] = d[n] / b[n]
    for i in range(n - 1, -1, -1):
        m[i] = (d[i] - c[i] * m[i+1]) / b[i]

    coeffs = []
    for i in range(n):
        a_i = y[i]
        c_i = m[i] / 2.0
        d_i = (m[i+1] - m[i]) / (6.0 * h)
        b_i = (y[i+1] - y[i]) / h - h * (2*m[i] + m[i+1]) / 6.0
        coeffs.append((a_i, b_i, c_i, d_i))

    return coeffs, x

def eval_spline(coeffs, x_nodes, x_eval):

    n = len(x_nodes) - 1
    if np.isscalar(x_eval):
        if x_eval <= x_nodes[0]:
            i = 0
        elif x_eval >= x_nodes[-1]:
            i = n - 1
        else:
            i = np.searchsorted(x_nodes, x_eval) - 1
        a, b, c, d = coeffs[i]
        dx = x_eval - x_nodes[i]
        return a + b*dx + c*dx**2 + d*dx**3
    else:
        result = np.empty_like(x_eval)
        for idx, xv in enumerate(x_eval):
            if xv <= x_nodes[0]:
                i = 0
            elif xv >= x_nodes[-1]:
                i = n - 1
            else:
                i = np.searchsorted(x_nodes, xv) - 1
            a, b, c, d = coeffs[i]
            dx = xv - x_nodes[i]
            result[idx] = a + b*dx + c*dx**2 + d*dx**3
        return result

a, b = -2.0, 2.0
funcs = [(f1, f1_prime, "f1(x) = sin(2x)·ln(x+5)"),
         (f2, f2_prime, "f2(x) = sqrt(2|x| + x^2)")]

for f, f_prime, title in funcs:
    print(f"\n=== {title} ===")

    N = 15
    x_nodes = np.linspace(a, b, N+1)
    y_nodes = f(x_nodes)
    yp_left = f_prime(a)
    yp_right = f_prime(b)

    coeffs, nodes = cubic_spline_coeffs(x_nodes, y_nodes, yp_left, yp_right)

    x_fine = np.linspace(a, b, 501)   
    y_fine = f(x_fine)
    y_spline = eval_spline(coeffs, nodes, x_fine)
    err = np.max(np.abs(y_fine - y_spline))
    print(f"N = {N}, максимальная погрешность = {err:.2e}")

    plt.figure()
    plt.plot(x_fine, y_fine, label='f(x)')
    plt.plot(x_fine, y_spline, '--', label='Сплайн')
    plt.title(f'{title}, N = {N}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{f.__name__}_N15.png')
    plt.show()
    
    N_list = list(range(5, 101, 5))
    errors = []
    for N in N_list:
        x_nodes = np.linspace(a, b, N+1)
        y_nodes = f(x_nodes)
        yp_left = f_prime(a)
        yp_right = f_prime(b)
        coeffs, nodes = cubic_spline_coeffs(x_nodes, y_nodes, yp_left, yp_right)
        x_fine = np.linspace(a, b, 501)
        y_fine = f(x_fine)
        y_spline = eval_spline(coeffs, nodes, x_fine)
        err = np.max(np.abs(y_fine - y_spline))
        errors.append(err)

    plt.figure()
    plt.loglog(N_list, errors, 'o-')
    plt.xlabel('Число интервалов N')
    plt.ylabel('Максимальная погрешность')
    plt.title(f'Сходимость: {title}')
    plt.grid(True, which='both')
    plt.savefig(f'{f.__name__}_convergence.png')
    plt.show()

    # Вывод таблицы значений
    print(f"N\tПогрешность")
    for N, err in zip(N_list, errors):
        print(f"{N}\t{err:.2e}")

print("\nВсе эксперименты завершены.")