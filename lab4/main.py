import numpy as np
import matplotlib.pyplot as plt

def f(x, u):
    return (u + x) ** 2

def df_du(x, u):
    return 2.0 * (u + x)

def u_exact(x):
    return np.tan(x) - x

a, b = 0.0, 1.0
u0 = 0.0
p = 3
h = 0.1
h2 = h / 2.0

def implicit_trapezoidal(f, df_du, a, b, u0, h, tol=1e-12, max_iter=50):
    n_steps = int(round((b - a) / h))
    x = np.linspace(a, b, n_steps + 1)
    u = np.zeros(n_steps + 1)
    u[0] = u0
    for i in range(n_steps):
        xn = x[i]
        un = u[i]
        x_next = xn + h
        u_guess = un + h * f(xn, un)
        u_new = u_guess
        for _ in range(max_iter):
            F = u_new - un - 0.5 * h * (f(xn, un) + f(x_next, u_new))
            dF = 1.0 - 0.5 * h * df_du(x_next, u_new)
            delta = F / dF
            u_new -= delta
            if abs(delta) < tol:
                break
        u[i+1] = u_new
    return x, u

def runge_kutta(f, a, b, u0, h, p):
    n_steps = int(round((b - a) / h))
    x = np.linspace(a, b, n_steps + 1)
    u = np.zeros(n_steps + 1)
    u[0] = u0
    for i in range(n_steps):
        xn = x[i]
        un = u[i]
        if p == 2:
            k1 = f(xn, un)
            k2 = f(xn + h, un + h * k1)
            u[i+1] = un + h * (k1 + k2) / 2.0
        elif p == 3:
            k1 = f(xn, un)                                 # c1=0, a11=0
            k2 = f(xn + h/2.0, un + h/2.0 * k1)           # c2=1/2, a21=1/2
            k3 = f(xn + h, un - h * k1 + 2.0 * h * k2)    # c3=1, a31=-1, a32=2
            u[i+1] = un + h * (k1 + 4.0*k2 + k3) / 6.0    # bi = 1/6, 4/6, 1/6
        else:
            raise ValueError("Только p=2 или p=3")
    return x, u

x_h_imp, y_h_imp = implicit_trapezoidal(f, df_du, a, b, u0, h)
x_h2_imp, y_h2_imp = implicit_trapezoidal(f, df_du, a, b, u0, h2)

x_h_exp, y_h_exp = runge_kutta(f, a, b, u0, h, p)
x_h2_exp, y_h2_exp = runge_kutta(f, a, b, u0, h2, p)

u_exact_h2 = u_exact(x_h2_exp)
max_error = np.max(np.abs(u_exact_h2 - y_h2_exp))
print("РЕЗУЛЬТАТЫ ДЛЯ ВАРИАНТА 8 (p = 3)")
print(f"Максимальная абсолютная ошибка |точное - y_h/2| (явный РК): {max_error:.3e}")

N = len(x_h_exp) - 1
diff_at_coarse = np.abs(y_h_exp - y_h2_exp[::2])
max_diff = np.max(diff_at_coarse)
richardson_est = max_diff / (2.0**p - 1.0)
print(f"max_i |y_h_i - y_{{h/2}}_{{2i}}| = {max_diff:.3e}")
print(f"Оценка ошибки: {richardson_est:.3e}")

x_dense = np.linspace(a, b, 200)
u_exact_dense = u_exact(x_dense)

plt.figure(figsize=(8, 5))
plt.plot(x_dense, u_exact_dense, 'k-', linewidth=2, label='Точное решение')
plt.plot(x_h2_exp, y_h2_exp, 'ro', markersize=4, label=f'Явный РК (p={p}), h={h2}')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Решение задачи Коши (вариант 8)')
plt.legend()
plt.grid(True)
plt.savefig('solution_plot_variant8.png', dpi=150)
plt.show()

print("\nВыборочные значения (точное vs явный РК с h/2):")
print("   x        точное       y_h/2        ошибка")
for i in range(0, len(x_h2_exp), max(1, len(x_h2_exp)//10)):
    xi = x_h2_exp[i]
    exact_i = u_exact(xi)
    num_i = y_h2_exp[i]
    print(f"{xi:.3f}   {exact_i:.6f}   {num_i:.6f}   {abs(exact_i-num_i):.2e}")

u_exact_h2_imp = u_exact(x_h2_imp)
max_error_imp = np.max(np.abs(u_exact_h2_imp - y_h2_imp))
print("\n" + "=" * 60)
print("Неявный метод трапеций (h/2) - максимальная ошибка:")
print(f"max |точное - y_h/2| = {max_error_imp:.3e}")