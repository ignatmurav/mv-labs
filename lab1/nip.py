import numpy as np
import matplotlib.pyplot as plt
import os

#функции
def f1(x):
    return np.sin(2*x) * np.log(x + 5)

def f2(x):
    return np.sqrt(2*np.abs(x) + x**2)

a, b = -2, 2

#узлы
def equidistant_nodes(n):
    i = np.arange(n+1)
    return a + i*(b-a)/n

def chebyshev_nodes(n):
    i = np.arange(n+1)
    return (a+b)/2 + (b-a)/2 * np.cos((2*i+1)*np.pi/(2*(n+1)))

#вычисление разделенных разностей
def divided_differences(x_nodes, y_nodes):
    
    n = len(x_nodes) - 1

    dd = np.copy(y_nodes).astype(float)
    for j in range(1, n+1):
        for i in range(n, j-1, -1):
            dd[i] = (dd[i] - dd[i-1]) / (x_nodes[i] - x_nodes[i-j])
    return dd

#вычисление значения интерполяционного многочлена Ньютона в точке x по схеме Горнера
def newton_eval(x_nodes, dd, x):

    n = len(dd) - 1

    result = dd[n]
    for k in range(n-1, -1, -1):
        result = result * (x - x_nodes[k]) + dd[k]
    return result

#точки для графиков
def generate_plot_points(nodes_func, f, n):
   
    x_nodes = nodes_func(n)
    y_nodes = f(x_nodes)
    dd = divided_differences(x_nodes, y_nodes)
    x_grid = np.linspace(a, b, 501)
    y_interp = np.array([newton_eval(x_nodes, dd, xi) for xi in x_grid])
    return x_grid, y_interp

def save_points(filename, x_grid, y_interp):
    with open(filename, 'w') as f:
        for x, y in zip(x_grid, y_interp):
            f.write(f"{x:.6f} {y:.6f}\n")

def plot_interpolation(f, f_name, nodes_type, nodes_func):
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"{f_name}, узлы: {nodes_type}")
    
    for idx, n in enumerate([2, 10, 20]):
        ax = axes[idx]
        x_plot = np.linspace(a, b, 1000)
        y_true = f(x_plot)
        ax.plot(x_plot, y_true, 'k-', label='f(x)', linewidth=2)
        

        x_grid, y_interp = generate_plot_points(nodes_func, f, n)
        ax.plot(x_grid, y_interp, 'r--', label=f'P_{n}(x)', linewidth=1.5)
        

        x_nodes = nodes_func(n)
        y_nodes = f(x_nodes)
        ax.plot(x_nodes, y_nodes, 'bo', markersize=4, label='узлы')
        
        ax.set_title(f'n = {n}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend()
        ax.grid(True)
        
        filename = f"{f_name}_{nodes_type}_n{n}.txt"
        save_points(filename, x_grid, y_interp)
        print(f"Сохранено {filename}")
    
    plt.tight_layout()
    plt.savefig(f"{f_name}_{nodes_type}.png")
    plt.show()

def plot_convergence(f, f_name):
   
    n_values = np.arange(1, 62, 5)
    err_eq = []
    err_cheb = []
    
    x_grid = np.linspace(a, b, 501)  
    
    for n in n_values:
        x_nodes_eq = equidistant_nodes(n)
        y_nodes_eq = f(x_nodes_eq)
        dd_eq = divided_differences(x_nodes_eq, y_nodes_eq)
        y_interp_eq = np.array([newton_eval(x_nodes_eq, dd_eq, xi) for xi in x_grid])
        err_eq.append(np.max(np.abs(y_interp_eq - f(x_grid))))
        
        x_nodes_cheb = chebyshev_nodes(n)
        y_nodes_cheb = f(x_nodes_cheb)
        dd_cheb = divided_differences(x_nodes_cheb, y_nodes_cheb)
        y_interp_cheb = np.array([newton_eval(x_nodes_cheb, dd_cheb, xi) for xi in x_grid])
        err_cheb.append(np.max(np.abs(y_interp_cheb - f(x_grid))))
    
    #построение графика
    plt.figure(figsize=(10, 6))
    plt.semilogy(n_values, err_eq, 'b-o', label='равноотстоящие узлы')
    plt.semilogy(n_values, err_cheb, 'r-s', label='чебышевские узлы')
    plt.xlabel('Степень многочлена n')
    plt.ylabel('Равномерная норма погрешности ||r_n||')
    plt.title(f'Сходимость интерполяционного процесса для {f_name}')
    plt.legend()
    plt.grid(True, which='both')
    plt.savefig(f"convergence_{f_name}.png")
    plt.show()

if __name__ == "__main__":
    os.makedirs("points", exist_ok=True)
    os.chdir("points")
    
    functions = [(f1, "f1"), (f2, "f2")]
    node_types = [("равноотстоящие", equidistant_nodes), ("чебышевские", chebyshev_nodes)]
    
    for f, f_name in functions:
        for node_name, node_func in node_types:
            plot_interpolation(f, f_name, node_name, node_func)
    
    for f, f_name in functions:
        plot_convergence(f, f_name)
    
    print("Все графики построены и сохранены.")