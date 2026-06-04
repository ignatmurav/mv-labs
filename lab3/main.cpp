#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <functional>
#include <clocale>

using namespace std;

const double EPS = 1e-7;
const double A = 0.1;
const double B = 1.0;

double f(double x) {
    const double eps = 1e-12;
    if (fabs(x - 1.0) < eps) return 1.0;
    if (fabs(x - 1.0) < 1e-8)
        return 1.0 - (x - 1.0) / 2.0;
    return sin(log(x)) / (x - 1.0);
}

double composite_midpoint(int N) {
    double h = (B - A) / N;
    double sum = 0.0;
    for (int i = 0; i < N; ++i) {
        double x = A + (i + 0.5) * h;
        sum += f(x);
    }
    return sum * h;
}

double composite_simpson(int N) {
    if (N % 2 != 0) ++N;
    double h = (B - A) / N;
    double sum = f(A) + f(B);
    for (int i = 1; i < N; ++i) {
        double x = A + i * h;
        if (i % 2 == 0)
            sum += 2.0 * f(x);
        else
            sum += 4.0 * f(x);
    }
    return sum * h / 3.0;
}

struct ResultEntry {
    int N;
    double h;
    double S;
    double R;
    double abs_error;
    double order;
};

vector<ResultEntry> runge_rule(double eps, int p,
                               function<double(int)> composite_method,
                               double exact) {
    vector<ResultEntry> results;
    int N = 4;
    if (p == 4 && N % 2 != 0) ++N;
    double S_prev = composite_method(N);
    double h_prev = (B - A) / N;
    results.push_back({N, h_prev, S_prev, 0.0, fabs(S_prev - exact), 0.0});

    while (true) {
        int N_cur = N * 2;
        double S_cur = composite_method(N_cur);
        double h_cur = (B - A) / N_cur;
        double R = fabs(S_cur - S_prev) / (pow(2.0, p) - 1.0);
        double abs_err = fabs(S_cur - exact);
        double order = 0.0;
        if (results.back().R > 0.0)
            order = log2(results.back().R / R);
        results.push_back({N_cur, h_cur, S_cur, R, abs_err, order});

        if (R < eps) break;
        S_prev = S_cur;
        N = N_cur;
    }
    return results;
}

void print_table(const string& title, const vector<ResultEntry>& results) {
    cout << "\n" << title << "\n";
    cout << string(120, '-') << "\n";
    cout << right 
         << setw(12) << "N (разбиений)"
         << setw(18) << "Шаг h"
         << setw(24) << "Приближ. значение S_h"
         << setw(20) << "Оценка R_h"
         << setw(20) << "|I - S_h|"
         << setw(14) << "Порядок" << "\n";
    cout << fixed << setprecision(10);
    for (const auto& r : results) {
        cout << right
             << setw(12) << r.N << " "
             << setw(18) << r.h << " "
             << setw(24) << r.S << " "
             << setw(20) << r.R << " "
             << setw(20) << r.abs_error << " "
             << setw(14) << r.order << "\n";
    }
    cout << string(120, '-') << "\n";
}

double gauss_legendre_10(double a, double b) {
    static const double x[10] = {
        -0.9739065285171717, -0.8650633666889845, -0.6794095682990244,
        -0.4333953941292472, -0.1488743389816312,  0.1488743389816312,
         0.4333953941292472,  0.6794095682990244,  0.8650633666889845,
         0.9739065285171717
    };
    static const double w[10] = {
        0.0666713443086881, 0.1494513491505806, 0.2190863625159820,
        0.2692667193099963, 0.2955242247147529, 0.2955242247147529,
        0.2692667193099963, 0.2190863625159820, 0.1494513491505806,
        0.0666713443086881
    };
    double mid = (a + b) / 2.0;
    double half = (b - a) / 2.0;
    double sum = 0.0;
    for (int i = 0; i < 10; ++i) {
        double xi = mid + half * x[i];
        sum += w[i] * f(xi);
    }
    return sum * half;
}

int main() {
    setlocale(LC_ALL, "Russian");
    system("chcp 1251 > nul");
    
    cout << setprecision(12);
    
    double I_exact = 1.071074352740254;
    cout << "Точное значение интеграла (вычислено с высокой точностью): " << I_exact << "\n";

    auto midpoint_method = [](int N) { return composite_midpoint(N); };
    vector<ResultEntry> mid_res = runge_rule(EPS, 2, midpoint_method, I_exact);
    print_table("Составная формула средних прямоугольников", mid_res);

    auto simpson_method = [](int N) { return composite_simpson(N); };
    vector<ResultEntry> simpson_res = runge_rule(EPS, 4, simpson_method, I_exact);
    print_table("Составная формула Симпсона", simpson_res);

    double gl_value = gauss_legendre_10(A, B);
    double gl_error = fabs(gl_value - I_exact);
    cout << "\nКвадратура Гаусса-Лежандра (10 узлов):\n";
    cout << "  Приближённое значение = " << setprecision(15) << gl_value << "\n";
    cout << "  Абсолютная погрешность = " << gl_error << "\n";

    cout << "\nВыводы:\n";
    cout << " Обе составные квадратурные формулы достигли требуемой точности =1e-7.\n";
    cout << " Формула Симпсона сходится быстрее (выше порядок точности), чем формула средних прямоугольников.\n";
    cout << " Квадратура Гаусса-Лежандра с 10 узлами даёт очень высокую точность.\n";

    cout << "\nНажмите Enter для выхода...";
    cin.get();
    return 0;
}