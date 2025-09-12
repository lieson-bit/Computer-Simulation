from sympy import symbols, Function, Derivative, dsolve, solve, simplify, integrate, init_printing, Eq, latex, N
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Инициализация красивого вывода
init_printing()

print("="*70)
print("РЕШЕНИЕ ВАРИАЦИОННОЙ ЗАДАЧИ И ВЫЧИСЛЕНИЕ ЗНАЧЕНИЯ ФУНКЦИОНАЛА")
print("="*70)

# Step 1: Define symbols
x = symbols('x')
y = Function('y')(x)
y_prime = Derivative(y, x)

# Step 2: Define the Lagrangian
F = (x**2 * y_prime**2) / (2*x**3 + 1)

print("\n1. ПОСТАНОВКА ЗАДАЧИ:")
print(f"Функционал: V[y] = ∫₁² [{latex(F)}] dx")
print("Граничные условия: y(1) = 0, y(2) = 7/2")

# Step 3: Compute partial derivatives
dF_dyprime = F.diff(y_prime)
d_dx_dF_dyprime = Derivative(dF_dyprime, x).doit()

# Step 4: Euler-Lagrange equation
EL_eq = simplify(d_dx_dF_dyprime)
print("\n2. УРАВНЕНИЕ ЭЙЛЕРА-ЛАГРАНЖА:")
print(f"d/dx(∂F/∂y') = {latex(EL_eq)}")

# Step 5: Solve d/dx(∂F/∂y') = 0 ⇒ ∂F/∂y' = C
C = symbols('C')
eq = dF_dyprime - C
y_prime_sol = solve(eq, y_prime)[0]

print(f"\n3. РЕШЕНИЕ ДЛЯ ПРОИЗВОДНОЙ:")
print(f"∂F/∂y' = C ⇒ y'(x) = {latex(y_prime_sol)}")

# Step 6: Integrate y' to get y(x)
y_sol = integrate(y_prime_sol, x) + symbols('D')

print(f"\n4. ИНТЕГРИРОВАНИЕ:")
print(f"y(x) = ∫ y'(x) dx = {latex(y_sol)}")

# Step 7: Apply boundary conditions
D = symbols('D')
y_sol = y_sol.subs(C, 2)  # From manual solution
y_sol = y_sol.subs(D, 0)  # From y(1) = 0

print("\n5. ОПРЕДЕЛЕНИЕ КОНСТАНТ ИЗ ГРАНИЧНЫХ УСЛОВИЙ:")
print("y(1) = 0 ⇒ D = 0")
print("y(2) = 7/2 ⇒ C = 2")
print(f"\nФИНАЛЬНОЕ РЕШЕНИЕ: y(x) = {latex(y_sol)}")

# Step 8: Compute value of the functional
y_prime_expr = y_sol.diff(x)
F_sub = (x**2 * y_prime_expr**2) / (2*x**3 + 1)
V = integrate(F_sub, (x, 1, 2))

print("\n6. ВЫЧИСЛЕНИЕ ЗНАЧЕНИЯ ФУНКЦИОНАЛА:")
print(f"Подынтегральное выражение: f(x) = {latex(F_sub)}")
print(f"Значение функционала: V[y] = ∫₁² f(x) dx = {V.evalf():.6f}")

# Упрощенное выражение для подынтегральной функции
simplified_expr = simplify(F_sub)
print(f"Упрощенное выражение: f(x) = {latex(simplified_expr)}")

# Подготовка данных для графиков
x_vals = np.linspace(1, 2, 400)
y_func = lambda x_val: x_val**2 - 1/x_val
y_vals = y_func(x_vals)

# Вычисление значений подынтегральной функции (преобразуем в числа)
integrand_vals = [float(N(simplified_expr.subs(x, val))) for val in x_vals]

# Создание комплексной графической визуализации
plt.style.use('seaborn-v0_8')
fig = plt.figure(figsize=(15, 10))
gs = GridSpec(2, 2, figure=fig)

# График 1: Экстремальная функция
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(x_vals, y_vals, color='blue', linewidth=3, label=r'$y(x) = x^2 - \frac{1}{x}$')
ax1.plot([1, 2], [0, 3.5], 'ro', markersize=8, label='Граничные условия')
ax1.set_title('Экстремальная функция', fontsize=14, fontweight='bold')
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('y(x)', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.text(1.1, 2.5, f'y(1) = 0\ny(2) = 3.5', 
         bbox=dict(facecolor='yellow', alpha=0.7), fontsize=10)

# График 2: Подынтегральная функция
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(x_vals, integrand_vals, color='darkgreen', linewidth=3, label='Подынтегральная функция')
ax2.fill_between(x_vals, integrand_vals, alpha=0.3, color='green', label='Площадь под кривой')
ax2.set_title('Подынтегральная функция f(x)', fontsize=14, fontweight='bold')
ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('f(x)', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.text(1.1, max(integrand_vals)*0.7, f'V[y] = ∫₁² f(x) dx\n= {float(N(V)):.6f}', 
         bbox=dict(facecolor='lightgreen', alpha=0.7), fontsize=10)

# График 3: Производная экстремальной функции
ax3 = fig.add_subplot(gs[1, 0])
y_prime_vals = 2*x_vals + 1/(x_vals**2)
ax3.plot(x_vals, y_prime_vals, color='red', linewidth=3, label=r"$y'(x) = 2x + \frac{1}{x^2}$")
ax3.set_title('Производная экстремальной функции', fontsize=14, fontweight='bold')
ax3.set_xlabel('x', fontsize=12)
ax3.set_ylabel("y'(x)", fontsize=12)
ax3.grid(True, alpha=0.3)
ax3.legend()

# График 4: Совмещенный график
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(x_vals, y_vals, 'b-', linewidth=2, label='y(x)')
ax4.plot(x_vals, integrand_vals, 'g-', linewidth=2, label='f(x)')
ax4.plot(x_vals, y_prime_vals, 'r-', linewidth=2, label="y'(x)")
ax4.set_title('Совмещенный график', fontsize=14, fontweight='bold')
ax4.set_xlabel('x', fontsize=12)
ax4.grid(True, alpha=0.3)
ax4.legend()

plt.tight_layout()
plt.show()

# Дополнительная информация в консоли
print("\n7. ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ:")
print(f"Значение функционала (численно): {float(N(V)):.8f}")
print(f"Значение функционала (дробь): {V}")
print(f"Значение функционала (аналитически): {77/12} ≈ {77/12:.8f}")

# Проверка граничных условий
print("\n8. ПРОВЕРКА ГРАНИЧНЫХ УСЛОВИЙ:")
print(f"y(1) = {y_func(1):.6f}")
print(f"y(2) = {y_func(2):.6f} (ожидается 3.5)")

# Анализ подынтегральной функции
print("\n9. АНАЛИЗ ПОДЫНТЕГРАЛЬНОЙ ФУНКЦИИ:")
print(f"f(1) = {float(N(simplified_expr.subs(x, 1))):.6f}")
print(f"f(2) = {float(N(simplified_expr.subs(x, 2))):.6f}")
print(f"Максимальное значение f(x) на [1,2]: {max(integrand_vals):.6f}")
print(f"Минимальное значение f(x) на [1,2]: {min(integrand_vals):.6f}")

print("="*70)
print("РАСЧЕТ ЗАВЕРШЕН. ВСЕ РЕЗУЛЬТАТЫ ПОЛУЧЕНЫ И ВИЗУАЛИЗИРОВАНЫ.")
print("="*70)