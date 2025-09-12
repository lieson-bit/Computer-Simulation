from sympy import symbols, Function, Derivative, dsolve, solve, simplify, integrate, init_printing
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Symbolic part (your code)
# -------------------------
init_printing()

# Step 1: Define symbols
x = symbols('x')
y = Function('y')(x)
y_prime = Derivative(y, x)

# Step 2: Define the Lagrangian
F = (x**2 * y_prime**2) / (2*x**3 + 1)

# Step 3: Compute partial derivatives
dF_dyprime = F.diff(y_prime)
d_dx_dF_dyprime = Derivative(dF_dyprime, x).doit()

# Step 4: Euler-Lagrange equation
EL_eq = simplify(d_dx_dF_dyprime)
print("Euler-Lagrange equation:")
print(EL_eq)

# Step 5: Solve d/dx(∂F/∂y') = 0 ⇒ ∂F/∂y' = C
C = symbols('C')
eq = dF_dyprime - C
y_prime_sol = solve(eq, y_prime)[0]

# Step 6: Integrate y' to get y(x)
y_sol = integrate(y_prime_sol, x) + symbols('D')

# Step 7: Apply boundary conditions (manual substitution as you gave)
D = symbols('D')
y_sol = y_sol.subs(C, 2)  # From manual solution
y_sol = y_sol.subs(D, 0)  # From y(1) = 0
print("\nSolution y(x):")
print(y_sol)

# Step 8: Compute value of the functional
y_prime_expr = y_sol.diff(x)
F_sub = (x**2 * y_prime_expr**2) / (2*x**3 + 1)
V = integrate(F_sub, (x, 1, 2))
print("\nValue of the functional V[y]:")
print(V.evalf())

# -------------------------
# Numeric + plotting part
# -------------------------

# Step 9: Visualize the solution y(x)
def y_func(x_vals):
    return x_vals**2 - 1/x_vals   # keeps your exact expression

# Generate x values and compute y(x)
x_vals = np.linspace(1, 2, 400)
y_vals = y_func(x_vals)

# Step 1: Define the integrand (your expression)
expr = x**2 * (2*x + 1/x**2)**2 / (2*x**3 + 1)

# Step 2: Simplify the expression
simplified_expr = simplify(expr)
print("\nSimplified integrand:")
print(simplified_expr)

# Step 3: Compute the definite integral from x = 1 to x = 2
definite_integral = integrate(expr, (x, 1, 2))
print("\nValue of the definite integral from 1 to 2:")
print(definite_integral.evalf())

# Evaluate integrand numerically for plotting (convert SymPy values -> floats)
integrand_vals = np.array([float(simplified_expr.evalf(subs={x: float(val)})) for val in x_vals])

# Create a single page with two attractive subplots
plt.style.use('seaborn-v0_8')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: Extremal function
ax1.plot(x_vals, y_vals, color='blue', linewidth=2, label=r'$y(x) = x^2 - \frac{1}{x}$')
ax1.set_title('Extremal Function for the Variational Problem', fontsize=13, fontweight='bold')
ax1.set_xlabel('x', fontsize=11)
ax1.set_ylabel('y(x)', fontsize=11)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend()

# Right: Integrand of the functional
ax2.plot(x_vals, integrand_vals, label='Integrand', color='darkgreen', linewidth=2)
ax2.fill_between(x_vals, integrand_vals, color='green', alpha=0.2)
ax2.set_title('Integrand of the Functional over [1, 2]', fontsize=13, fontweight='bold')
ax2.set_xlabel('x', fontsize=11)
ax2.set_ylabel('f(x)', fontsize=11)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend()

# Annotate the computed integral value on the integrand plot
ax2.text(1.02, 0.90, f"∫ f(x) dx on [1,2] = {definite_integral.evalf():.6f}",
         transform=ax2.transAxes, fontsize=11, bbox=dict(facecolor='white', edgecolor='black', alpha=0.8))

plt.suptitle('Найти значение функционала на полученной экстремали', fontsize=15, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
