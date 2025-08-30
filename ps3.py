import matplotlib.pyplot as plt
import math
from tabulate import tabulate
import numpy as np

# 1.
def p_2N(x, N):
    product = np.ones_like(x, dtype=float)
    for j in range(1, N + 1):
        product *= (1 - (N * x / j)**2)
    return product

N_values = [2, 4, 6, 8, 12, 20]

x_domain = np.linspace(-1, 1, 1000)

plt.figure(figsize=(12, 8))
for N in N_values:
    y_values = p_2N(x_domain, N)
    plt.plot(x_domain, y_values, label=f'N = {N} (degree {2*N})')

plt.title(r'Behavior of Polynomial $p_{2N}(x)$ as N increases', fontsize=16)
plt.xlabel('x')
plt.ylabel(r'$p_{2N}(x)$')
plt.axhline(0, color='black', linewidth=0.5) # x-axis
plt.axvline(0, color='black', linewidth=0.5) # y-axis
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.ylim(-0.4, 1.2) # Set y-axis limits for better comparison
plt.show()

# 2.
def get_func_val(x):
  return (1/(np.sqrt(2 * math.pi))) * np.exp(-x**2 / 2)

def trapezoidal(a, b, n):
    x = np.linspace(a, b, n)
    y = get_func_val(x)
    h = (b - a) / (n - 1)
    return (h/2) * (y[0] + 2*np.sum(y[1:-1]) + y[-1])

def simpsons(a, b, n):
    x = np.linspace(a, b, n)
    y = get_func_val(x)
    h = (b - a) / (n - 1)

    return (h/3) * (y[0] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-2:2]) + y[-1])

a, b = -100, 2
true_value = 0.977249868051821

for n in [2001, 4001]:
    trapezoidal_res = trapezoidal(a, b, n)
    simpson_res = simpsons(a, b, n)

    print(f"\nUsing {n} points:")
    print(f"Trapezoidal Result: {trapezoidal_res:.15f}, Absolute Error: {abs(true_value - trapezoidal_res):.2e}")
    print(f"Simpson Result:     {simpson_res:.15f}, Absolute Error: {abs(true_value - simpson_res):.2e}")

# 3.
def get_func_val(x):
    return np.sin(x)

a, b = 0, np.pi
n = 20

if n%2==0:
  n += 1
true_val = 2

trapezoid_res = trapezoidal(a, b, n)
simpson_res = simpsons(a, b, n)

print(f"Trapezoidal Result: {trapezoid_res:.15f}, Absolute Error: {abs(true_val - trapezoid_res):.2e}")
print(f"Simpson Result:     {simpson_res:.15f}, Absolute Error: {abs(true_val - simpson_res):.2e}")

# 4.
def get_func_val(x):
    return np.exp(-x**2)

a, b = 0, 1

n_vals = [100, 200, 400]
results = []

for n in n_vals:
    trapezoid_res = trapezoidal(a, b, n)
    results.append(trapezoid_res)
    print(f"Trapezoidal Result with {n} subintervals: {trapezoid_res:.15f}")

p = np.log2(abs((results[0] - results[1]) / (results[1] - results[2])))

print(f"\nEstimated Order of Accuracy (p): {p:.6f}")