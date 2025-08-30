import math
import matplotlib.pyplot as plt

import numpy as np
from scipy import stats

# 1.
def f(t, y):
  return 2 - math.exp(-4*t) - (2*y)

t0 = 0

y0 = 1

# h = float(input("Enter step size h: "))
h = 0.1

#n = int(input("Enter number of steps n: "))
n = 100

t_vals = [t0]
y_vals = [y0]

# Euler's calculation
for j in range(n):
  m = f(t0, y0)
  y1 = y0 + h * m
  t1 = t0 + h

  print(f"Step {j+1}: t = {t1:.4f}, y = {y1:.4f}")

  t_vals.append(t1)
  y_vals.append(y1)

  t0 = t1
  y0 = y1

plt.figure(figsize=(12, 8))
plt.plot(t_vals, y_vals, label='Euler''s approximation', linewidth=2)

plt.title('Euler''s Method')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.grid(True)
plt.legend()
plt.show()

#2.
def rk4_second_order(f, x0, y0, y_dash_0, x_end, h):
    """
        f     - function representing y'' = f(x, y, y')
        x0    - initial x
        y0    - initial y(x0)
        y_dash_0    - initial y'(x0)
        x_end - final x value
        h     - step size
    """

    x = x0
    y = y0
    y_dash = y_dash_0

    x_vals = [x]
    y_vals = [y]

    while x < x_end:
        k1_y = h * y_dash # k1 = h * f(x0, y0, z0) ==> f(x0, y0, z0) = y'
        k1_y_dash = h * f(x, y, y_dash) # Here, f(x0, y0, z0) = y''

        k2_y = h * (y_dash + 0.5 * k1_y_dash)
        k2_y_dash = h * f(x + 0.5*h, y + 0.5*k1_y, y_dash + 0.5*k1_y_dash)

        k3_y = h * (y_dash + 0.5*k2_y_dash)
        k3_y_dash = h * f(x + 0.5*h, y + 0.5*k2_y, y_dash + 0.5*k2_y_dash)

        k4_y = h * (y_dash + k3_y_dash)
        k4_y_dash = h * f(x + h, y + k3_y, y_dash + k3_y_dash)

        # Update y and y'
        y = y + (k1_y + 2*k2_y + 2*k3_y + k4_y) / 6.0
        y_dash = y_dash + (k1_y_dash + 2*k2_y_dash + 2*k3_y_dash + k4_y_dash) / 6.0
        x = x + h

        x_vals.append(x)
        y_vals.append(y)

    return x_vals, y_vals

def func(x, y, y_dash):
  return math.sin(x) - y*y_dash - 3*y

x_res, y_res = rk4_second_order(func, x0=0, y0=-1, y_dash_0=1, x_end=10, h=0.1)

for i in range(10):
    print(f"x={x_res[i]:.2f}, y={y_res[i]:.4f}")

coeffs = np.polyfit(x_res, y_res, deg=len(x_res)-1)
poly = np.poly1d(coeffs)
y_poly = poly(x_res)

t_stat, p_value = stats.ttest_rel(y_res, y_poly)

x_curve = np.linspace(0, 10, 300)
y_curve_rk4 = np.interp(x_curve, x_res, y_res)
y_curve_poly = poly(x_curve)

plt.plot(x_curve, y_curve_rk4, label='RK4 Solution', linestyle='solid')
plt.plot(x_curve, y_curve_poly, label='Interpolating Polynomial', linestyle='dashed')
plt.title("RK4 (2nd Order ODE) vs Polynomial Interpolation")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

print(f"T-statistic = {t_stat:.4f}, P-value = {p_value:.6f}")

# 3.
def f(y, t):
    return 2 - math.exp(-4 * t) - 2*y

N = 1000            # number of time steps
dt = (10 - 0) / N    # step size
n = 100             # number of random samples for expectation

t = np.arange(0, 10, dt)
y = np.zeros(N)

# Initial condition
y[0] = 1

def expectation(f, n, yi, tf, ts):
    expv = np.zeros(n)
    trand = np.random.uniform(tf, ts, size=n)
    for k in range(1, n):
        expv[k] = expv[k-1] + (1/n) * f(yi, trand[k-1])
    return expv[n-1]

# Monte-Carlo
for i in range(1, N):
    yi = y[i-1]
    tf = t[i-1]
    ts = t[i]
    y[i] = y[i-1] + expectation(f, n, yi, tf, ts) * dt

plt.figure(figsize=(8,5))
plt.title('Monte Carlo Simulation Solution for ODE')
plt.plot(t, y, label='Monte Carlo Solution', color='blue')
# plt.plot(t, 5*np.exp(-2*t), '--', label='Analytical Solution (5e^{-2t})', color='red')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.grid(True)
plt.show()

# Milne
def rk4_step(x, y, h):
    k1 = h * f(x, y)
    k2 = h * f(x + h/2, y + k1/2)
    k3 = h * f(x + h/2, y + k2/2)
    k4 = h * f(x + h, y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6

def milne_method(x0, y0, h, N):
    x = np.zeros(N)
    y = np.zeros(N)

    # Initial values
    x[0] = x0
    y[0] = y0

    # Generate first 3 points using RK4
    for i in range(1, 4):
        x[i] = x[i-1] + h
        y[i] = rk4_step(x[i-1], y[i-1], h)

    # Milne's Predictor-Corrector
    for i in range(3, N-1):
        x[i+1] = x[i] + h

        # Predictor
        y_pred = y[i-3] + (4*h/3)*(2*f(x[i-2], y[i-2]) - f(x[i-1], y[i-1]) + 2*f(x[i], y[i]))

        # Corrector
        y_corr = y[i-1] + (h/3)*(f(x[i-1], y[i-1]) + 4*f(x[i], y[i]) + f(x[i+1], y_pred))

        y[i+1] = y_corr

    return x, y

t0 = 0
y0 = 1
h = 0.1
N = 20

t_vals, y_vals = milne_method(t0, y0, h, N)

# Plot solution
plt.figure(figsize=(8,5))
plt.plot(t_vals, y_vals, 'o-', label='Milne Predictor-Corrector')
plt.title("Solution using Milne's Method")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.legend()
plt.grid(True)
plt.show()

# Print results
for ti, yi in zip(t_vals, y_vals):
    print(f"t = {ti:.2f}, y = {yi:.6f}")

