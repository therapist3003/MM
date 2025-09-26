import numpy as np
import matplotlib.pyplot as plt

# Example data
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 5, 10, 17, 26])

def fit_polynomial(x, y, degree):
    # Vandermonde matrix
    X = np.vander(x, N=degree+1, increasing=True)
    # Normal equation
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    return beta

def predict(x, beta):
    X = np.vander(x, N=len(beta), increasing=True)
    return X @ beta

# Fit polynomials
beta_linear = fit_polynomial(x, y, 1)
beta_quad = fit_polynomial(x, y, 2)
beta_cubic = fit_polynomial(x, y, 3)

# Create smooth x values for plotting
x_plot = np.linspace(min(x)-1, max(x)+1, 200)
y_linear = predict(x_plot, beta_linear)
y_quad = predict(x_plot, beta_quad)
y_cubic = predict(x_plot, beta_cubic)

# Plot
plt.scatter(x, y, color='black', label='Data')
plt.plot(x_plot, y_linear, label='Linear', linestyle='--')
plt.plot(x_plot, y_quad, label='Quadratic', linestyle='-.')
plt.plot(x_plot, y_cubic, label='Cubic', linestyle=':')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Regression (Matrix Method)')
plt.legend()
plt.grid(True)
plt.show()
