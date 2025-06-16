import numpy as np
import matplotlib.pyplot as plt

# Desired upper bounds for 50 % mass, ordered from smallest to largest
upper_bounds = [1.25, 1.5, 1.75, 2.0]
lambdas = [np.log(2) / (u - 1.0) for u in upper_bounds]

x = np.linspace(1, 4, 500)

# Plot in reverse order for proper layering
plt.figure()
for idx, (upper, lam) in enumerate(zip(reversed(upper_bounds), reversed(lambdas))):
    pdf = lam * np.exp(-lam * (x - 1))
    label = f"50 % in [1, {upper}] (λ≈{lam:.2f})"
    alpha = 1.0 if upper == 2.0 else 0.3  # Full opacity only for [1, 2]
    plt.plot(x, pdf, label=label, zorder=idx + 1)
    plt.fill_between(x, 0, pdf, where=(x >= 1) & (x <= upper), alpha=alpha, zorder=idx + 1)

plt.title("Shifted Exponential PDFs with 50 % Mass in [1, upper]")
plt.xlabel("x")
plt.ylabel("Probability Density")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
