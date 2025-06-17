import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],  # This will map to the best available serif "Times"
    "font.size": 15,             # Base font size
    "axes.titlesize": 17,        # Title font size
    "axes.labelsize": 15,        # Axis label size
    "xtick.labelsize": 13,       # X tick label size
    "ytick.labelsize": 13,       # Y tick label size
    "legend.fontsize": 13,       # Legend font size
})


# Upper bounds where 95 % of the mass should lie
upper_bounds = [1.2, 1.5, 2.0, 3.0, 4.0]
p = 0.95
lambdas = [-np.log(1 - p) / (u - 1.0) for u in upper_bounds]

x = np.linspace(1, 5, 800)

# Create 3D plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot filled areas and add outline lines along each distribution
for i, (upper, lam) in enumerate(zip(upper_bounds, lambdas)):
    pdf = lam * np.exp(-lam * (x - 1))
    norm_const = 1 - np.exp(-lam * (upper - 1))
    #pdf_norm = np.where(x <= upper, pdf / norm_const, 0.0)
    pdf_norm = pdf / norm_const
    y = np.full_like(x, i)
    
    # Filled surface
    ax.plot_surface(np.vstack([x, x]), np.vstack([y, y]), np.vstack([np.zeros_like(pdf_norm), pdf_norm]),
                    color=f"C{i}", alpha=0.9, rstride=1, cstride=1)

    # Line along the top of each filled distribution
    ax.plot(x, y, pdf_norm, color=f"C{i}",alpha=0.9, linewidth=0.8, label=f"95% in [1, {upper}]")

    ax.text(x=1.25, y=4, z=i*2 + 5, s=f"95% in [1, {upper}]", color=f"C{i}")

# Customize view and appearance
ax.set_xlabel(r"$\omega$")
ax.set_ylabel("")  # Hide label
ax.set_zlabel("Probability Density")
ax.set_zticks([0, 3,6,9,12])  # Remove y-axis ticks
ax.set_xticks([1,2,3,4,5])
ax.set_yticks([])

# Proper orientation
ax.set_xlim(1, 5)
ax.view_init(elev=5, azim=-75)
ax.grid(False)
plt.savefig("./tables/plot_95mass.png")   # save if desired
plt.show()