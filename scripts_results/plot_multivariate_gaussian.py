import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Set seed
np.random.seed(45)



# ----------- TARGET MEANS (Used for filtering source later) -------------
# 5 target Gaussians: 3 for class 0, 2 for class 1
target_means = [np.array([-2.5, -2]), np.array([-2, -0.5]),  # Class 0
                np.array([0, 1]), np.array([2.5, -0.5]), np.array([2.1, 1])]                        # Class 1

# ----------- FILTER SOURCE MEANS TO AVOID OVERLAP WITH TARGETS ----------
n_source = 100
min_dist = 2  # Distance threshold to avoid overlap with target means

def is_far_from_targets(mean, targets, min_dist):
    return all(np.linalg.norm(mean - t) > min_dist for t in targets)

filtered_source_means = []
while len(filtered_source_means) < n_source:
    candidate = np.random.uniform(-10, 10, 2)
    if is_far_from_targets(candidate, target_means, min_dist):
        filtered_source_means.append(candidate)
source_means = np.array(filtered_source_means)

# Grid for density
x, y = np.mgrid[-10:10:.1, -10:10:.1]
pos = np.dstack((x, y))

# ----------- SOURCE DOMAIN (Background Blue Density) -------------
source_covs = [np.eye(2) * np.random.uniform(1.0, 3.0) for _ in range(n_source)]
source_pdf = sum(multivariate_normal(mean=m, cov=c).pdf(pos)
                 for m, c in zip(source_means, source_covs))
source_pdf /= np.max(source_pdf)

# ----------- TARGET DOMAIN (Foreground Red Density) -------------
target_covs = [np.eye(2) for _ in target_means]

# Class assignment
n_samples_per_mode = 2
class0 = np.vstack([
    np.random.multivariate_normal(target_means[0], target_covs[0]/16, n_samples_per_mode),
    np.random.multivariate_normal(target_means[1], target_covs[1]/16, n_samples_per_mode),
])
class1 = np.vstack([
        np.random.multivariate_normal(target_means[2], target_covs[2]/16, n_samples_per_mode),
    np.random.multivariate_normal(target_means[3], target_covs[3]/16, n_samples_per_mode),
    np.random.multivariate_normal(target_means[4], target_covs[4]/16, n_samples_per_mode),
])

# Randomly choose 3 points from each class
class0 = class0[np.random.choice(class0.shape[0], 3, replace=False)]
class1 = class1[np.random.choice(class1.shape[0], 3, replace=False)]


target_pdf = sum(multivariate_normal(mean=m, cov=c).pdf(pos)
                 for m, c in zip(target_means, target_covs))
target_pdf /= np.max(target_pdf)

# ---------------------- PLOTTING -----------------------------
plt.figure(figsize=(6, 6))

plt.rcParams.update({
    'font.size': 16,           # Global font size
    'axes.titlesize': 18,      # Title font size
    'axes.labelsize': 16,      # X/Y label size
    'xtick.labelsize': 14,     # X tick label size
    'ytick.labelsize': 14,     # Y tick label size
    'legend.fontsize': 15,     # Legend font size
})

# Blue faded background (source)
plt.contourf(x, y, source_pdf, levels=np.linspace(0.2, 1.0, 8), cmap='Blues', alpha=0.6)


# Red foreground (target)
plt.contourf(x, y, target_pdf, levels=np.linspace(0.2, 1.0, 8), cmap='Reds', alpha=0.6)

# Class points
plt.scatter(class0[:, 0], class0[:, 1], color='red', edgecolor='black', s=70,
            marker='o', label='Class 0')  # Square
plt.scatter(class1[:, 0], class1[:, 1], color='red', edgecolor='black', s=70,
           marker='s', label='Class 1')  # Circle

# Axis limits and labels
plt.xlim([-3, 6])
plt.ylim([-6, 3])
plt.xlim([-10, 10])
plt.ylim([-10, 10])
plt.xticks([])
plt.yticks([])
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.legend(loc='upper right', frameon=False)
plt.xlabel("")
plt.ylabel("")


from matplotlib.patches import Patch
# New colorblind-friendly colors
# Create custom legend handles using scatter (not Line2D)
class0_handle = plt.scatter([], [], color='red', edgecolor='black', s=70, marker='s', label='Class 0 Real Sample')
class1_handle = plt.scatter([], [], color='red', edgecolor='black', s=70, marker='o', label='Class 1 Real Sample')
class00_handle = plt.scatter([], [], color='black', edgecolor='black', s=70, marker='s', label='Class 0 generated')
##class11_handle = plt.scatter([], [], color='black', edgecolor='black', s=70, marker='o', label='Class 1 generated')

custom_legend = [
    Patch(facecolor='blue', edgecolor='blue', alpha=0.4, label='Source Domain'),
    Patch(facecolor='red', edgecolor='red', alpha=0.4, label='Target Domain'),
    class0_handle,
    class1_handle,
   class00_handle,
##    class11_handle,
]
plt.legend(handles=custom_legend, loc='upper left', frameon=False)



# Clean, whitespace-friendly layout
plt.tight_layout()
plt.savefig("tables/plot_multivariate_gaussian_clean.png", dpi=300, bbox_inches='tight')
plt.show()
