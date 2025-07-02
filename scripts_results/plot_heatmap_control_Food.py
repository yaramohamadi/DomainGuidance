import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

# Update matplotlib settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 24,
    'axes.titlesize': 24,
    'axes.labelsize': 22,
    'xtick.labelsize': 22,
    'ytick.labelsize': 21,
    'legend.fontsize': 20,
})

# New dataset to visualize
data = {
    r"$\omega$": [1, 1.5, 2, 3, 4, 5],
    r"12.0 ($95\% \leq 1.25$)": [15.40, 12.42, 12.77, 15.40, 17.13, 18.25],
    r"6.0 ($95\% \leq 1.5$)": [14.39, 11.17, 11.41, 15.29, 18.39, 20.32],
    r"3.0 ($95\% \leq 2$)": [13.37, 10.23, 11.42, 18.34, 24.13, 27.76],
    r"1.5 ($95\% \leq 3$)": [12.01, 9.71, 11.56, 19.00, 25.95, 30.55],
    r"1.0 ($95\% \leq 4$)": [11.67, 9.55, 11.80, 19.70, 27.05, 32.79],
    r"0.7 ($95\% \leq 5$)": [11.45, 9.86, 12.89, 21.12, 28.83, 35.28],
}

# Create DataFrame
df = pd.DataFrame(data)
df = df.set_index("$\omega$").T

# Create figure and axes
fig = plt.figure(figsize=(6, 5))
ax = fig.add_axes([0.1, 0.1, 0.65, 0.8])
cbar_ax = fig.add_axes([0.78, 0.1, 0.05, 0.8])

# Create custom colormap based on inferno_r and append gray
base_cmap = plt.cm.get_cmap("inferno_r", 256)
newcolors = base_cmap(np.linspace(0, 1, 256))
gray = np.array([0.7, 0.7, 0.7, 1.0])  # RGBA gray
newcolors = np.vstack([newcolors])
custom_cmap = ListedColormap(newcolors)

# Plot main heatmap
sns.heatmap(df, cmap=custom_cmap, cbar=True, cbar_ax=cbar_ax,
            linewidths=0.5, linecolor='gray', square=True, ax=ax)


# Axis labels and ticks
ax.set_ylabel(r"$\lambda$", rotation=0)
ax.set_xlabel(r"Test-time $\omega$")
ax.tick_params(axis='both')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

# Save figure
plt.savefig("tables/heatmap_sampling_custombar_updated_food.png", dpi=300, bbox_inches="tight")
plt.close()
