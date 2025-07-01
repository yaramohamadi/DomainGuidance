import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 24,             # base font size
    'axes.titlesize': 24,        # title font size for each subplot
    'axes.labelsize': 22,        # x/y axis label font size
    'xtick.labelsize': 22,       # x tick label font size
    'ytick.labelsize': 21,       # y tick label font size
    'legend.fontsize': 20,       # legend font size
})

# Data
data = {
    r"$\omega$": [1, 1.5, 2, 3, 4, 5],
    r"12.0 ($95\% \leq 1.25$)": [14.35, 10.06, 10.06, 14.36, 18.45, 20.69],
    r"6.0 ($95\% \leq 1.5$)": [12.2, 8.96, np.nan, 15.83, 19.67, 22.05],
    r"3.0 ($95\% \leq 2$)": [11.56, 10.05, 12.41, 17.66, 21.48, 24.18],
    r"1.5 ($95\% \leq 3$)": [10.55, 11.09, 14.34, 20.13, 24.99, 28.83],
    r"1.0 ($95\% \leq 4$)": [10.83, np.nan, 15.55, 21.79, 27.47, 32.28],
    r"0.7 ($95\% \leq 5$)": [10.78, 13.07, 16.70, 24.21, 31.08, 36.98]
}
df = pd.DataFrame(data)
df = df.set_index("$\omega$").T
# Interpolate missing values along rows (axis=0)
df = df.interpolate(method='polynomial', order=2, axis=1)

# Create figure and axes manually
fig = plt.figure(figsize=(6, 5))  # Adjust as needed
ax = fig.add_axes([0.1, 0.1, 0.65, 0.8])     # [left, bottom, width, height]
cbar_ax = fig.add_axes([0.78, 0.1, 0.05, 0.8])  # wider colorbar

# Plot heatmap with custom colorbar axis
sns.heatmap(df, cmap="inferno_r", cbar=True, cbar_ax=cbar_ax,
            linewidths=0.5, linecolor='gray', square=True, ax=ax)

# Labels and ticks
ax.set_ylabel(r"$\lambda$", rotation=0)
ax.set_xlabel(r"Test-time $\omega$")
ax.tick_params(axis='both')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

plt.savefig("./tables/heatmap_sampling_custombar.png", dpi=300, bbox_inches="tight")
plt.close()
