import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# New data to be visualized
data = {
    r"$\omega$": [1, 1.5, 2, 3, 4, 5],
    r"12.0 ($95\% \leq 1.25$)": [22.79, 19.98, 18.73, 18.40, 19.21, 20.18],
    r"6.0 ($95\% \leq 1.5$)": [21.69, 17.46, 17.23, 21.58, 25.72, 28.76],
    r"3.0 ($95\% \leq 2$)": [20.56, 17.19, 18.27, 25.18, 31.27, 37.27],
    r"1.5 ($95\% \leq 3$)": [19.37, 17.11, 19.25, 28.27, 37.18, 43.92],
    r"1.0 ($95\% \leq 4$)": [358.49, 358.49, 358.36, 359.30, 359.81, 359.89],
    r"0.7 ($95\% \leq 5$)": [350.00, 350.27, 350.59, 351.14, 351.45, 351.59],
}

# Create DataFrame
df = pd.DataFrame(data)
df = df.set_index("$\omega$").T

# Create figure and axes
fig = plt.figure(figsize=(6, 5))
ax = fig.add_axes([0.1, 0.1, 0.65, 0.8])
cbar_ax = fig.add_axes([0.78, 0.1, 0.05, 0.8])

# Separate values to be masked for color scaling but shown visually
high_val_mask = df >= 300
masked_df = df.mask(high_val_mask)
# Custom colormap from inferno_r but ending in gray
from matplotlib.colors import ListedColormap

# Use inferno_r colormap and append gray
base_cmap = plt.cm.get_cmap("inferno_r", 256)
newcolors = base_cmap(np.linspace(0, 1, 256))
gray = np.array([0.7, 0.7, 0.7, 1.0])  # RGBA gray
newcolors = np.vstack([newcolors, gray])
custom_cmap = ListedColormap(newcolors)

# Plot with masking
sns.heatmap(masked_df, cmap=custom_cmap, cbar=True, cbar_ax=cbar_ax,
            linewidths=0.5, linecolor='gray', square=True, ax=ax,
            mask=high_val_mask, vmin=df[~high_val_mask].min().min(), vmax=45)

# Overlay the high values with fixed gray fill
sns.heatmap(df, cmap=ListedColormap([gray]), cbar=False,
            linewidths=0.5, linecolor='gray', square=True, ax=ax,
            mask=~high_val_mask)


# Labels and ticks
ax.set_ylabel(r"$\lambda$", rotation=0)
ax.set_xlabel(r"Test-time $\omega$")
ax.tick_params(axis='both')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

# Save the heatmap
plt.savefig("tables/heatmap_sampling_custombar_updated_art.png", dpi=300, bbox_inches="tight")
plt.close()
