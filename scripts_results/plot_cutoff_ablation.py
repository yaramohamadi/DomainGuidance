import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d
import numpy as np
import ace_tools as tools

# Update matplotlib settings
plt.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 15,
    'axes.labelsize': 13,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
})

# Define the new cutoff threshold ablation data
mg_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# FID
fid_df = pd.DataFrame({
    "MG": mg_values,
    "Caltech-101": [25.40, 23.01, 21.86, 21.68, 21.80, 21.94, 22.38, 22.62, 23.01, 23.01],
    "Food-101": [14.96, 12.38, 11.08, 10.77, 10.45, 10.64, 10.85, 10.95, 10.93, 10.89],
    "Stanford-cars": [15.28, 13.43, np.nan, 9.15, 8.98, 9.36, 9.77, 10.25, 10.37, 10.36],
    "Artbench-10": [20.40, 17.03, 16.98, 16.84, 16.84, np.nan, 16.70, 16.91, 16.78, 16.69],
    "Cub-200-2011": [7.27, 4.85, 4.01, 3.94, 3.78, np.nan, 3.76, 3.77, 3.75, 3.75],
    "Ffhq256": [14.34, 12.73, 11.30, np.nan, np.nan, np.nan, np.nan, 15.03, 15.36, 15.29]
})

# Precision
precision_df = pd.DataFrame({
    "MG": mg_values,
    "Caltech-101": [0.665, 0.713, 0.744, 0.749, 0.757, 0.767, 0.782, 0.782, 0.794, 0.792],
    "Food-101": [0.787, 0.805, 0.831, 0.844, 0.857, 0.866, 0.876, 0.877, 0.880, 0.878],
    "Stanford-cars": [0.516, 0.552, np.nan, 0.669, 0.714, 0.751, 0.767, 0.773, 0.777, 0.767],
    "Artbench-10": [0.699, 0.744, 0.746, 0.744, 0.757, np.nan, 0.767, 0.765, 0.762, 0.764],
    "Cub-200-2011": [0.628, 0.703, 0.735, 0.755, 0.787, np.nan, 0.816, 0.830, 0.825, 0.819],
    "Ffhq256": [0.721, 0.743, 0.773, np.nan, np.nan, np.nan, np.nan, 0.847, 0.847, 0.851]
})

# Recall
recall_df = pd.DataFrame({
    "MG": mg_values,
    "Caltech-101": [0.739, 0.720, 0.705, 0.696, 0.687, 0.667, 0.634, 0.630, 0.618, 0.620],
    "Food-101": [0.533, 0.543, 0.534, 0.520, 0.507, 0.496, 0.488, 0.484, 0.482, 0.479],
    "Stanford-cars": [0.603, 0.617, np.nan, 0.606, 0.567, 0.538, 0.513, 0.502, 0.500, 0.503],
    "Artbench-10": [0.554, 0.532, 0.533, 0.541, 0.520, 0.516, 0.516, 0.513, 0.512, 0.517],
    "Cub-200-2011": [0.802, 0.771, 0.750, 0.736, 0.706, 0.682, 0.682, 0.676, 0.671, 0.678],
    "Ffhq256": [0.667, 0.663, 0.643, np.nan, np.nan, np.nan, np.nan, 0.573, 0.563, 0.575]
})

# Helper function for plotting
def plot_panel(ax, df, title, ylabel=None, smooth=False):
    for col in df.columns[1:]:
        x = df["MG"]
        y = df[col]
        mask = y.notna()
        x = np.array(x)[mask]
        y = np.array(y)[mask]
        if smooth:
            x_smooth = np.linspace(x.min(), x.max(), 300)
            y_smooth = make_interp_spline(x, y)(x_smooth)
            ax.plot(x_smooth, y_smooth, label=col)
        else:
            smoothed = gaussian_filter1d(y, sigma=1)
            ax.plot(x, smoothed, label=col)
    ax.set_title(title)
    ax.set_xlabel("Cutoff Threshold (MG)")
    ax.grid(True, linestyle='-', alpha=0.5)
    if ylabel:
        ax.set_ylabel(ylabel)

# Create figure
fig, axs = plt.subplots(1, 3, figsize=(14, 4), sharex=True)
plot_panel(axs[0], fid_df, "FID (↓)", "FID", smooth=False)
plot_panel(axs[1], precision_df, "Precision (↑)", "Precision", smooth=True)
plot_panel(axs[2], recall_df, "Recall (↑)", "Recall", smooth=True)

# Shared legend
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.01), ncol=6)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("cutoff_threshold_ablation_results.png", bbox_inches='tight', dpi=300)
tools.display_image("cutoff_threshold_ablation_results.png")
