import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d
import numpy as np

# Define the color mapping from the reference image (manually matched)
color_order = {
    "Caltech": "#ff7f0e",    # Orange
    "FFHQ": "#d62728",           # Red
    "Stanford-Cars": "#8c564b",  # Brown
        "ArtBench": "#1f77b4",       # Blue
    "CUB-Birds": "#2ca02c",      # Green
    "Food": "#9467bd",       # Purple
}

# Adjust x-tick positions to only show 5 values
xticks = [0, 0.25, 0.50, 0.75, 1]

# Define the MG values for the cutoff threshold ablation
mg_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# FID data with new MG=0 values
fid_df = pd.DataFrame({
    "MG": mg_values,
    "Caltech": [30.02, 25.40, 23.01, 21.86, 21.68, 21.80, 21.94, 22.38, 22.62, 23.01, 23.01],
    "Food": [16.75, 14.96, 12.38, 11.08, 10.77, 10.45, 10.64, 10.85, 10.95, 10.93, 10.89],
    "Stanford-Cars": [16.24, 15.28, 13.43, 11.65, 9.15, 8.98, 9.36, 9.77, 10.25, 10.37, 10.36],
    "ArtBench": [23.36, 20.40, 17.03, 16.98, 16.84, 16.84, 16.78, 16.70, 16.91, 16.78, 16.69],
    "CUB-Birds": [9.35, 7.27, 4.85, 4.01, 3.94, 3.78, 3.77, 3.76, 3.77, 3.75, 3.75],
    "FFHQ": [15.94, 14.34, 12.73, 11.30, 10.80, 10.36, 10.98, 13.56, 15.03, 15.36, 15.29],
})

# Precision data with new MG=0 values
precision_df = pd.DataFrame({
    "MG": mg_values,
    "Caltech": [0.6, 0.665, 0.713, 0.744, 0.749, 0.757, 0.767, 0.782, 0.782, 0.794, 0.792],
    "Food": [0.79, 0.787, 0.805, 0.831, 0.844, 0.857, 0.866, 0.876, 0.877, 0.880, 0.878],
    "Stanford-Cars": [0.51, 0.516, 0.552, 0.60, 0.669, 0.714, 0.751, 0.767, 0.773, 0.777, 0.767],
    "ArtBench": [0.67, 0.699, 0.744, 0.746, 0.744, 0.757, 0.760, 0.767, 0.765, 0.762, 0.764],
    "CUB-Birds": [0.58, 0.628, 0.703, 0.735, 0.755, 0.787, 0.800, 0.816, 0.830, 0.825, 0.819],
    "FFHQ": [0.7021, 0.721, 0.743, 0.773, 0.793, 0.815, 0.823, 0.844, 0.847, 0.847, 0.851],
})

# Recall data with new MG=0 values
recall_df = pd.DataFrame({
    "MG": mg_values,
    "Caltech": [0.75, 0.739, 0.720, 0.705, 0.696, 0.687, 0.667, 0.634, 0.630, 0.618, 0.620],
    "Food": [0.52, 0.533, 0.543, 0.534, 0.520, 0.507, 0.496, 0.488, 0.484, 0.482, 0.479],
    "Stanford-Cars": [0.58, 0.603, 0.617, 0.621, 0.606, 0.567, 0.538, 0.513, 0.502, 0.500, 0.503],
    "ArtBench": [0.57, 0.554, 0.532, 0.533, 0.541, 0.520, 0.516, 0.516, 0.513, 0.512, 0.517],
    "CUB-Birds": [0.81, 0.802, 0.771, 0.750, 0.736, 0.706, 0.682, 0.682, 0.676, 0.671, 0.678],
    "FFHQ": [0.6747, 0.667, 0.663, 0.643, 0.63, 0.61, 0.58, 0.59, 0.573, 0.563, 0.575],
})


# Apply plot style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 20,
    'axes.titlesize': 22,
    'axes.labelsize': 19,
    'xtick.labelsize': 19,
    'ytick.labelsize': 18,
    'legend.fontsize': 17,
})
# Correct the order of the legend and plot lines to match the reference image
ordered_datasets = [
        "ArtBench",       # Blue
        "Caltech",    # Orange
        "CUB-Birds",      # Green
        "FFHQ",           # Red
        "Food",       # Purple
    "Stanford-Cars",  # Brown
]

# Regenerate plot with corrected dataset order
fig, axs = plt.subplots(1, 3, figsize=(10, 4), sharex=True)

# Updated plotting with correct order and colors
def plot_panel(ax, df, title, ylabel=None, smooth=False):
    for col in ordered_datasets:
        if col not in df.columns:
            continue
        x = df["MG"]
        y = df[col]
        mask = y.notna()
        x = np.array(x)[mask]
        y = np.array(y)[mask]
        color = color_order.get(col, None)
        if smooth:
            x_smooth = np.linspace(x.min(), x.max(), 300)
            y_smooth = make_interp_spline(x, y)(x_smooth)
            ax.plot(x_smooth, y_smooth, label=col, color=color)
        else:
            smoothed = gaussian_filter1d(y, sigma=1)
            ax.plot(x, smoothed, label=col, color=color)
    ax.axvspan(0.4, 0.6, color='gray', alpha=0.2)
    ax.set_title(title)
    ax.set_xticks([0, 0.3, 0.6, 0.9])
    ax.set_xlim([-0.05, 1.05])
    ax.grid(True, linestyle='-', alpha=0.5)
    if ylabel:
        ax.set_ylabel(ylabel)

# Plot panels
plot_panel(axs[0], fid_df, "FID (↓)", "", smooth=False)
plot_panel(axs[1], precision_df, "Precision (↑)", "", smooth=True)
plot_panel(axs[2], recall_df, "Recall (↑)", "", smooth=True)

for ax in axs:
    ax.set_xlabel("Time-step")

axs[1].set_ylim([0.49, 0.91])
axs[2].set_ylim([0.39, 0.81])

# Create and display legend in correct order
handles = [plt.Line2D([], [], color=color_order[k], label=k) for k in ordered_datasets]
labels = ordered_datasets

print(ordered_datasets)

fig.legend(
    handles, labels,
    loc='lower center',
    bbox_to_anchor=(0.5, -0.1),
    ncol=3,
    columnspacing=1.2,
    handletextpad=0.5,
    frameon=True,
    fancybox=True,
    edgecolor='gray',
    facecolor='white',
)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("cutoff_threshold_ablation_final_ordered.png", bbox_inches='tight', dpi=300)
plt.show()
