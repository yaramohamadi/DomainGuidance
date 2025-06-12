import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d
import numpy as np




# === Data ===

# FID
fid_data = {
    "Iteration": [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 22000],
    "Artbench-10": [22.08, 21.22, 19.90, 19.26, 18.69, 18.15, 17.33, 17.34, 17.31, 16.86, 16.32, 16.79, 17.18, 17.26, 18.66, 20.07],
    "Caltech-101": [22.56, 22.33, 22.16, 21.92, 21.85, 21.81, 21.68, 21.68, 21.73, 21.76, 21.80, 21.82, 21.86, 22.52, 23.32, 25.24],
    "Cub-200-2011": [3.85, 3.86, 3.82, 3.82, 3.77, 3.80, 3.72, 3.69, 3.72, 3.69, 3.65, 3.75, 4.14, 4.34, 5.33, 6.98],
    "Ffhq256": [12.72, 12.16, 12.16, 12.16, 11.29, 11.29, 10.81, 10.81, 10.49, 10.73, 10.67, 10.52, 10.90, 11.44, 11.54, 13.64],
    "Food-101": [12.69, 11.82, 11.43, 11.29, 10.94, 10.83, 10.89, 10.85, 10.72, 10.76, 10.83, 11.04, 11.33, 11.96, 12.95, 14.55],
    "Stanford-cars": [10.31, 10.24, 9.76, 9.70, 9.88, 9.75, 9.45, 9.39, 9.39, 9.43, 9.35, 9.43, 9.72, 10.54, 11.69, 13.24],
}
fid_df = pd.DataFrame(fid_data)

# Precision
precision_data = {
    "Iteration": fid_data["Iteration"],
    "Artbench-10": [0.7805, 0.7838, 0.7808, 0.7853, 0.7841, 0.7877, 0.7826, 0.7804, 0.78, 0.7767, 0.7636, 0.7624, 0.743, 0.7264, 0.7111, 0.6943],
    "Caltech-101": [0.7267, 0.7331, 0.7425, 0.7484, 0.7460, 0.7533, 0.7491, 0.7500, 0.7568, 0.7553, 0.7477, 0.7355, 0.7173, 0.6905, 0.6677, 0.6169],
    "Cub-200-2011": [0.7998, 0.8040, 0.8113, 0.8113, 0.8226, 0.8182, 0.8260, 0.8208, 0.8234, 0.8124, 0.8048, 0.7848, 0.7638, 0.7439, 0.6950, 0.6371],
    "Ffhq256": [0.7851, 0.8007, 0.8007, 0.8007, 0.8106, 0.8106, 0.8023, 0.8023, 0.8078, 0.7999, 0.7951, 0.7882, 0.7773, 0.7680, 0.7565, 0.7336],
    "Food-101": [0.8517, 0.8498, 0.8559, 0.8674, 0.8699, 0.8630, 0.8727, 0.8755, 0.8735, 0.8749, 0.8727, 0.8676, 0.8568, 0.8453, 0.8310, 0.8155],
    "Stanford-cars": [0.7205, 0.7314, 0.7426, 0.7447, 0.7466, 0.7464, 0.7435, 0.7527, 0.7445, 0.7445, 0.7374, 0.7090, 0.6761, 0.6464, 0.6129, 0.5596],
}
precision_df = pd.DataFrame(precision_data)

# Recall
recall_data = {
    "Iteration": fid_data["Iteration"],
    "Artbench-10": [0.4419, 0.4528, 0.4646, 0.4704, 0.4712, 0.4743, 0.4915, 0.4924, 0.4917, 0.4996, 0.5067, 0.5213, 0.5318, 0.5424, 0.5482, 0.5527],
    "Caltech-101": [0.7024, 0.7095, 0.6910, 0.7020, 0.7072, 0.7088, 0.6960, 0.7021, 0.6984, 0.7037, 0.7267, 0.7266, 0.7533, 0.7593, 0.7541, 0.7736],
    "Cub-200-2011": [0.6999, 0.6862, 0.6846, 0.6846, 0.6716, 0.6724, 0.6701, 0.6788, 0.6727, 0.6904, 0.6944, 0.7132, 0.7247, 0.7465, 0.7657, 0.7871],
    "Ffhq256": [0.6141, 0.6054, 0.6054, 0.6054, 0.6305, 0.6305, 0.6310, 0.6310, 0.6280, 0.6288, 0.6234, 0.6496, 0.6481, 0.6513, 0.6589, 0.6516],
    "Food-101": [0.4774, 0.4930, 0.4865, 0.4819, 0.4900, 0.4926, 0.4889, 0.4883, 0.4995, 0.4907, 0.4962, 0.4995, 0.5079, 0.5094, 0.5135, 0.5096],
    "Stanford-cars": [0.5301, 0.5281, 0.5426, 0.5341, 0.5208, 0.5273, 0.5292, 0.5415, 0.5368, 0.5441, 0.5475, 0.5577, 0.5751, 0.5848, 0.5844, 0.5980],
}
recall_df = pd.DataFrame(recall_data)

# === Plotting ===
fig, axs = plt.subplots(1, 3, figsize=(14, 4), sharex=True)
xticks = [0, 5000, 10000, 15000, 20000]
xtick_labels = ['0', '5', '10', '15', '20']

highlight_start = 6000
highlight_end = 12000

# Plotting helper function
def plot_panel(ax, df, title, ylabel=None, smooth=False):
    for col in df.columns[1:]:
        x = df["Iteration"]
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
    ax.axvspan(highlight_start, highlight_end, color='gray', alpha=0.2)
    
    # ax.set_title(title)
    ax.set_xticks(xticks)
    ax.grid(True, linestyle='-', alpha=0.5)
    ax.set_xticklabels(xtick_labels)
    if ylabel:
        ax.set_ylabel(ylabel)

plot_panel(axs[0], fid_df, "FID (↓)", "FID (↓)", smooth=False)
plot_panel(axs[1], precision_df, "Precision (↑)", "Precision (↑)", smooth=True)
plot_panel(axs[2], recall_df, "Recall (↑)", "Recall (↑)", smooth=True)


for ax in axs:
    ax.set_xlabel("Iteration (K)")

# Legend at bottom
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.01), ncol=6)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("tables/latestart_ablation_results_highlighted.png", bbox_inches='tight', dpi=300)
plt.show()
