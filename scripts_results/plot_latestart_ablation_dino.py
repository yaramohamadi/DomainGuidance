import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d

# Updated DINO results

iterations = [
    0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 22000
]


plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 20,             # base font size
    'axes.titlesize': 22,        # title font size for each subplot
    'axes.labelsize': 19,        # x/y axis label font size
    'xtick.labelsize': 19,       # x tick label font size
    'ytick.labelsize': 18,       # y tick label font size
    'legend.fontsize': 17,       # legend font size
})


# FID values
fid_data = {
    "Iteration": iterations,
    "ArtBench": [
        306.8102, 284.7417873, 274.399235, 269.0624221, 263.4002598, 264.7566567, 272.7303078,
        271.3133647, 279.916, 292.4717, 306.8102, 329.6944
    ],
    "Caltech": [
        381.2042285, 379.281088, 378.798641, 377.156305, 376.986548, 379.281088, 381.2042285,
        386.5787838, 386.8907556, 391.1427744, 412.4564823, 454.0826285
    ],
    "CUB-Birds": [
        158.2790277, 145.8542657, 146.6448507, 143.4236433, 146.0297964, 149.7036733, 158.2790277,
        173.6719018, 192.2207533, 221.6294745, 264.4861091, 326.7150698
    ],
    "FFHQ": [
        305.4223691, 293.3255177, 283.830439, 282.3160756, 283.6773077, 282.1121094, 284.0498918,
        286.9336331, 290.6631792, 301.4002551, 326.419861, 365.5820646
    ],
    "Food": [
        314.0616982, 300.6924432, 292.3376315, 293.2435753, 298.7341063, 314.0616982, 328.287094,
        351.2902175, 380.792518, 417.7205444, 468.2533092, 529.5434175
    ],
    "Stanford-Cars": [
        156.3147823, 148.861377, 145.0262274, 147.2045736, 145.564975, 150.1171409, 157.6608683,
        171.1145196, 188.8842719, 207.7688835, 256.2966841, 306.2372515
    ]
}

# Precision values
precision_data = {
    "Iteration": iterations,
    "ArtBench": [
        0.9201, 0.9271, 0.9193, 0.9166, 0.9102, 0.9054, 0.8998, 0.8914, 0.8853, 0.8812, 0.8672, 0.8554
    ],
    "Caltech": [
        0.660135992, 0.666474588, 0.668664285, 0.671775959, 0.666474588, 0.659135992, 0.648, 0.637, 0.61, 0.57, 0.545, 0.51
    ],
    "CUB-Birds": [
        0.3646, 0.3763, 0.3787, 0.3814, 0.3646, 0.3657, 0.3416, 0.3203, 0.2927, 0.2592, 0.2289, 0.1877
    ],
    "FFHQ": [
        0.8266, 0.8289, 0.8347, 0.8299, 0.8366, 0.8294, 0.8266, 0.8185, 0.8103, 0.7961, 0.771, 0.7538
    ],
    "Food": [
        0.5952, 0.5948, 0.6021, 0.5932, 0.5952, 0.5846, 0.5641, 0.5467, 0.5233, 0.5018, 0.4777, 0.4448
    ],
    "Stanford-Cars": [
        0.5768, 0.5812, 0.5691, 0.5768, 0.5811, 0.5771, 0.5548, 0.5273, 0.4995, 0.4698, 0.4208, 0.3727
    ]
}

# Recall values
recall_data = {
    "Iteration": iterations,
    "ArtBench": [
        0.1875, 0.1753, 0.1875, 0.2013, 0.2136, 0.2075, 0.2078, 0.2141, 0.2163, 0.2149, 0.217, 0.2164
    ],
    "Caltech": [
        0.595, 0.602795897, 0.612795897, 0.600898928, 0.617264031, 0.621989167, 0.620, 0.622, 0.627, 0.643, 0.648, 0.65
    ],
    "CUB-Birds": [
        0.821, 0.824, 0.8351, 0.836, 0.8392, 0.8461, 0.8587, 0.8732, 0.89, 0.9064, 0.9202, 0.9376
    ],
    "FFHQ": [
        0.4151, 0.4155, 0.429, 0.4418, 0.4428, 0.4337, 0.4425, 0.4555, 0.4666, 0.4668, 0.468, 0.4589
    ],
    "Food": [
        0.6012, 0.5958, 0.6109, 0.6208, 0.6438, 0.6437, 0.6416, 0.6572, 0.6533, 0.6572, 0.6628, 0.6639
    ],
    "Stanford-Cars": [
        0.8263, 0.818, 0.8334, 0.8228, 0.8264, 0.8313, 0.8325, 0.839, 0.8378, 0.84, 0.8379, 0.8335
    ]
}

# Create dataframes
fid_df = pd.DataFrame(fid_data)
precision_df = pd.DataFrame(precision_data)
recall_df = pd.DataFrame(recall_data)

# === Plotting ===
fig, axs = plt.subplots(1, 3, figsize=(10, 4), sharex=True)
xticks = [0, 5000, 10000, 15000, 20000]
xtick_labels = ['0', '5', '10', '15', '20']

highlight_start = 3000
highlight_end = 9000

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
    ax.set_title(title)
    ax.set_xticks(xticks)
    ax.grid(True, linestyle='-', alpha=0.5)
    ax.set_xticklabels(xtick_labels)
    if ylabel:
        ax.set_ylabel(ylabel)

plot_panel(axs[0], fid_df,  r"$\text{FD}_{\text{DINOV2}}$ (↓)", "", smooth=False)
plot_panel(axs[1], precision_df, r"Precision$_{\text{DINOV2}}$ (↑)", "", smooth=True)
plot_panel(axs[2], recall_df, r"Recall$_{\text{DINOV2}}$ (↑)", "", smooth=True)


for ax in axs:
    ax.set_xlabel("Iteration (K)")

#axs[1].set_ylim([0.49, 0.91])
#axs[2].set_ylim([0.39, 0.81])
# Full list of handles and labels
handles, labels = axs[0].get_legend_handles_labels()
# Get legend handles and labels
handles, labels = axs[0].get_legend_handles_labels()

# Pad to even number if needed (e.g., make total 6 entries for 3 per row)
target_cols = 3
if len(handles) % target_cols != 0:
    pad_count = target_cols - (len(handles) % target_cols)
    for _ in range(pad_count):
        handles.append(plt.Line2D([], [], linestyle=''))  # Invisible handle
        labels.append('')  # Empty label

fig.legend(
    handles, labels,
    loc='lower center',
    bbox_to_anchor=(0.5, -0.1),
    ncol=3,  # or whatever number of columns you use
    columnspacing=1.2,
    handletextpad=0.5,
    frameon=True,               # <-- enables the legend box
    fancybox=True,             # <-- optional: rounded corners
    edgecolor='gray',         # <-- box border color
    facecolor='white',         # <-- background color
)
# Adjust layout
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("latestart_ablation_results_highlighted_DINO.png", bbox_inches='tight', dpi=300)
plt.show()
