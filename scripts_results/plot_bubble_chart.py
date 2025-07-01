import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Global font sizes
plt.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 15,
    'axes.labelsize': 13,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
})

# Define color per method type
method_colors = {
    "Finetune": "#1f77b4",         # blue
    "Finetune + CFG": "#2ca02c",   # green
    "Finetune + DoG": "#d62728",   # red
    "MG": "#9467bd",               # purple
    "DoGFit (OURS)": "#ff7f0e",       # orange
}

# Data
methods = [
    (3.6614e14, 19.144, "Finetune", "DiT/XL2"),
    (7.32281e14, 14.46, "Finetune + CFG", "DiT/XL2"),
    (7.32281e14, 13.09, "Finetune + DoG", "DiT/XL2"),
    (3.6614e14, 14.126, "MG", "DiT/XL2"),
    (3.6614e14, 12.336, "DoGFit (OURS)", "DiT/XL2"),
    (4.54014e14, 17.144, "Finetune", "SiT/XL2"),
    (1.08378e15, 12.46, "Finetune + CFG", "SiT/XL2"),
    (1.08378e15, 10.09, "Finetune + DoG", "SiT/XL2"),
    (4.54014e14, 12.126, "MG", "SiT/XL2"),
    (4.54014e14, 9.80, "DoGFit (OURS)", "SiT/XL2"),
]

methods = [
    (3.6614e14, 461.45, "Finetune", "DiT/XL2"),
    (7.32281e14, 311.03, "Finetune + CFG", "DiT/XL2"),
    (7.32281e14, 245.31, "Finetune + DoG", "DiT/XL2"),
    (3.6614e14, 312.78, "MG", "DiT/XL2"),
    (3.6614e14, 244.23, "DoGFit (OURS)", "DiT/XL2"),
    (4.54014e14, 389.57, "Finetune", "SiT/XL2"),
    (1.08378e15, 280.94, "Finetune + CFG", "SiT/XL2"),
    (1.08378e15, 240.32, "Finetune + DoG", "SiT/XL2"),
    (4.54014e14, 297.16, "MG", "SiT/XL2"),
    (4.54014e14, 245.29, "DoGFit (OURS)", "SiT/XL2"),
]


# Convert to TFLOPS and extract
flops = np.array([m[0] for m in methods]) / 1e12
fid = np.array([m[1] for m in methods])
labels = [m[2] for m in methods]
model_types = [m[3] for m in methods]
colors = [method_colors[label] for label in labels]
alphas = [1.0 if "DoGFit" in label else 0.3 for label in labels]
fontweights = ['bold' if "DoGFit" in label else 'normal' for label in labels]

# Organize data per method type to draw lines
method_lines = {}
for i, label in enumerate(labels):
    if label not in method_lines:
        method_lines[label] = {"x": [], "y": []}
    method_lines[label]["x"].append(flops[i])
    method_lines[label]["y"].append(fid[i])
# Plot
plt.figure(figsize=(7, 5))
# Lines connecting same method across models
for label, data in method_lines.items():
    x_vals = np.array(data["x"])
    y_vals = np.array(data["y"])
    # plt.plot(x_vals, y_vals, linestyle='--', color=method_colors[label], alpha=0.6)

# Annotate only on square (SiT/XL2) markers
for i in range(len(methods)):
    marker = 'o' if model_types[i] == "DiT/XL2" else 's'
    plt.scatter(flops[i], fid[i], s=500, color=colors[i],
                alpha=alphas[i], marker=marker)

    if model_types[i] == "SiT/XL2":
        is_dogfit = "DoGFit" in labels[i]
        # bump the text up for DoGFit so it sits higher
        y_offset = 10 
        fontweight = 'bold' if is_dogfit else 'normal'

        plt.text(
            flops[i] + 35,            # x-offset
            fid[i] + y_offset,        # y-offset
            labels[i],
            ha='left',
            color=colors[i],
            fontweight=fontweight,
            alpha=1
        )


# Axis setup
plt.xlabel("Sampling TFLOPS")
plt.ylabel(r"Average $\text{FD}_{\text{DINOV2}}$ (↓)")
plt.xlim([300, 1400])
# plt.ylim([9, 21])
plt.grid(True, linestyle='--', alpha=1)

# Custom legend for model type shapes
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='DiT/XL2',
           markerfacecolor='gray', markeredgecolor='black', markersize=10),
    Line2D([0], [0], marker='s', color='w', label='SiT/XL2',
           markerfacecolor='gray', markeredgecolor='black', markersize=10)
]
plt.legend(handles=legend_elements, title="Model Type", loc='upper right')

# Save
plt.tight_layout()
plt.savefig("tables/bubble_chart_fid_vs_flops_lines_midlabel.png", dpi=300)
plt.show()
