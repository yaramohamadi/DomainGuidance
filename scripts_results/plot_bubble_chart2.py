import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# ---------- 0.  data  -------------------------------------------------
methods_FID = [
    (3.6614e14, 19.144, "Finetune",          "DiT/XL2"),
    (7.32281e14, 14.46,  "Finetune + CFG",   "DiT/XL2"),
    (7.32281e14, 13.09,  "Finetune + DoG",   "DiT/XL2"),
    (3.6614e14, 14.126, "MG",                "DiT/XL2"),
    (3.6614e14, 12.336, "DoGFit (OURS)",     "DiT/XL2"),
    (4.54014e14, 17.144, "Finetune",         "SiT/XL2"),
    (1.08378e15, 12.46,  "Finetune + CFG",   "SiT/XL2"),
    (1.08378e15, 10.09,  "Finetune + DoG",   "SiT/XL2"),
    (4.54014e14, 12.126, "MG",               "SiT/XL2"),
    (4.54014e14,  9.80,  "DoGFit (OURS)",    "SiT/XL2"),
]

methods_FD = [
    (3.6614e14, 461.45, "Finetune",          "DiT/XL2"),
    (7.32281e14, 311.03, "Finetune + CFG",   "DiT/XL2"),
    (7.32281e14, 245.31, "Finetune + DoG",   "DiT/XL2"),
    (3.6614e14, 312.78, "MG",                "DiT/XL2"),
    (3.6614e14, 244.23, "DoGFit (OURS)",     "DiT/XL2"),
    (4.54014e14, 389.57, "Finetune",         "SiT/XL2"),
    (1.08378e15, 280.94, "Finetune + CFG",   "SiT/XL2"),
    (1.08378e15, 240.32, "Finetune + DoG",   "SiT/XL2"),
    (4.54014e14, 297.16, "MG",               "SiT/XL2"),
    (4.54014e14, 245.29, "DoGFit (OURS)",    "SiT/XL2"),
]

# ---------- 1.  pull arrays ------------------------------------------
flops  = np.array([m[0] for m in methods_FID]) / 1e12      # TFLOPs
avgFID = np.array([m[1] for m in methods_FID])             #   x-axis
fdDIN  = np.array([m[1] for m in methods_FD])               #   y-axis
labels = [m[2] for m in methods_FID]
models = [m[3] for m in methods_FID]

method_colors = {
    "Finetune":          "#1f77b4",
    "Finetune + CFG":    "#2ca02c",
    "Finetune + DoG":    "#d62728",
    "MG":                "#9467bd",
    "DoGFit (OURS)":     "#ff7f0e",
}
colors      = [method_colors[l] for l in labels]
alphas      = [1.0 if "DoGFit" in l else 0.3 for l in labels]

# ---------- 2.  bubble size ∝ flops ----------------------------------
max_area   = 40      # (px²) size of largest bubble – tweak freely
sizes      = flops / flops.max() * max_area

# ---------- 3.  dashed lines (same label across backbones) -----------
method_lines = {}
for i, lab in enumerate(labels):
    method_lines.setdefault(lab, {"x": [], "y": []})
    method_lines[lab]["x"].append(avgFID[i])
    method_lines[lab]["y"].append(fdDIN[i])

# ---------- 4.  plot --------------------------------------------------
plt.figure(figsize=(7, 5))

# connect the two points for each method
for lab, pts in method_lines.items():
    plt.plot(pts["x"], pts["y"], '--', color=method_colors[lab], alpha=0.6)

for i in range(len(labels)):
    marker = 'o' if models[i] == "DiT/XL2" else 's'
    plt.scatter(
        avgFID[i], fdDIN[i],
        s=sizes[i]**2,
        color=colors[i], alpha=alphas[i], marker=marker, edgecolor='black'
    )
    # annotate SiT points (square markers)
    if models[i] == "SiT/XL2":
        y_off = 14 if "DoGFit" in labels[i] else 8
        plt.text(avgFID[i] + 0.5, fdDIN[i] + y_off,
                 labels[i], color=colors[i],
                 fontweight='bold' if "DoGFit" in labels[i] else 'normal')

plt.xlabel(r"Average FID (↓)")
plt.ylabel(r"Average $\mathrm{FD}_{\text{DINOV2}}$ (↓)")
plt.grid(True, linestyle='--', alpha=1)

# legend for shape
legend_elems = [
    Line2D([0], [0], marker='o', color='w', label='DiT/XL2',
           markerfacecolor='gray', markeredgecolor='black', markersize=10),
    Line2D([0], [0], marker='s', color='w', label='SiT/XL2',
           markerfacecolor='gray', markeredgecolor='black', markersize=10)
]
plt.legend(handles=legend_elems, title="Model Type", loc='upper right')

plt.tight_layout()
plt.savefig("tables/bubble_chart_fd_vs_fid_bubbles.png", dpi=300)
plt.show()
