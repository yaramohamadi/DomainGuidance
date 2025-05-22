import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# === Output path ===
output_dir = "./tables"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "efficiency_vs_performance.png")

# === Data ===
methods = [
    "Fine-tuning", 
    "+ Classifier-free Guidance", 
    "+ Domain Guidance", 
    "Model Guidance", 
    "Domain Guided Fine-tuning"
]
fid_scores = [19.14, 14.46, 13.09, 14.13, 12.61]
sampling_costs = [1, 2, 2, 1, 1]
reference_fid = fid_scores[-1]

# === Color assignment
bar_colors = ["orange" if cost == 1 else "royalblue" for cost in sampling_costs]

# === Custom x positions
x = [0, 0.65, 1.3, 2.2, 3.1]

# === Plot
fig, ax = plt.subplots(figsize=(6, 4))

# Draw bars and FID labels
for i, (method, fid) in enumerate(zip(methods, fid_scores)):
    color = bar_colors[i]
    bar = ax.bar(
        x[i],
        fid,
        color=color,
        edgecolor='black',
        width=0.6,
        zorder=3
    )
    label_color = 'red' if i == len(fid_scores) - 1 else 'black'
    weight = 'bold' if i == len(fid_scores) - 1 else 'normal'
    ax.text(
        x[i],
        fid + 0.3,
        f"{fid:.2f}",
        ha='center',
        va='bottom',
        fontsize=10,
        color=label_color,
        fontweight=weight
    )

# Red reference line
ax.axhline(y=reference_fid, color='red', linestyle='--', linewidth=1.5, zorder=4)

# Style
tick_labels = []
for i, method in enumerate(methods):
    if i == len(methods) - 1:
        tick_labels.append(r"\textbf{\textcolor{red}{" + method + "}}")  # Will show in LaTeX rendering environments
    else:
        tick_labels.append(method)

ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=15, ha='right', fontsize=10)
ax.get_xticklabels()[-1].set_color("red")
ax.get_xticklabels()[-1].set_fontweight("bold")

ax.set_ylabel("Average FID-10K ↓", fontsize=12)
ax.set_ylim(10, max(fid_scores) + 2)
plt.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)

# Legend
legend_elements = [
    mpatches.Patch(color='orange', label='1 forward pass'),
    mpatches.Patch(color='royalblue', label='2 forward pass'),
]
ax.legend(handles=legend_elements, loc='upper right')

# Save
plt.tight_layout()
plt.savefig(output_path, dpi=300)
print(f"✅ Saved to {output_path}")
