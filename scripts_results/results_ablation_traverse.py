import os
import pandas as pd
import re
import matplotlib.pyplot as plt

# === Config ===
root_dir = "/home/ens/AT74470/results/DoG/cub-200-2011_processed/ablation_train_traverse/"
metric_model = "inception"  # e.g., 'inception' or 'dinov2'
metric_name = "fd"          # e.g., 'fd', 'precision', 'recall', etc.
output_dir = "./tables"
os.makedirs(output_dir, exist_ok=True)
output_csv_path = os.path.join(output_dir, f"comparison_{metric_name}_{metric_model}.csv")

# Mapping for clarity
method_folders = {
    "DGF": os.path.join(root_dir, "dogfinetune_LATE_START_ITER3000_MG0.75", "results"),
    "DoG": os.path.join(root_dir, "baselines_finetune", "results_dog1_5"),
    "CFG": os.path.join(root_dir, "baselines_finetune", "results_cg1_5"),
    "MG": os.path.join(root_dir, "baseline_mgfinetune", "results"),
    "Finetune": os.path.join(root_dir, "baselines_finetune", "results_cg1"),
}

# === Helper ===
def extract_metric(filepath, metric_key):
    try:
        with open(filepath, "r") as f:
            for line in f:
                if line.startswith(f"{metric_key}:"):
                    return float(line.split(":")[1].strip())
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return None

# === Collect Data ===
records = []

for method, path in method_folders.items():
    if not os.path.isdir(path):
        continue

    for filename in os.listdir(path):
        if not filename.startswith(f"{metric_name}_{metric_model}"):
            continue

        # Parse filename
        match = re.match(
            rf"fd_{metric_model}_(.+?)_processed-(\d+)_.*\.txt", filename
        )
        if not match:
            continue

        dataset_name, step = match.groups()
        step = int(step)
        dataset_name = dataset_name.replace("-", "_").replace("cub_200_2011", "CUB")

        # Read metric
        value = extract_metric(os.path.join(path, filename), metric_name)

        if value is None:
            continue

        records.append({
            "method": method,
            "dataset": dataset_name,
            "step": step,
            "value": value
        })

# === Create DataFrame ===
df = pd.DataFrame(records)

print(df)
pivot_df = df.pivot_table(index=["dataset", "step"], columns="method", values="value")
pivot_df = pivot_df.sort_index()
df_sorted = df.sort_values(by=["dataset", "step"])

print(pivot_df)
# Save
pivot_df.to_csv(output_csv_path)
pivot_df.head()




# === Load Table ===
df = pd.read_csv(output_csv_path)

# === Show Table ===
print("Full Table:")
print(df)

# Optional: if 'dataset' and 'step' were part of the index
if 'dataset' in df.columns and 'step' in df.columns:
    df.set_index(['dataset', 'step'], inplace=True)

# === Example: Filter or Access
print("\nSubset for CUB:")
print(df.loc['CUB'])


import matplotlib.pyplot as plt

# Re-read and filter for plotting
df = pd.read_csv(output_csv_path)
df.set_index(["dataset", "step"], inplace=True)
target_dataset = "CUB"
dataset_df = df.loc[target_dataset]
dataset_df = dataset_df[dataset_df.index >= 12000]

# Plot size and styles
plt.figure(figsize=(6, 7))  # Smaller width, taller height

# Define styles
line_styles = {
    "Finetune": {"linestyle": "-", "linewidth": 1.8},
    "MG": {"linestyle": "-", "linewidth": 1.8},
    "DGF": {"linestyle": "-", "linewidth": 2.8},
}
colors = {
    "Finetune": "gray",
    "MG": "#88aaff",       # pale blue
    "DGF": "green",
    "DoG": "red",
    "CFG": "orange",
}

markers = {
    "DGF": "^",
    "MG": "^",
    "CFG": "x",
    "DoG": "x",
    "Finetune": "o",
}

# Plot main lines
for method in ["Finetune", "MG", "DGF"]:
    if method in dataset_df.columns:
        plt.plot(
            dataset_df.index / 1000,  # Convert to thousands
            dataset_df[method],
            marker=markers[method],
            markersize=10,
            color=colors[method],
            label=method,
            **line_styles[method]
        )

# Plot CFG and DoG as dots + vertical lines
for i, method in enumerate(["CFG", "DoG"]):
    if method in dataset_df.columns:
        last_step = dataset_df[method].dropna().index.max()
        x = last_step / 1000
        offset = (0.4 if method == "CFG" else 0.8)
        value = dataset_df.loc[last_step, method]
        finetune_value = dataset_df.loc[last_step, "Finetune"]

        # Plot vertical line slightly offset and behind
        plt.plot([x + offset, x + offset], [finetune_value, value],
                 linestyle="--", color=colors[method], alpha=0.8, zorder=1)

        # Plot point on top
        plt.scatter([x + offset], [value], color=colors[method], marker=markers[method], label=method, s=150, zorder=5)




# Final plot formatting
plt.title("CUB Bird", fontsize=20, weight="bold")
plt.xlabel("Training Iterations (K)", fontsize=18)
plt.ylabel("FID10K", fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True)
plt.legend(title="Method", fontsize=16, title_fontsize=17)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"comparison_{metric_name}_{metric_model}_focused_final.png"))
plt.show()
