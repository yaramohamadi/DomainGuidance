import pandas as pd
import matplotlib.pyplot as plt
import os

# === Path setup ===
root_dir = "./"
output_dir = "tables"
full_output_path = os.path.join(root_dir, output_dir)

# === Configs ===
metric = "fd"
models = ["dinov2", "inception"]
datasets = ["Artbench-10", "Cub-200-2011", "Stanford-cars", "Caltech-101", "Food-101"]

# === Load data for both models ===
data = {}
for model in models:
    file_name = f"final_ablation_mghigh_{metric}_{model}.csv"
    file_path = os.path.join(full_output_path, file_name)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        data[model] = df
    else:
        print(f"Missing: {file_path}")

# === Check and plot only if both models are available ===
if len(data) == 2:
    fig, axs = plt.subplots(len(datasets), len(models), figsize=(10, 15), sharex=True)

    for i, dataset in enumerate(datasets):
        for j, model in enumerate(models):
            ax = axs[i, j]
            df = data[model]
            if dataset in df.columns:
                ax.plot(df["MG value"], df[dataset], marker='o')
            ax.set_title(f"{dataset} - {model}")
            ax.set_xlabel("MG value")
            ax.set_ylabel("FD")
            ax.grid(True)

    fig.suptitle("FD Ablation Results per Dataset", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    output_file = os.path.join(full_output_path, "ablation_summary_fd.png")
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Saved: {output_file}")
