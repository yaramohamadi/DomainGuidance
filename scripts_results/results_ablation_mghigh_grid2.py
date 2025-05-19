import pandas as pd
import matplotlib.pyplot as plt
import os

# === Path setup ===
root_dir = "./"
output_dir = "tables"
full_output_path = os.path.join(root_dir, output_dir)

# === Configs ===
metrics = ["recall", "precision"]  # FD removed
models = ["dinov2", "inception"]
datasets = ["Artbench-10", "Cub-200-2011", "Stanford-cars", "Caltech-101", "Food-101"]

# === Plotting function ===
def plot_all_datasets_together(model_name, data_dict, save_path):
    for metric in metrics:
        df = data_dict[metric]
        plt.figure(figsize=(8, 8))  # Taller figure
        for dataset in datasets:
            if dataset in df.columns:
                plt.plot(df["MG value"], df[dataset], marker='o', label=dataset)
        plt.title(f"{metric.capitalize()} - {model_name.capitalize()}")
        plt.xlabel("MG value")
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        output_file = os.path.join(save_path, f"ablation_{metric}_{model_name}.png")
        plt.savefig(output_file, dpi=300)
        plt.close()
        print(f"Saved: {output_file}")

# === Main loop ===
for model in models:
    data = {}
    for metric in metrics:
        file_name = f"final_ablation_mghigh_{metric}_{model}.csv"
        file_path = os.path.join(full_output_path, file_name)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            data[metric] = df
        else:
            print(f"Missing: {file_path}")
    if len(data) == len(metrics):  # Ensure both precision and recall loaded
        plot_all_datasets_together(model, data, save_path=full_output_path)
