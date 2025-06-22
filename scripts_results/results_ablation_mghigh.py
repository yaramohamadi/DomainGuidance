import os
import pandas as pd
import re

# === Config ===
root_dir =  "/home/ens/AT74470/results/DoG/" # "/home/ymbahram/scratch/results/DoG/" # 
# root_dir = "/export/datasets/public/diffusion_datasets/tmp_weights/"
metric_to_extract = "fd"         # Options: 'fd', 'precision', 'recall', 'density', 'coverage'
model_type = "inception"            # Options: 'dinov2' or 'inception'
output_dir = "./tables"
os.makedirs(output_dir, exist_ok=True)
output_csv_path = os.path.join(
    output_dir, f"ablation_best_latestart_mg{metric_to_extract}_{model_type}.csv"
)

# === Helper ===
def extract_metric(filepath, metric_name):
    try:
        with open(filepath, "r") as f:
            for line in f:
                if line.startswith(f"{metric_name}:"):
                    return float(line.split(":")[1].strip())
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return None

# === Data Collection ===
table_data = {}

for dataset_folder in sorted(os.listdir(root_dir)):
    dataset_path = os.path.join(root_dir, dataset_folder)
    ablation_path = os.path.join(dataset_path, "mega_ablation_mg")
    if not os.path.isdir(ablation_path):
        continue

    dataset_name = dataset_folder.replace("_processed", "").capitalize()

    for mg_folder in sorted(os.listdir(ablation_path)):
        mg_path = os.path.join(ablation_path, mg_folder)
        if not os.path.isdir(mg_path):
            continue

        # Extract MG value (e.g., from MG0.6)
        match = re.search(r"MG([0-9.]+)(?:_|$)", mg_folder, re.IGNORECASE)

        
        if not match:
            continue

        print(mg_folder)
        mg_value = float(match.group(1))

        # Go to results folder and find metric file
        results_path = os.path.join(mg_path, "results")
        if not os.path.isdir(results_path):
            continue

        metric_file = None
        for file in os.listdir(results_path):
            if file.startswith(f"fd_{model_type}") and file.endswith(".txt"):
                metric_file = os.path.join(results_path, file)
                break
        if not metric_file:
            continue

        # Extract metric value
        value = extract_metric(metric_file, metric_to_extract)
        if value is None:
            continue

        # Store
        if mg_value not in table_data:
            table_data[mg_value] = {}
        table_data[mg_value][dataset_name] = value

# === Create and Save DataFrame ===
df = pd.DataFrame.from_dict(table_data, orient="index")
df = df.sort_index()                  # Sort by MG values
df = df[sorted(df.columns)]          # Sort dataset columns
df.index.name = "MG value"

os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
df.to_csv(output_csv_path)
print(f"✅ Saved ablation table to {output_csv_path}")
print(df)