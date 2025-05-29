import os
import pandas as pd
import re

# === Config ===
root_dir = "/export/datasets/public/diffusion_datasets/tmp_weights/" #   "/home/ens/AT74470/results/DoG/" # 
metric_to_extract = "fd"         # Options: 'fd', 'precision', 'recall', 'density', 'coverage'
model_type = "dinov2"            # Options: 'dinov2' or 'inception'
output_dir = "./tables"
os.makedirs(output_dir, exist_ok=True)
output_csv_path = os.path.join(
    output_dir, f"mega_ablation_latestart_{metric_to_extract}_{model_type}.csv"
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

# === Main collection ===
table_data = {}

for dataset_folder in sorted(os.listdir(root_dir)):
    dataset_path = os.path.join(root_dir, dataset_folder)
    ablation_path = os.path.join(dataset_path, "mega_ablation_latestart")
    if not os.path.isdir(ablation_path):
        continue

    dataset_name = dataset_folder.replace("_processed", "").capitalize()
    
    for iter_folder in sorted(os.listdir(ablation_path)):
        iter_path = os.path.join(ablation_path, iter_folder)
        if not os.path.isdir(iter_path):
            continue

        # Path to the 'results' folder
        results_path = os.path.join(iter_path, "results")
        if not os.path.isdir(results_path):
            continue

        # Extract iteration number
        match = re.search(r"ITER(\d+)", iter_folder)
        if not match:
            continue
        iter_num = int(match.group(1))

        # Find the correct file (fd_dinov2... or fd_inception...)
        metric_file = None
        for file in os.listdir(results_path):
            if file.startswith(f"fd_{model_type}") and file.endswith(".txt"):
                metric_file = os.path.join(results_path, file)
                print(f"Found: {metric_file}")
                break
        if not metric_file:
            continue

        # Extract value
        metric_value = extract_metric(metric_file, metric_to_extract)
        if metric_value is None:
            continue

        # Initialize iteration row if needed
        if iter_num not in table_data:
            table_data[iter_num] = {}
        table_data[iter_num][dataset_name] = metric_value

# === Create DataFrame ===
df = pd.DataFrame.from_dict(table_data, orient="index")
df = df.sort_index()  # Sort by iteration
df = df[sorted(df.columns)]  # Sort dataset columns
df.index.name = "Iteration"

# === Save to CSV ===
os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
df.to_csv(output_csv_path)
print(f"✅ Saved metric table to {output_csv_path}")
print(df)
