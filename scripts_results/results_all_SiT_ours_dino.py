import os
import pandas as pd

# === Config ===
root_dir = "/export/datasets/public/diffusion_datasets/tmp_weights/"# "/home/ens/AT74470/results/DoG"
save_dir = "./scripts_results/tables"
metric_to_extract = "fd"  # 'fd', 'precision', 'recall', etc.
model_prefix = "fd_dinov2" # fd_dinov2 fd_inception

os.makedirs(save_dir, exist_ok=True)

# === Internal-to-clean mapping ===
method_name_map = {
    "dogfinetune_nodropout": "Ours",
}

# === Dataset-specific parameters for Inception ===
PAIR_MAP = {
    "stanford-cars_processed": "8000,1",
    "caltech-101_processed": "8000,0.7",
    "food-101_processed": "8000,1",
    "artbench-10_processed": "8000,0.9",
    "cub-200-2011_processed": "6000,1",
    "ffhq256": "10000,0.6"
}

# === Pretty display names for LaTeX ===
custom_method_names = {
    "Finetune": "Fine-tuning (w/o guidance)",
    "CFG": "+ Classifier-free guidance",
    "DoG": "+ Domain guidance",
    "MG": "Class guided fine-tuning",
    "Ours": "\\textbf{Domain guided fine-tuning}",
}

custom_dataset_names = {
    "Food101": "Food",
    "Sun397": "SUN",
    "Caltech-101": "Caltech",
    "Cub-200-2011": "CUB Bird",
    "Stanford-Cars": "Stanford Car",
    "Artbench-10": "ArtBench",
    "Domainnet-Real": "DF-20M",
    "Sun": "SUN",
    "Ffhq": "FFHQ",
}

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
data = {}
dataset_names_cleaned = []

for dataset_name in sorted(os.listdir(root_dir)):
    dataset_path = os.path.join(root_dir, dataset_name)
    if not os.path.isdir(dataset_path):
        continue

    # Clean dataset name and track for remapping
    clean_dataset_name = dataset_name.replace("_processed", "").replace("_", "-").capitalize()
    dataset_names_cleaned.append(clean_dataset_name)

    # Get params for this dataset
    params_str = PAIR_MAP.get(dataset_name.lower())
    if not params_str:
        print(f"⚠️ Skipping {dataset_name}: No ITER/MG parameters found in PAIR_MAP.")
        continue

    iter_val, mg_val = params_str.split(",")

    raw_method_paths = {
        "dogfinetune_nodropout": os.path.join(
            dataset_path,
            "SiT_dino_ours",
            f"dogfinetune_LATE_START_ITER{iter_val}_MG{mg_val}_W_TRAIN_DOG1.5",
            "results"
        ),
    }

    for raw_method, path in raw_method_paths.items():
        display_name = method_name_map[raw_method]
        if display_name not in data:
            data[display_name] = {}

        value = None
        if os.path.isdir(path):
            for file in os.listdir(path):
                if file.startswith(model_prefix) and file.endswith(".txt"):
                    value = extract_metric(os.path.join(path, file), metric_to_extract)
                    break
        data[display_name][clean_dataset_name] = value

# === Build DataFrame ===
df = pd.DataFrame.from_dict(data, orient='index')
df = df[sorted(df.columns)]
df.index.name = "Method"

# Add average column
avg_col_name = f"Average {metric_to_extract.capitalize()}"
df[avg_col_name] = df.mean(axis=1, skipna=True)

# Save raw CSV
csv_path = os.path.join(save_dir, f"{model_prefix}_{metric_to_extract}_table.csv")
df.to_csv(csv_path)
print(f"✅ Saved table with average to {csv_path}")

print(df)

# === Format for LaTeX with highlighting ===
def highlight_best(df, higher_is_better=True):
    styled_df = df.copy()
    for col in df.columns:
        col_values = df[col]
        if col_values.isnull().all():
            continue

        sorted_vals = col_values.sort_values(ascending=not higher_is_better)
        best_val = sorted_vals.iloc[0]
        second_best_val = sorted_vals.iloc[1] if len(sorted_vals) > 1 else None

        for idx in df.index:
            val = df.at[idx, col]
            if pd.isna(val):
                styled_df.at[idx, col] = ""
            elif val == best_val:
                styled_df.at[idx, col] = f"\\textbf{{{val:.2f}}}"
            elif val == second_best_val:
                styled_df.at[idx, col] = f"\\underline{{{val:.2f}}}"
            else:
                styled_df.at[idx, col] = f"{val:.2f}"
    return styled_df

# Apply LaTeX formatting
higher_is_better = metric_to_extract in ["precision", "recall", "coverage"]
latex_df = highlight_best(df, higher_is_better=higher_is_better)

# Rename rows (methods) and columns (datasets)
latex_df.rename(index=custom_method_names, inplace=True)
latex_df.rename(columns=custom_dataset_names, inplace=True)

# === LaTeX Export ===
# === Generate tabular code with vertical lines after method and before average ===
num_main_cols = len(latex_df.columns) - 1  # all except the last column
col_format = "l|" + "c" * num_main_cols + "|c"  # vertical lines around average column

latex_tabular = latex_df.to_latex(
    index=True,
    escape=False,
    column_format=col_format,
    header=True,
    bold_rows=False
)

# === Post-process to insert custom horizontal lines ===
lines = latex_tabular.splitlines()
new_lines = []
for i, line in enumerate(lines):
    new_lines.append(line)

    # Insert line after "+ Domain guidance"
    if "+ Domain guidance" in line:
        new_lines.append("\\midrule")

    # Insert line before average column header
    if i > 0 and "Average" in line and "\\toprule" not in line:
        new_lines.insert(i, "\\cmidrule(lr){" + str(num_main_cols + 1) + "-" + str(num_main_cols + 1) + "}")



# === Wrap in table* with caption and label ===
latex_table = f"""\\begin{{table*}}[ht]
\\centering
\\caption{{{metric_to_extract.upper()} results for {model_prefix.replace('_', ' ').title()}}}
\\label{{tab:{model_prefix}_{metric_to_extract}}}
{chr(10).join(new_lines)}
\\end{{table*}}"""


# Save .tex
latex_path = os.path.join(save_dir, f"{model_prefix}_{metric_to_extract}_table.tex")
with open(latex_path, "w") as f:
    f.write(latex_table)

print(f"✅ LaTeX table saved to {latex_path}")
