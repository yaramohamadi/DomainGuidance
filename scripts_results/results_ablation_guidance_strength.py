import os
import re
import pandas as pd

# Define the base path
base_path = "/home/ymbahram/scratch/results/DoG/artbench-10_processed/ablation_guidance_strengths"

# Initialize a dictionary to hold results
results = {
    "guidance_strength": [],
    "DoG": [],
    "CFG": [],
    "MG": [],
    "DGF": []
}

# Helper function to extract FD from file
def extract_fd(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("fd:"):
                return float(line.split("fd:")[1].strip())
    return None

# Walk through the directories
for root, dirs, files in os.walk(base_path):
    for file in files:
        if file.startswith("fd_inception") and file.endswith(".txt"):
            full_path = os.path.join(root, file)
            relative_path = os.path.relpath(full_path, base_path)
            
            # Identify the method and guidance strength
            if "dogfinetune_LATE_START_ITER12000_MG" in relative_path:
                match = re.search(r"MG([\d.]+)_W_TRAIN_DOG([\d.]+)", relative_path)
                if match:
                    method = "DGF"
                    strength = float(match.group(2))
            elif "baselines_finetune_W_CFG" in relative_path and "results_dog1_5" in relative_path:
                match = re.search(r"W_CFG([\d.]+)", relative_path)
                if match:
                    method = "DoG"
                    strength = float(match.group(1))
            elif "baselines_finetune_W_CFG" in relative_path and "results_cg1_5" in relative_path:
                match = re.search(r"W_CFG([\d.]+)", relative_path)
                if match:
                    method = "CFG"
                    strength = float(match.group(1))
            elif "baselines_mg_W_CFG" in relative_path:
                match = re.search(r"W_CFG([\d.]+)", relative_path)
                if match:
                    method = "MG"
                    strength = float(match.group(1))
            else:
                continue

            fd_value = extract_fd(full_path)
            if fd_value is not None:
                if strength not in results["guidance_strength"]:
                    results["guidance_strength"].append(strength)
                index = results["guidance_strength"].index(strength)
                while len(results["DoG"]) <= index:
                    results["DoG"].append(None)
                    results["CFG"].append(None)
                    results["MG"].append(None)
                    results["DGF"].append(None)
                results[method][index] = fd_value

# Convert to DataFrame and sort
df = pd.DataFrame(results)
df = df.sort_values(by="guidance_strength")
# Save to CSV
output_csv_path = "/home/ymbahram/projects/def-hadi87/ymbahram/DomainGuidance/tables/fd_guidance_strengths_summary.csv"
df.to_csv(output_csv_path, index=False)

print(df)
