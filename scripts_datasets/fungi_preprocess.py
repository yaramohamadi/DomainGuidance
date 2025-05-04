import os
import shutil
import pandas as pd

# ==== CONFIGURATION ====
train_csv = "/export/datasets/public/diffusion_datasets/fungi/DF20-train_metadata_PROD-2.csv"


test_csv = "/export/datasets/public/diffusion_datasets/fungi/DF20-public_test_metadata_PROD-2.csv"
image_root = "/export/datasets/public/diffusion_datasets/fungi/DF20_300"
output_dir = "/export/datasets/public/diffusion_datasets/fungi/organized_by_species"

# ==== CREATE OUTPUT FOLDER ====
os.makedirs(output_dir, exist_ok=True)

# ==== READ METADATA ====
df_train = pd.read_csv(train_csv, sep="\t" if train_csv.endswith(".tsv") else ",")
df_test = pd.read_csv(test_csv, sep="\t" if test_csv.endswith(".tsv") else ",")

# ==== COMBINE ====
df = pd.concat([df_train, df_test], ignore_index=True)

# ==== COPY IMAGES ====
for idx, row in df.iterrows():
    if pd.isna(row["species"]):
        continue  # skip rows with missing species

    species = str(row["species"]).strip().replace("/", "_")
    image_name = str(row["image_path"]).strip()

    src_path = os.path.join(image_root, image_name)
    dst_folder = os.path.join(output_dir, species)
    dst_path = os.path.join(dst_folder, image_name)

    os.makedirs(dst_folder, exist_ok=True)

    if os.path.isfile(src_path):
        shutil.copy(src_path, dst_path)
    else:
        print(f"[WARNING] Missing image: {src_path}")
