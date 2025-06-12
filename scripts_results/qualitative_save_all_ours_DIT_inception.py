import os
import shutil

# === Config ===
root_dir = "/home/ens/AT74470/results/DoG"
save_dir = "./scripts_results/images_samples"
os.makedirs(save_dir, exist_ok=True)

chunk_index = 1  # 0 = first 50, 1 = second 50, etc.
chunk_size = 50

# === Dataset display name mapping ===
custom_dataset_names = {
    "Food101": "Food",
    "Sun397": "SUN",
    "Caltech-101": "Caltech",
    "Cub-200-2011": "CUB Bird",
    "Stanford-Cars": "Stanford Car",
    "Artbench-10": "ArtBench",
    "Ffhq256": "FFHQ",
}

# === Custom method parameters for each dataset ===
custom_method_config = {
    "cub-200-2011_processed": {"ITER": "6000", "MG": "0.8"},
    "caltech-101_processed": {"ITER": "7000", "MG": "0.4"},
    "stanford-cars_processed": {"ITER": "5000", "MG": "0.6"},
    "food-101_processed": {"ITER": "8000", "MG": "0.8"},
    "artbench-10_processed": {"ITER": "12000", "MG": "0.8"},
    "ffhq256": {"ITER": "8000", "MG": "0.5"},
    # Add more datasets here
}

# === Image Extraction for Custom Method ===
for dataset_folder, params in custom_method_config.items():
    dataset_path = os.path.join(root_dir, dataset_folder)
    if not os.path.isdir(dataset_path):
        print(f"❌ Dataset folder missing: {dataset_path}")
        continue

    # Clean dataset name
    clean_dataset_name = dataset_folder.replace("_processed", "").replace("_", "-").capitalize()
    display_dataset_name = custom_dataset_names.get(clean_dataset_name, clean_dataset_name)

    iter_str = params["ITER"]
    mg_str = params["MG"]
    sample_folder = f"dogfinetune_LATE_START_ITER{iter_str}_MG{mg_str}/samples/0024000"
    src_sample_path = os.path.join(dataset_path, "ablation_latestart", sample_folder)

    dst_path = os.path.join(save_dir, "Ours", display_dataset_name)
    os.makedirs(dst_path, exist_ok=True)

    if not os.path.isdir(src_sample_path):
        print(f"❌ Missing: {src_sample_path}")
        continue

    image_files = sorted([
        f for f in os.listdir(src_sample_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    start = chunk_index * chunk_size
    end = start + chunk_size
    chunk = image_files[start:end]

    for img_file in chunk:
        src_file = os.path.join(src_sample_path, img_file)
        dst_file = os.path.join(dst_path, img_file)
        shutil.copy(src_file, dst_file)

    print(f"✅ Ours      | {display_dataset_name:20s} → Copied {len(chunk)} images from chunk {chunk_index}")
