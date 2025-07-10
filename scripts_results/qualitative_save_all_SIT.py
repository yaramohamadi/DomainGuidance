import os
import shutil

# === Config ===
root_dir = "/home/ens/AT74470/results/DoG"
save_dir = "./scripts_results/images_samples"
os.makedirs(save_dir, exist_ok=True)

chunk_index = 1  # 0 = first 50, 1 = second 50, 2 = third 50, ...
chunk_size = 50  # Number of images per chunk

# === Internal-to-clean mapping ===
method_name_map = {
    "baseline_mgfinetune_wtraincfg1.5": "MG",
    "baselines_finetune_W_CFG1.5/results_cg1": "Finetune",
    "baselines_finetune_W_CFG1.5/results_cg1_5": "CFG",
    "baselines_finetune_W_CFG1.5/results_dog1_5": "DoG",
}

# === Pretty display names ===
custom_dataset_names = {
    "Food101": "Food",
    "Sun397": "SUN",
    "Caltech-101": "Caltech",
    "Cub-200-2011": "CUB Bird",
    "Stanford-Cars": "Stanford Car",
    "Artbench-10": "ArtBench",
    "Ffhq256": "FFHQ",
}

# === Sample folder mapping ===
sample_subdirs = {
    "baseline_mgfinetune_wtraincfg1.5": "0024000",
    "baselines_finetune_W_CFG1.5/results_cg1": "0024000_cg1",
    "baselines_finetune_W_CFG1.5/results_cg1_5": "0024000_cg1_5",
    "baselines_finetune_W_CFG1.5/results_dog1_5": "0024000_dog1_5"
}

# === Image Extraction ===
for dataset_name in sorted(os.listdir(root_dir)):
    dataset_path = os.path.join(root_dir, dataset_name)
    if not os.path.isdir(dataset_path):
        continue

    clean_dataset_name = dataset_name.replace("_processed", "").replace("_", "-").capitalize()
    display_dataset_name = custom_dataset_names.get(clean_dataset_name, clean_dataset_name)

    for raw_method, sample_subdir in sample_subdirs.items():
        method_key = method_name_map[raw_method]
        method_folder = raw_method.split('/')[0] if '/' in raw_method else raw_method

        src_sample_path = os.path.join(dataset_path, "All_SiT", method_folder, "samples", sample_subdir)
        dst_path = os.path.join(save_dir, method_key, display_dataset_name)
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

        if not chunk:
            print(f"⚠️  {method_key:8s} | {display_dataset_name:20s} → No images found in chunk {chunk_index}")
            continue

        for img_file in chunk:
            src_file = os.path.join(src_sample_path, img_file)
            dst_file = os.path.join(dst_path, img_file)
            shutil.copy(src_file, dst_file)

        print(f"✅ {method_key:8s} | {display_dataset_name:20s} → Copied {len(chunk)} images from chunk {chunk_index}")
