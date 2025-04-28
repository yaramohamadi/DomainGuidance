#!/bin/bash

# ====================== CONFIGURATION ======================
ENV_PATH="/projets/Ymohammadi/envs/DiT"
DATA_DIR_ZIP="/export/datasets/public/Caltech-101/caltech-101.zip"
TARGET_DIR="/projets/Ymohammadi/DomainGuidance/datasets"
GENERATED_DIR="/export/livia/home/vision/Ymohammadi/DoG/results_dogfinetune1_5/001-DiT-XL-2/samples/0024000/samples_CFG1"
REAL_DATA_DIR="$TARGET_DIR/caltech-101"
FLAT_REAL_DIR="$TARGET_DIR/caltech-101-flat"
REAL_STATS_PATH="/export/livia/home/vision/Ymohammadi/DoG/dataset_stats/real_stats_caltech101.npz"
CHECKPOINT_DIR="/export/livia/home/vision/Ymohammadi/DoG/results_dogfinetune1_5/001-DiT-XL-2/checkpoints/0024000.pt"

# Define CUDA devices here (will apply everywhere)
CUDA_DEVICES="2,3"
FID_DEVICE="cuda:0"  # Device to compute FID (usually just one GPU)

# ====================== FUNCTIONS ======================

create_environment() {
    echo ">>> Checking Conda environment..."
    if [ -d "$ENV_PATH" ]; then
        echo "Environment at '$ENV_PATH' already exists. Activating it..."
    else
        echo "Environment not found. Creating new environment at '$ENV_PATH'..."
        conda create --prefix "$ENV_PATH" python=3.13.2 -y
    fi
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_PATH"
    conda install pytorch torchvision pytorch-cuda=12.6 -c pytorch -c nvidia
    pip install timm diffusers accelerate pytorch-fid
    echo ">>> Environment ready!"
}

prepare_dataset() {
    echo ">>> Preparing dataset..."
    mkdir -p "$TARGET_DIR"
    cp "$DATA_DIR_ZIP" "$TARGET_DIR/"
    cd "$TARGET_DIR"
    unzip -o caltech-101.zip
    echo ">>> Dataset extracted to: $REAL_DATA_DIR"
    cd "$OLDPWD"
}

flatten_dataset() {
    echo ">>> Flattening dataset..."

    INPUT_DIR="$1"    # original real dataset (with class folders)
    OUTPUT_DIR="$2"   # flattened output folder

    mkdir -p "$OUTPUT_DIR"

    find "$INPUT_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) -exec cp {} "$OUTPUT_DIR" \;

    echo ">>> Dataset flattened to: $OUTPUT_DIR"
}


flatten_and_resize_dataset() {
    echo ">>> Flattening and resizing real dataset to 256x256..."

    python3 <<EOF
import os
from multiprocessing import Pool
from PIL import Image
from tqdm import tqdm

input_dir = "$REAL_DATA_DIR"
output_dir = "$FLAT_REAL_DIR"
os.makedirs(output_dir, exist_ok=True)

def process_image(task):
    in_path, out_path = task
    try:
        img = Image.open(in_path).convert('RGB')
        width, height = img.size
        crop_size = min(width, height)

        # Center crop manually
        left = (width - crop_size) / 2
        top = (height - crop_size) / 2
        right = (width + crop_size) / 2
        bottom = (height + crop_size) / 2

        img = img.crop((left, top, right, bottom))
        img = img.resize((299, 299), Image.BILINEAR)
        img.save(out_path)
    except Exception as e:
        print(f"Failed to process {in_path}: {e}")

tasks = []
for root, _, files in os.walk(input_dir):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            in_path = os.path.join(root, file)
            out_filename = f"{os.path.basename(root)}_{file}"  # prevent name collision
            out_path = os.path.join(output_dir, out_filename)
            tasks.append((in_path, out_path))

print(f"Found {len(tasks)} images to flatten (crop+resize)...")
with Pool(16) as pool:
    list(tqdm(pool.imap(process_image, tasks), total=len(tasks)))

print(">>> Flattening (crop+resize) complete.")
EOF
}

save_real_stats() {
    echo ">>> Checking if real dataset statistics exist..."
    if [ -f "$REAL_STATS_PATH" ]; then
        echo "Real stats already exist at: $REAL_STATS_PATH"
    else
        echo "Saving statistics for flattened real dataset..."
        flatten_and_resize_dataset
        python -m pytorch_fid --save-stats "$FLAT_REAL_DIR" "$REAL_STATS_PATH" --device "cuda:0"
        echo ">>> Real dataset statistics saved!"
    fi
}

run_sampling() {
    echo ">>> Starting image sampling..."

    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES torchrun --nproc_per_node=2 sample_ddp.py \
      --model DiT-XL/2 \
      --vae ema \
      --sample-dir "$GENERATED_DIR" \
      --ckpt "$CHECKPOINT_DIR" \
      --per-proc-batch-size 32 \
      --num-fid-samples 10000 \
      --image-size 256 \
      --num-classes 101 \
      --cfg-scale 1 \
      --num-sampling-steps 50

# Finetuning with Guidance 1.5
# CUDA_VISIBLE_DEVICES=$CUDA_DEVICES torchrun --nproc_per_node=2 sample_ddp.py \
#   --model DiT-XL/2 \
#   --vae ema \
#   --sample-dir /projets/Ymohammadi/DomainGuidance/samples/cfg_1_5 \
#   --ckpt /projets/Ymohammadi/DomainGuidance/results/000-DiT-XL-2/checkpoints/0024000.pt \
#   --per-proc-batch-size 32 \
#   --num-fid-samples 10000 \
#   --image-size 256 \
#   --num-classes 101 \
#   --cfg-scale 1.5 \
#   --num-sampling-steps 50
# 
#    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES torchrun --nproc_per_node=2 sample_dog_ddp.py \
#      --model DiT-XL/2 \
#      --vae ema \
#      --sample-dir "$GENERATED_DIR" \
#      --ckpt "$CHECKPOINT_DIR" \
#      --per-proc-batch-size 32 \
#      --num-fid-samples 10000 \
#      --image-size 256 \
#      --num-classes 101 \
#      --cfg-scale 1.5 \
#      --num-sampling-steps 50

    echo ">>> Sampling completed!"
}



calculate_fid() {
    echo ">>> Calculating FID score..."
    python -m pytorch_fid "$REAL_STATS_PATH" "$GENERATED_DIR" --device "$FID_DEVICE"
    echo ">>> FID calculation done!"
}

cleanup_dataset() {
    echo ">>> Cleaning up dataset files..."
    rm -rf "$TARGET_DIR"
    echo ">>> Dataset removed."
}

# ====================== MAIN ======================

create_environment
prepare_dataset
save_real_stats
run_sampling
calculate_fid
cleanup_dataset


echo ">>> All tasks completed successfully!"
