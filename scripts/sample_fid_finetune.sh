#!/bin/bash

# ====================== CONFIGURATION ======================
ENV_PATH="/projets/Ymohammadi/envs/DiT"
DATA_DIR_ZIP="/export/datasets/public/Caltech-101/caltech-101.zip"
TARGET_DIR="/projets/Ymohammadi/DomainGuidance/datasets"
GENERATED_DIR="/export/livia/home/vision/Ymohammadi/DoG/results_finetune/004-DiT-XL-2/samples_CFG1"
REAL_DATA_DIR="$TARGET_DIR/caltech-101"
REAL_STATS_PATH="/export/livia/home/vision/Ymohammadi/DoG/dataset_stats/real_stats_caltech101.npz"

# Define CUDA devices here (will apply everywhere)
CUDA_DEVICES="2,3"
FID_DEVICE="cuda:2"  # Device to compute FID (usually just one GPU)

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

precompute_real_stats() {
    if [ -f "$REAL_STATS_PATH" ]; then
        echo ">>> Real dataset statistics already exist at '$REAL_STATS_PATH'. Skipping computation."
    else
        echo ">>> Real dataset statistics not found. Computing and saving to '$REAL_STATS_PATH'..."
        python -m pytorch_fid --save-stats "$REAL_DATA_DIR" "$REAL_STATS_PATH" --device cuda:2
        echo ">>> Real dataset statistics saved!"
    fi
}

run_sampling() {
    echo ">>> Starting image sampling..."

    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES torchrun --nproc_per_node=2 sample_ddp.py \
      --model DiT-XL/2 \
      --vae ema \
      --sample-dir "$GENERATED_DIR" \
      --ckpt /projets/Ymohammadi/DomainGuidance/tmp/local_pretrained_ckpt.pt \
      --per-proc-batch-size 32 \
      --num-fid-samples 10000 \
      --image-size 256 \
      --num-classes 101 \
      --cfg-scale 1 \
      --num-sampling-steps 50

    echo ">>> Sampling completed!"

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
# CUDA_VISIBLE_DEVICES=$CUDA_DEVICES torchrun --nproc_per_node=2 sample_dog_ddp.py \
#   --model DiT-XL/2 \
#   --vae ema \
#   --sample-dir /projets/Ymohammadi/DomainGuidance/samples/dog_1_5 \
#   --ckpt /projets/Ymohammadi/DomainGuidance/results/000-DiT-XL-2/checkpoints/0024000.pt \
#   --per-proc-batch-size 32 \
#   --num-fid-samples 10000 \
#   --image-size 256 \
#   --num-classes 101 \
#   --cfg-scale 1.5 \
#   --num-sampling-steps 50
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
precompute_real_stats
run_sampling
calculate_fid
cleanup_dataset

echo ">>> All tasks completed successfully!"