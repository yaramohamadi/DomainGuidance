#!/bin/bash

# ______________________ CREATE ENVIRONMENT ______________________
# This script creates a Conda environment for the DiT model.
# It checks if the environment already exists and creates it if not.
# It also activates the environment after creation.

ENV_PATH="/projets/Ymohammadi/envs/DiT"  # or wherever your Conda installs

# Step 1: Check if environment path exists
if [ -d "$ENV_PATH" ]; then
    echo "Conda environment at '$ENV_PATH' already exists."
else
    echo "Environment not found at '$ENV_PATH'. Creating it..."
    conda env create --prefix "$ENV_PATH" -f environment_updated.yml
fi

# Step 2: Activate environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PATH"

echo "Environment ready!"


# ______________________ SAMPLING ______________________
# This script samples images from the trained model.

# Finetuning without Guidance
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 sample_ddp.py \
  --model DiT-XL/2 \
  --vae ema \
  --sample-dir /projets/Ymohammadi/DomainGuidance/samples/cfg_1 \
  --ckpt /projets/Ymohammadi/DomainGuidance/results/000-DiT-XL-2/checkpoints/0024000.pt \
  --per-proc-batch-size 32 \
  --num-fid-samples 10000 \
  --image-size 256 \
  --num-classes 101 \
  --cfg-scale 1 \
  --num-sampling-steps 50

# Finetuning with Guidance 1.5
# CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 sample_ddp.py \
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
# CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 sample_dog_ddp.py \
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