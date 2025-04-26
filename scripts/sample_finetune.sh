#!/bin/bash

# ______________________ CREATE ENVIRONMENT ______________________
# This script creates a Conda environment for the DiT model.
# It checks if the environment already exists and creates it if not.
# It also activates the environment after creation.

ENV_PATH="/projets/Ymohammadi/envs/DiT"  # or wherever your Conda installs

# Step 1: Check if environment path exists
if [ -d "$ENV_PATH" ]; then
    echo "Conda environment at '$ENV_PATH' already exists."
    # Step 2: Activate environment
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_PATH"
else
    echo "Environment not found at '$ENV_PATH'. Creating it..."
    conda create --prefix "$ENV_PATH" python=3.13.2 -y
    # Step 2: Activate environment
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_PATH"
    conda install pytorch torchvision pytorch-cuda=12.6 -c pytorch -c nvidia
    pip install timm diffusers accelerate pytorch-fid
fi

echo "Environment ready!"


# ______________________ SAMPLING ______________________
# This script samples images from the trained model.

# Finetuning without Guidance
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 sample_ddp.py \
  --model DiT-XL/2 \
  --vae ema \
  --sample-dir /export/livia/home/vision/Ymohammadi/DoG/results_pretrained/samples_CFG2 \
  --ckpt /projets/Ymohammadi/DomainGuidance/tmp/local_pretrained_ckpt.pt \
  --per-proc-batch-size 32 \
  --num-fid-samples 10 \
  --image-size 256 \
  --num-classes 101 \
  --cfg-scale 2 \
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