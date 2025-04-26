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

# ______________________ DATASET PREPARATION ______________________
# This script prepares the Caltech-101 dataset for training.
# It checks if the dataset is already unzipped and unzips it if not.
# It also moves the dataset to a specified target directory.
# It assumes the dataset is in a zip file located at DATA_DIR_ZIP.
# It also assumes the target directory is specified in TARGET_DIR.


# # Define paths
DATA_DIR_ZIP=/export/datasets/public/Caltech-101/caltech-101.zip
TARGET_DIR=/projets/Ymohammadi/DomainGuidance/datasets

# Create target directory
mkdir -p "$TARGET_DIR"

# Move zip file to target directory
echo "Moving caltech-101.zip to $TARGET_DIR..."
cp "$DATA_DIR_ZIP" "$TARGET_DIR/"

# Change to target directory
cd "$TARGET_DIR"

# Unzip the dataset
echo "Unzipping caltech-101.zip..."
unzip caltech-101.zip

# Final structure check
echo "Done!"
echo "Dataset extracted to: $TARGET_DIR/caltech-101/"

# Example path for training
echo "When training, set --data-path to: $TARGET_DIR/caltech-101/"

cd "$OLDPWD"


# ______________________ TRAINING ______________________
# This script trains the DiT model on the Caltech-101 dataset.

CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 train.py \
  --data-path $TARGET_DIR/caltech-101/ \
  --results-dir /export/livia/home/vision/Ymohammadi/DoG/results_finetune/ \
  --model DiT-XL/2 \
  --image-size 256 \
  --num-classes 101 \
  --total-steps 24000 \
  --log-every 100 \
  --ckpt-every 4000 \
  --global-batch-size 32 \
  --vae ema \
  --num-workers 4

# ______________________ Removing Dataset ______________________
# This script removes the dataset files after training.
  
echo "Removing Dataset files..."
rm -rf "$TARGET_DIR"
echo "Done!"