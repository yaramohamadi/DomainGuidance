#!/bin/bash

# ========== GLOBAL CONFIGURATION ==========
SERVER="bool"
CUDA_DEVICES="1"
EXPERIMENT_PRENAME="All_SiT"
MODEL="SiT-XL/2" # DiT-XL/2 or SiT-XL/2

# ========== DATASET TO SCRIPT MAPPING ==========
# Format: dataset_name script_name

declare -a TASKS=(
  "caltech-101_processed run_baseline_mg.sh"
  "caltech-101_processed run_baselines_finetune.sh"
)

# ========== EXECUTION LOOP ==========
for TASK in "${TASKS[@]}"; do
  read -r DATASET SCRIPT <<< "$TASK"

  echo "=============================================="
  echo "Running $SCRIPT on dataset $DATASET"
  echo "Server: $SERVER | CUDA Devices: $CUDA_DEVICES"
  echo "----------------------------------------------"
  
  bash "scripts/$SCRIPT" \
    --dataset "$DATASET" \
    --server "$SERVER" \
    --cuda_devices "$CUDA_DEVICES" \
    --experiment_prename "$EXPERIMENT_PRENAME" \
    --model "$MODEL"

  echo "✅ Finished $SCRIPT on $DATASET"
  echo "=============================================="
done