#!/bin/bash

# ========== GLOBAL CONFIGURATION ==========
SERVER="taylor"
CUDA_DEVICES="0,1"
EXPERIMENT_PRENAME="DiT_inception_all_DiffFit"

# ========== DATASET TO SCRIPT MAPPING ==========
# Format: dataset_name script_name

declare -a TASKS=(
 "food-101_processed run_baselines_finetune.sh"
 "caltech-101_processed run_baselines_finetune.sh"
 "cub-200-2011_processed run_baselines_finetune.sh"
 "artbench-10_processed run_baselines_finetune.sh"
 "ffhq256 run_baselines_finetune.sh"
)

# ========== EXECUTION LOOP ==========
for TASK in "${TASKS[@]}"; do
  read -r DATASET SCRIPT <<< "$TASK"

  echo "=============================================="
  echo "Running $SCRIPT on dataset $DATASET"
  echo "Server: $SERVER | CUDA Devices: $CUDA_DEVICES"
  echo "----------------------------------------------"
  
  CMD="scripts/$SCRIPT \
    --dataset "$DATASET" \
    --server "$SERVER" \
    --cuda_devices "$CUDA_DEVICES" \
    --experiment_prename "$EXPERIMENT_PRENAME" \
    --model_name \"DiT-XL/2\" \
    --difffit \"1\""

  eval "bash $CMD"

  echo "✅ Finished $SCRIPT on $DATASET"
  echo "=============================================="
done