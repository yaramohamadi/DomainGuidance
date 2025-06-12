#!/bin/bash

# ========== GLOBAL CONFIGURATION ==========
SERVER="taylor"
CUDA_DEVICES="0,1"
EXPERIMENT_PRENAME="All_unconditional"

# ========== DATASET TO SCRIPT MAPPING ==========
# Format: dataset_name script_name

declare -a TASKS=(
 "stanford-cars_processed run_ours.sh"
 "caltech-101_processed run_ours.sh"
 "food-101_processed run_ours.sh"
 "df-20m_processed run_ours.sh"
 "artbench-10_processed run_ours.sh"
 "ffhq256 run_ours.sh"
 "cub-200-2011_processed run_ours.sh"
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
    --experiment_prename "$EXPERIMENT_PRENAME"

  echo "✅ Finished $SCRIPT on $DATASET"
  echo "=============================================="
done