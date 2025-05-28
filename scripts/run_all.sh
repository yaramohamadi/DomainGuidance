#!/bin/bash

# ========== GLOBAL CONFIGURATION ==========
SERVER="bool"
CUDA_DEVICES="1"
EXPERIMENT_PRENAME="All_DiT"
MODEL="DiT-XL/2" # DiT-XL/2 or SiT-XL/2

# ========== DATASET TO SCRIPT MAPPING ==========
# Format: dataset_name script_name

#  "stanford-cars_processed run_ours.sh"
#  "stanford-cars_processed run_baseline_mg.sh"
#  "stanford-cars_processed run_baselines_finetune.sh"
#  "food-101_processed run_ours.sh"
#  "food-101_processed run_baseline_mg.sh"
#  "food-101_processed run_baselines_finetune.sh"
#  "df-20m_processed run_ours.sh"
#  "df-20m_processed run_baseline_mg.sh"
#  "df-20m_processed run_baselines_finetune.sh"
#  "artbench-10_processed run_ours.sh"
#  "artbench-10_processed run_baseline_mg.sh"
#  "artbench-10_processed run_baselines_finetune.sh"
#  "caltech-101_processed run_ours.sh"
#  "caltech-101_processed run_baseline_mg.sh"
#  "caltech-101_processed run_baselines_finetune.sh"
#  "cub-200-2011_processed run_ours.sh"
#  "cub-200-2011_processed run_baseline_mg.sh"
#  "cub-200-2011_processed run_baselines_finetune.sh"
#  "cub-200-2011_processed run_ours.sh"
#  "cub-200-2011_processed run_baseline_mg.sh"
#  "cub-200-2011_processed run_baselines_finetune.sh"
#  "ffhq256 run_ours.sh"
#  "ffhq256 run_baseline_mg.sh"
#  "ffhq256 run_baselines_finetune.sh"


declare -a TASKS=(
  "caltech-101_processed run_ours.sh"
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