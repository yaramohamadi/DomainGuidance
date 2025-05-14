#!/bin/bash

# ========== GLOBAL CONFIGURATION ==========
SERVER="taylor"
CUDA_DEVICES="0,1"
EXPERIMENT_PRENAME="ablation_train_traverse"

# ========== DATASET TO SCRIPT MAPPING ==========
# Format: dataset_name script_name

#  "cub-200-2011_processed run_ours.sh"
#  "cub-200-2011_processed run_baseline_mg.sh"
#  "cub-200-2011_processed run_baselines_finetune.sh"
#  "cub-200-2011_processed run_ours.sh"
#  "cub-200-2011_processed run_baseline_mg.sh"
#  "cub-200-2011_processed run_baselines_finetune.sh"

declare -a TASKS=(
  "cub-200-2011_processed run_ours.sh"
  "cub-200-2011_processed run_baseline_mg.sh"
  "cub-200-2011_processed run_baselines_finetune.sh"
)

LATESTART=3000
MGHIGH=0.75

# ========== EXECUTION LOOP ==========
for TASK in "${TASKS[@]}"; do
  read -r DATASET SCRIPT <<< "$TASK"

  echo "=============================================="
  echo "Running $SCRIPT on dataset $DATASET"
  echo "Server: $SERVER | CUDA Devices: $CUDA_DEVICES"
  echo "----------------------------------------------"

  if [[ "$SCRIPT" == "run_ours.sh" ]]; then
    echo "Running special config for OURS with best hyperparameters"
      bash "scripts/$SCRIPT" \
      --dataset "$DATASET" \
      --server "$SERVER" \
      --cuda_devices "$CUDA_DEVICES" \
      --experiment_prename "$EXPERIMENT_PRENAME" \
      --latestart "$LATESTART" \
      --mghigh "$MGHIGH"

  else
    bash "scripts/$SCRIPT" \
      --dataset "$DATASET" \
      --server "$SERVER" \
      --cuda_devices "$CUDA_DEVICES" \
      --experiment_prename "$EXPERIMENT_PRENAME"
  fi

  echo "✅ Finished $SCRIPT on $DATASET"
  echo "=============================================="
done