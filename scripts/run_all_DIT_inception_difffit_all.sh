#!/bin/bash

# ========== GLOBAL CONFIGURATION ==========
SERVER="computecanada"
CUDA_DEVICES="0,1"
EXPERIMENT_PRENAME="DiT_inception_all_DiffFit"

# ========== DATASET TO SCRIPT MAPPING ==========
# Format: dataset_name script_name
#  "caltech-101_processed run_baselines_finetune.sh"
declare -a TASKS=(
 "artbench-10_processed run_baselines_finetune.sh"
 "ffhq256 run_baselines_finetune.sh"
 "caltech-101_processed run_baselines_finetune.sh"
 "food-101_processed run_baselines_finetune.sh"
 "stanford-cars_processed run_baselines_finetune.sh"

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

  if [[ "$SERVER" == "computecanada" ]]; then
      eval "JOB_NAME=$EXPERIMENT_PRENAME sbatch $CMD"
  else
    eval "bash $CMD"
  fi

  echo "✅ Finished $SCRIPT on $DATASET"
  echo "=============================================="
done