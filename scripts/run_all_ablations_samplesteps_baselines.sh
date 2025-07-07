#!/bin/bash

# ========== GLOBAL CONFIGURATION ==========
SERVER="bool"
CUDA_DEVICES="2,3"
EXPERIMENT_PRENAME="samplesteps_DiT_inception_MG"

# ========== NUM SAMPLE STEPS ==========
SAMPLE_STEPS_LIST=(25 100)

# ========== TASK DEFINITIONS ==========
declare -a TASKS=(
 "food-101_processed run_baseline_mg.sh"
 "artbench-10_processed run_baseline_mg.sh"
)

# ========== EXECUTION LOOP ==========
for TASK in "${TASKS[@]}"; do
  read -r DATASET SCRIPT <<< "$TASK"

  for NUM_SAMPLE_STEPS in "${SAMPLE_STEPS_LIST[@]}"; do
    echo "=============================================="
    echo "Running $SCRIPT on $DATASET with $NUM_SAMPLE_STEPS sample steps"
    echo "Server: $SERVER | CUDA Devices: $CUDA_DEVICES"
    echo "----------------------------------------------"

    bash "scripts/$SCRIPT" \
      --dataset "$DATASET" \
      --server "$SERVER" \
      --cuda_devices "$CUDA_DEVICES" \
      --experiment_prename "$EXPERIMENT_PRENAME" \
      --model_name "DiT-XL/2" \
      --num_sample_steps "$NUM_SAMPLE_STEPS"

    echo "✅ Finished $SCRIPT on $DATASET with $NUM_SAMPLE_STEPS steps"
    echo "=============================================="
  done
done
