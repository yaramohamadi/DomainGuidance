#!/bin/bash

# ========== GLOBAL CONFIGURATION ==========
SERVER="bool"
CUDA_DEVICES="2,3"
SCRIPT="run_ours.sh"
EXPERIMENT_PRENAME="samplesteps_DiT_inception_ours"

# ========== SAMPLE STEPS ==========
SAMPLE_STEPS_LIST=(100) # 25

#  "artbench-10_processed"
declare -a TASKS=(
 "food-101_processed"
)

# ========== Define per-task (latestart, mghigh) ==========
declare -A PAIR_MAP

PAIR_MAP["stanford-cars_processed"]="7000,0.6"
PAIR_MAP["caltech-101_processed"]="6000,0.4"
PAIR_MAP["food-101_processed"]="7000,0.5"
PAIR_MAP["artbench-10_processed"]="12000,1"
PAIR_MAP["cub-200-2011_processed"]="6000,0.7"
PAIR_MAP["ffhq256"]="8000,0.5"

# ========== EXECUTION LOOP ==========
for NUM_SAMPLE_STEPS in "${SAMPLE_STEPS_LIST[@]}"; do
  for DATASET in "${TASKS[@]}"; do
    PAIRS=(${PAIR_MAP["$DATASET"]})
    for PAIR in "${PAIRS[@]}"; do
      IFS=',' read -r LATESTART MGHIGH <<< "$PAIR"

      echo "=============================================="
      echo "Running $SCRIPT on $DATASET | latestart: $LATESTART | mghigh: $MGHIGH | sample_steps: $NUM_SAMPLE_STEPS"
      echo "Server: $SERVER | CUDA Devices: $CUDA_DEVICES"
      echo "----------------------------------------------"

      CMD="scripts/$SCRIPT \
        --dataset \"$DATASET\" \
        --server \"$SERVER\" \
        --cuda_devices \"$CUDA_DEVICES\" \
        --experiment_prename \"$EXPERIMENT_PRENAME\" \
        --latestart \"$LATESTART\" \
        --mghigh \"$MGHIGH\" \
        --model_name \"DiT-XL/2\" \
        --num_sample_steps \"$NUM_SAMPLE_STEPS\""

      if [[ "$SERVER" == "computecanada" ]]; then
        eval "JOB_NAME=$EXPERIMENT_PRENAME sbatch $CMD"
      else
        eval "bash $CMD"
      fi

      echo "✅ Finished $SCRIPT on $DATASET | sample_steps: $NUM_SAMPLE_STEPS"
      echo "=============================================="
    done
  done
done
