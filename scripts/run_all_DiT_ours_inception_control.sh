#!/bin/bash

# ========== GLOBAL CONFIGURATION ==========
SERVER="taylor"
CUDA_DEVICES="0,1"
SCRIPT="run_ours.sh"
EXPERIMENT_PRENAME="SiT_inception_ours"

# "food-101_processed"
# "artbench-10_processed"
# "caltech-101_processed"
# "cub-200-2011_processed"
# "ffhq256"

declare -a TASKS=(
  "stanford-cars_processed"
)

# ========== Define per-task (latestart, mghigh, experiment_prename) triples ==========
declare -A PAIR_MAP

# Inception:
PAIR_MAP["stanford-cars_processed"]="0,1,1 0,1,2 0,1,3 0,1,4" # "7000,0.6"

# ========== EXECUTION LOOP ==========
for DATASET in "${TASKS[@]}"; do
  PAIRS=(${PAIR_MAP["$DATASET"]})
  for PAIR in "${PAIRS[@]}"; do
    IFS=',' read -r LATESTART MGHIGH W_MAX <<< "$PAIR"

    echo "=============================================="
    echo "Running $SCRIPT on $DATASET | w_max: $W_MAX | latestart: $LATESTART | mghigh: $MGHIGH | prename: $EXPERIMENT_PRENAME"
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
      --guidance_control \"1\" \
      --w_max \"$W_MAX\""

    eval "bash $CMD"

    echo "✅ Finished $SCRIPT on $DATASET | latestart: $LATESTART | mghigh: $MGHIGH | prename: $EXPERIMENT_PRENAME"
    echo "=============================================="
  done
done
