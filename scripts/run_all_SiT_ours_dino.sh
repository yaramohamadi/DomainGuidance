#!/bin/bash

# ========== GLOBAL CONFIGURATION ==========
SERVER="taylor"
CUDA_DEVICES="2,3"
SCRIPT="run_ours.sh"
EXPERIMENT_PRENAME="SiT_dino_ours"

# "food-101_processed"
# "artbench-10_processed"
# "caltech-101_processed"
# "cub-200-2011_processed"
# "ffhq256"

# "stanford-cars_processed"

declare -a TASKS=(
"food-101_processed"
"artbench-10_processed"
"caltech-101_processed"
"ffhq256"
)

# "cub-200-2011_processed"

# ========== Define per-task (latestart, mghigh, experiment_prename) triples ==========
declare -A PAIR_MAP

# DINO:
PAIR_MAP["stanford-cars_processed"]="8000,1" # 8000
PAIR_MAP["caltech-101_processed"]="8000,0.7"
PAIR_MAP["food-101_processed"]="8000,1"
PAIR_MAP["artbench-10_processed"]="8000,0.9"
PAIR_MAP["cub-200-2011_processed"]="6000,1"
PAIR_MAP["ffhq256"]="10000,0.6"

# ========== EXECUTION LOOP ==========
for DATASET in "${TASKS[@]}"; do
  PAIRS=(${PAIR_MAP["$DATASET"]})
  for PAIR in "${PAIRS[@]}"; do
    IFS=',' read -r LATESTART MGHIGH <<< "$PAIR"

    echo "=============================================="
    echo "Running $SCRIPT on $DATASET | latestart: $LATESTART | mghigh: $MGHIGH | prename: $EXPERIMENT_PRENAME"
    echo "Server: $SERVER | CUDA Devices: $CUDA_DEVICES"
    echo "----------------------------------------------"

    CMD="scripts/$SCRIPT \
      --dataset \"$DATASET\" \
      --server \"$SERVER\" \
      --cuda_devices \"$CUDA_DEVICES\" \
      --experiment_prename \"$EXPERIMENT_PRENAME\" \
      --latestart \"$LATESTART\" \
      --mghigh \"$MGHIGH\" \
      --model_name \"SiT-XL/2\""

    eval "bash $CMD"

    echo "✅ Finished $SCRIPT on $DATASET | latestart: $LATESTART | mghigh: $MGHIGH | prename: $EXPERIMENT_PRENAME"
    echo "=============================================="
  done
done
