#!/bin/bash

# ========== GLOBAL CONFIGURATION ==========
SERVER="taylor"
CUDA_DEVICES="0,1"
SCRIPT="run_ours.sh"
EXPERIMENT_PRENAME="DiT_inception_ours_DiffFit"

declare -a TASKS=(
  "stanford-cars_processed"
  "food-101_processed"
  "caltech-101_processed"
  "cub-200-2011_processed"
  "ffhq256"
)

#   "artbench-10_processed"

# ========== Define per-task (latestart, mghigh, experiment_prename) triples ==========
declare -A PAIR_MAP
# Format: "latestart,mghigh latestart,mghigh ..."
# 3000,0.8,ablation_best_latestart_mg cub-200-2011 -> Sampling is done but it NEEDS FID calculation!

# Inception:
PAIR_MAP["stanford-cars_processed"]="7000,0.6" # 7000 first one
PAIR_MAP["caltech-101_processed"]="6000,0.4"
PAIR_MAP["food-101_processed"]="7000,0.5"
PAIR_MAP["artbench-10_processed"]="12000,1"
PAIR_MAP["cub-200-2011_processed"]="6000,0.7"
PAIR_MAP["ffhq256"]="8000,0.5"

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
      --model_name \"DiT-XL/2\" \
      --difffit \"1\""

    eval "bash $CMD"

    echo "✅ Finished $SCRIPT on $DATASET | latestart: $LATESTART | mghigh: $MGHIGH | prename: $EXPERIMENT_PRENAME"
    echo "=============================================="
  done
done
