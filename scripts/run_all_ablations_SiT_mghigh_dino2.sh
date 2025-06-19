#!/bin/bash

# ========== GLOBAL CONFIGURATION ==========
SERVER="taylor"
CUDA_DEVICES="2,3"
SCRIPT="run_ours.sh"


declare -a TASKS=(
  "food-101_processed"
  "artbench-10_processed"
  "caltech-101_processed"
)

# ========== Define per-task (latestart, mghigh, experiment_prename) triples ==========
declare -A PAIR_MAP
PAIR_MAP["cub-200-2011_processed"]="6000,0.4,mega_ablation_mg 6000,0.5,mega_ablation_mg 6000,0.6,mega_ablation_mg 6000,0.7,mega_ablation_mg 6000,0.8,mega_ablation_mg 6000,0.9,mega_ablation_mg"
PAIR_MAP["stanford-cars_processed"]="6000,0.4,mega_ablation_mg 6000,0.5,mega_ablation_mg 6000,0.6,mega_ablation_mg 6000,0.7,mega_ablation_mg 6000,0.8,mega_ablation_mg 6000,0.9,mega_ablation_mg"
PAIR_MAP["food-101_processed"]="6000,0.4,mega_ablation_mg 6000,0.5,mega_ablation_mg 6000,0.6,mega_ablation_mg 6000,0.7,mega_ablation_mg 6000,0.8,mega_ablation_mg 6000,0.9,mega_ablation_mg"
PAIR_MAP["artbench-10_processed"]="12000,0.4,mega_ablation_mg 12000,0.5,mega_ablation_mg 12000,0.6,mega_ablation_mg 12000,0.7,mega_ablation_mg 12000,0.8,mega_ablation_mg 12000,0.9,mega_ablation_mg"
PAIR_MAP["caltech-101_processed"]="10000,0.4,mega_ablation_mg 10000,0.5,mega_ablation_mg 10000,0.6,mega_ablation_mg 10000,0.7,mega_ablation_mg 10000,0.8,mega_ablation_mg 10000,0.9,mega_ablation_mg"
PAIR_MAP["ffhq256"]="12000,0.4,mega_ablation_mg 12000,0.5,mega_ablation_mg 12000,0.6,mega_ablation_mg 12000,0.7,mega_ablation_mg 12000,0.8,mega_ablation_mg 12000,0.9,mega_ablation_mg"

# ========== EXECUTION LOOP ==========
for DATASET in "${TASKS[@]}"; do
  PAIRS=(${PAIR_MAP["$DATASET"]})
  for PAIR in "${PAIRS[@]}"; do
    IFS=',' read -r LATESTART MGHIGH EXPERIMENT_PRENAME <<< "$PAIR"

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

    if [[ "$SERVER" == "computecanada" ]]; then
      eval "JOB_NAME=$EXPERIMENT_PRENAME sbatch $CMD"
    else
      eval "bash $CMD"
    fi

    echo "✅ Finished $SCRIPT on $DATASET | latestart: $LATESTART | mghigh: $MGHIGH | prename: $EXPERIMENT_PRENAME"
    echo "=============================================="
  done
done
