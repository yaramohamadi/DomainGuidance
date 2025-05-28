#!/bin/bash

# ========== GLOBAL CONFIGURATION ==========
SERVER="taylor"
CUDA_DEVICES="0,1"
SCRIPT="run_ours.sh"

 
#   "stanford-cars_processed" "cub-200-2011_processed" "artbench-10_processed" "food-101_processed" "caltech-101_processed"
declare -a TASKS=(
  "food-101_processed"
  "caltech-101_processed"
  "stanford-cars_processed"
)

# ========== Define per-task (latestart, mghigh, experiment_prename) triples ==========
declare -A PAIR_MAP
# Format: "latestart,mghigh latestart,mghigh ..."
# 3000,0.8,ablation_best_latestart_mg cub-200-2011 -> Sampling is done but it NEEDS FID calculation!
PAIR_MAP["stanford-cars_processed"]="7000,0.1,ablation_best_latestart_mg 7000,0.2,ablation_best_latestart_mg 7000,0.3,ablation_best_latestart_mg 7000,0.4,ablation_best_latestart_mg 7000,0.5,ablation_best_latestart_mg 7000,0.6,ablation_best_latestart_mg 7000,0.7,ablation_best_latestart_mg 7000,0.8,ablation_best_latestart_mg 7000,0.9,ablation_best_latestart_mg 7000,1.0,ablation_best_latestart_mg"
PAIR_MAP["food-101_processed"]="7000,0.1,ablation_best_latestart_mg 7000,0.2,ablation_best_latestart_mg 7000,0.3,ablation_best_latestart_mg 7000,0.4,ablation_best_latestart_mg 7000,0.5,ablation_best_latestart_mg 7000,0.6,ablation_best_latestart_mg 7000,0.7,ablation_best_latestart_mg 7000,0.8,ablation_best_latestart_mg 7000,0.9,ablation_best_latestart_mg 7000,1.0,ablation_best_latestart_mg"
PAIR_MAP["caltech-101_processed"]="6000,0.1,ablation_best_latestart_mg 6000,0.2,ablation_best_latestart_mg 6000,0.3,ablation_best_latestart_mg 6000,0.4,ablation_best_latestart_mg 6000,0.5,ablation_best_latestart_mg 6000,0.6,ablation_best_latestart_mg 6000,0.7,ablation_best_latestart_mg 6000,0.8,ablation_best_latestart_mg 6000,0.9,ablation_best_latestart_mg 6000,1.0,ablation_best_latestart_mg"

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
      --mghigh \"$MGHIGH\""

    #if [[ "$SERVER" == "computecanada" ]]; then
    #  eval "JOB_NAME=$EXPERIMENT_PRENAME sbatch $CMD"
    #else
    eval "bash $CMD"
    #fi

    echo "✅ Finished $SCRIPT on $DATASET | latestart: $LATESTART | mghigh: $MGHIGH | prename: $EXPERIMENT_PRENAME"
    echo "=============================================="
  done
done
