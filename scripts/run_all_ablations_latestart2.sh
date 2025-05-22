#!/bin/bash

# ========== GLOBAL CONFIGURATION ==========
SERVER="taylor"
CUDA_DEVICES="2,3"
SCRIPT="run_ours.sh"

 
#   "stanford-cars_processed" "cub-200-2011_processed" "artbench-10_processed" "food-101_processed" "caltech-101_processed"
declare -a TASKS=(
  "ffhq256"
  "cub-200-2011_processed"
  "stanford-cars_processed"
)

# ========== Define per-task (latestart, mghigh, experiment_prename) triples ==========
declare -A PAIR_MAP
# Format: "latestart,mghigh latestart,mghigh ..."
# 3000,0.8,ablation_latestart cub-200-2011 -> Sampling is done but it NEEDS FID calculation!
PAIR_MAP["cub-200-2011_processed"]="8000,0.8,ablation_latestart 10000,0.8,ablation_latestart 12000,0.8,ablation_latestart 14000,0.8,ablation_latestart 16000,0.8,ablation_latestart 18000,0.8,ablation_latestart 20000,0.8,ablation_latestart 22000,0.8,ablation_latestart"
PAIR_MAP["stanford-cars_processed"]="8000,0.6,ablation_latestart 10000,0.6,ablation_latestart 12000,0.6,ablation_latestart 14000,0.6,ablation_latestart 16000,0.6,ablation_latestart 18000,0.6,ablation_latestart 20000,0.6,ablation_latestart 22000,0.6,ablation_latestart"
PAIR_MAP["food-101_processed"]="8000,0.8,ablation_latestart 10000,0.8,ablation_latestart 12000,0.8,ablation_latestart 14000,0.8,ablation_latestart 16000,0.8,ablation_latestart 18000,0.8,ablation_latestart 20000,0.8,ablation_latestart 22000,0.8,ablation_latestart"
PAIR_MAP["artbench-10_processed"]="8000,0.8,ablation_latestart 10000,0.8,ablation_latestart 12000,0.8,ablation_latestart 14000,0.8,ablation_latestart 16000,0.8,ablation_latestart 18000,0.8,ablation_latestart 20000,0.8,ablation_latestart 22000,0.8,ablation_latestart"
PAIR_MAP["caltech-101_processed"]="8000,0.8,ablation_latestart 10000,0.8,ablation_latestart 12000,0.8,ablation_latestart 14000,0.8,ablation_latestart 16000,0.8,ablation_latestart 18000,0.8,ablation_latestart 20000,0.8,ablation_latestart 22000,0.8,ablation_latestart"
PAIR_MAP["ffhq256"]="8000,0.5,ablation_latestart 10000,0.5,ablation_latestart 12000,0.5,ablation_latestart 14000,0.5,ablation_latestart 16000,0.5,ablation_latestart 18000,0.5,ablation_latestart 20000,0.5,ablation_latestart 22000,0.5,ablation_latestart 2000,0.5,ablation_latestart 4000,0.5,ablation_latestart 6000,0.5,ablation_latestart"


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
