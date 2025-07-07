#!/bin/bash

# ========== GLOBAL CONFIGURATION ==========
SERVER="computecanada"
CUDA_DEVICES="0,1"
SCRIPT="run_ours_jose.sh"


declare -a TASKS=(
  "artbench-10_processed"
  "food-101_processed"
  "caltech-101_processed"
)

# ========== Define per-task (latestart, mghigh, experiment_prename) triples ==========
declare -A PAIR_MAP
# Format: "latestart,mghigh,prename latestart,mghigh,prename ..."
PAIR_MAP["cub-200-2011_processed"]="0,0.2,ablation_mghigh 0,0.4,ablation_mghigh 0,0.6,ablation_mghigh 0,0.8,ablation_mghigh 0,1.0,ablation_mghigh"
PAIR_MAP["stanford-cars_processed"]="0,0.2,ablation_mghigh 0,0.4,ablation_mghigh 0,0.6,ablation_mghigh 0,0.8,ablation_mghigh 0,1.0,ablation_mghigh"
PAIR_MAP["artbench-10_processed"]="0,0.2,ablation_mghigh 0,0.4,ablation_mghigh 0,0.6,ablation_mghigh 0,0.8,ablation_mghigh 0,1.0,ablation_mghigh"
PAIR_MAP["food-101_processed"]="0,0.2,ablation_mghigh 0,0.4,ablation_mghigh 0,0.6,ablation_mghigh 0,0.8,ablation_mghigh 0,1.0,ablation_mghigh"
PAIR_MAP["caltech-101_processed"]="0,0.2,ablation_mghigh 0,0.4,ablation_mghigh 0,0.6,ablation_mghigh 0,0.8,ablation_mghigh 0,1.0,ablation_mghigh"
PAIR_MAP["ffhq256"]="0,0.2,ablation_mghigh 0,0.4,ablation_mghigh 0,0.6,ablation_mghigh 0,0.8,ablation_mghigh 0,1.0,ablation_mghigh"

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

    if [[ "$SERVER" == "computecanada" ]]; then
      eval "JOB_NAME=$EXPERIMENT_PRENAME sbatch $CMD"
    else
      eval "bash $CMD"
    fi

    echo "✅ Finished $SCRIPT on $DATASET | latestart: $LATESTART | mghigh: $MGHIGH | prename: $EXPERIMENT_PRENAME"
    echo "=============================================="
  done
done
