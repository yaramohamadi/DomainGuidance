#!/bin/bash

# ========== GLOBAL CONFIGURATION ==========
SERVER="computecanada"
CUDA_DEVICES="0,1"
SCRIPT="run_ours.sh"
EXPERIMENT_PRENAME="ablation_mghigh"
# EXPERIMENT_PRENAME="ablation_latestart"

#     "stanford-cars_processed" "cub-200-2011_processed"
declare -a TASKS=(
  "artbench-10_processed"
  "food-101_processed"
  "caltech-101_processed"
)

# ========== Define per-task (latestart, mghigh) pairs ==========
declare -A PAIR_MAP
# Format: "latestart,mghigh latestart,mghigh ..."
# PAIR_MAP["stanford-cars_processed"]="0,0.5 0,0.4 6000,1 7000 1"
# PAIR_MAP["caltech-101_processed"]="0,0.4 0,0.5, 0,0.6 0,0.7, 0,0.8 0,0.9 0,1.0"
# PAIR_MAP["caltech-101_processed"]="0,0.4 0,0.5, 0,0.6 0,0.7, 0,0.8 0,0.9 0,1.0"
PAIR_MAP["artbench-10_processed"]="1000,1.0,ablation_latestart 2000,1.0,ablation_latestart 3000,1.0,ablation_latestart 4000,1.0,ablation_latestart 5000,1.0,ablation_latestart 6000,1.0,ablation_latestart 7000,1.0,ablation_latestart"
PAIR_MAP["food-101_processed"]="1000,1.0,ablation_latestart 2000,1.0,ablation_latestart 3000,1.0,ablation_latestart 4000,1.0,ablation_latestart 5000,1.0,ablation_latestart 6000,1.0,ablation_latestart 7000,1.0,ablation_latestart"
PAIR_MAP["caltech-101_processed"]="1000,1.0,ablation_latestart 2000,1.0,ablation_latestart 3000,1.0,ablation_latestart 4000,1.0,ablation_latestart 5000,1.0,ablation_latestart 6000,1.0,ablation_latestart 7000,1.0,ablation_latestart"


# ========== EXECUTION LOOP ==========
for DATASET in "${TASKS[@]}"; do
  PAIRS=(${PAIR_MAP["$DATASET"]})
  echo "$PAIRS"
  for PAIR in "${PAIRS[@]}"; do
    LATESTART="${PAIR%%,*}"  # extract before comma
    MGHIGH="${PAIR##*,}"     # extract after comma
    

    echo "=============================================="
    echo "Running $SCRIPT on $DATASET | latestart: $LATESTART | mghigh: $MGHIGH"
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

    echo "✅ Finished $SCRIPT on $DATASET | latestart: $LATESTART | mghigh: $MGHIGH"
    echo "=============================================="
  done
done
