#!/bin/bash

# ========== GLOBAL CONFIGURATION ==========
SERVER="bool"
CUDA_DEVICES="2,3"
SCRIPT="run_ours.sh"

 
declare -a TASKS=(
  "ffhq256"
)

# ========== Define per-task (latestart, mghigh, experiment_prename) triples ==========
declare -A PAIR_MAP
PAIR_MAP["ffhq256"]="1000,1,ablation_latestart 2000,1,ablation_latestart 3000,1,ablation_latestart 4000,1,ablation_latestart 5000,1,ablation_latestart 6000,1,ablation_latestart 7000,1,ablation_latestart"

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
