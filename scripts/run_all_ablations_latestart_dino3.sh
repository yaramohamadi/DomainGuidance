#!/bin/bash

# ========== GLOBAL CONFIGURATION ==========
SERVER="bool"
CUDA_DEVICES="1,3"
SCRIPT="run_ours.sh"


declare -a TASKS=(
  "caltech-101_processed"
  "artbench-10_processed"
)

# ========== Define per-task (latestart, mghigh, experiment_prename) triples ==========
# 2000,0.8,mega_ablation_latestart 4000,0.8,mega_ablation_latestart 6000,0.8,mega_ablation_latestart 
declare -A PAIR_MAP
PAIR_MAP["cub-200-2011_processed"]="2000,1,mega_ablation_latestart 4000,1,mega_ablation_latestart 6000,1,mega_ablation_latestart 8000,1,mega_ablation_latestart 10000,1,mega_ablation_latestart 12000,1,mega_ablation_latestart 14000,1,mega_ablation_latestart 16000,1,mega_ablation_latestart 18000,1,mega_ablation_latestart 20000,1,mega_ablation_latestart 22000,1,mega_ablation_latestart"
PAIR_MAP["stanford-cars_processed"]="2000,1,mega_ablation_latestart 4000,1,mega_ablation_latestart 6000,1,mega_ablation_latestart 8000,1,mega_ablation_latestart 10000,1,mega_ablation_latestart 12000,1,mega_ablation_latestart 14000,1,mega_ablation_latestart 16000,1,mega_ablation_latestart 18000,1,mega_ablation_latestart 20000,1,mega_ablation_latestart 22000,1,mega_ablation_latestart"
PAIR_MAP["food-101_processed"]="2000,1,mega_ablation_latestart 4000,1,mega_ablation_latestart 6000,1,mega_ablation_latestart 8000,1,mega_ablation_latestart 10000,1,mega_ablation_latestart 12000,1,mega_ablation_latestart 14000,1,mega_ablation_latestart 16000,1,mega_ablation_latestart 18000,1,mega_ablation_latestart 20000,1,mega_ablation_latestart 22000,1,mega_ablation_latestart"
PAIR_MAP["artbench-10_processed"]="8000,0.8,mega_ablation_latestart 10000,0.8,mega_ablation_latestart 12000,0.8,mega_ablation_latestart 14000,0.8,mega_ablation_latestart 16000,0.8,mega_ablation_latestart 18000,0.8,mega_ablation_latestart 20000,0.8,mega_ablation_latestart 22000,0.8,mega_ablation_latestart"
PAIR_MAP["caltech-101_processed"]="2000,0.7,mega_ablation_latestart 4000,0.7,mega_ablation_latestart 6000,0.7,mega_ablation_latestart 8000,0.7,mega_ablation_latestart 10000,0.7,mega_ablation_latestart 12000,0.7,mega_ablation_latestart 14000,0.7,mega_ablation_latestart 16000,0.7,mega_ablation_latestart 18000,0.7,mega_ablation_latestart 20000,0.7,mega_ablation_latestart 22000,0.7,mega_ablation_latestart"
PAIR_MAP["ffhq256"]="2000,0.6,mega_ablation_latestart 4000,0.6,mega_ablation_latestart 6000,0.6,mega_ablation_latestart 8000,0.6,mega_ablation_latestart 10000,0.6,mega_ablation_latestart 12000,0.6,mega_ablation_latestart 14000,0.6,mega_ablation_latestart 16000,0.6,mega_ablation_latestart 18000,0.6,mega_ablation_latestart 20000,0.6,mega_ablation_latestart 22000,0.6,mega_ablation_latestart"

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

    # if [[ "$SERVER" == "computecanada" ]]; then
    #   eval "JOB_NAME=$EXPERIMENT_PRENAME sbatch $CMD"
    # else
    eval "bash $CMD"
    # fi

    echo "✅ Finished $SCRIPT on $DATASET | latestart: $LATESTART | mghigh: $MGHIGH | prename: $EXPERIMENT_PRENAME"
    echo "=============================================="
  done
done
