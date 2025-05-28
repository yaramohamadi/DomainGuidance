#!/bin/bash

# ========== GLOBAL CONFIGURATION ==========
SERVER="bool"
CUDA_DEVICES="1,2"
SCRIPT="run_baseline_mg.sh"

 
#   "stanford-cars_processed" "cub-200-2011_processed" "artbench-10_processed" "food-101_processed" "caltech-101_processed"
declare -a TASKS=(
  "artbench-10_processed"
)

# ========== Define per-task (latestart, mghigh, experiment_prename) triples ==========
declare -A PAIR_MAP
# Format: "latestart,mghigh latestart,mghigh ..."
PAIR_MAP["artbench-10_processed"]="1.1,ablation_guidance_strengths 1.2,ablation_guidance_strengths 1.3,ablation_guidance_strengths 1.4,ablation_guidance_strengths 1.5,ablation_guidance_strengths 1.6,ablation_guidance_strengths 1.7,ablation_guidance_strengths 1.8,ablation_guidance_strengths 1.9,ablation_guidance_strengths 2.0,ablation_guidance_strengths 2.1,ablation_guidance_strengths 2.2,ablation_guidance_strengths 2.3,ablation_guidance_strengths 2.4,ablation_guidance_strengths 2.5,ablation_guidance_strengths 2.6,ablation_guidance_strengths 2.7,ablation_guidance_strengths 2.8,ablation_guidance_strengths 2.9,ablation_guidance_strengths 3.0,ablation_guidance_strengths"

# ========== EXECUTION LOOP ==========
for DATASET in "${TASKS[@]}"; do
  PAIRS=(${PAIR_MAP["$DATASET"]})
  for PAIR in "${PAIRS[@]}"; do
    IFS=',' read -r W_TRAIN_CG EXPERIMENT_PRENAME <<< "$PAIR"

    echo "=============================================="
    echo "Running $SCRIPT on $DATASET | w_train_cg: $W_TRAIN_CG | prename: $EXPERIMENT_PRENAME"
    echo "Server: $SERVER | CUDA Devices: $CUDA_DEVICES"
    echo "----------------------------------------------"

    CMD="scripts/$SCRIPT \
      --dataset \"$DATASET\" \
      --server \"$SERVER\" \
      --cuda_devices \"$CUDA_DEVICES\" \
      --experiment_prename \"$EXPERIMENT_PRENAME\" \
      --wtraincg \"$W_TRAIN_CG\""

    if [[ "$SERVER" == "computecanada" ]]; then
      eval "JOB_NAME=$EXPERIMENT_PRENAME sbatch $CMD"
    else
     eval "bash $CMD"
    fi

    echo "✅ Finished $SCRIPT on $DATASET | w_train_cg: $W_TRAIN_CG | prename: $EXPERIMENT_PRENAME"
    echo "=============================================="
  done
done
