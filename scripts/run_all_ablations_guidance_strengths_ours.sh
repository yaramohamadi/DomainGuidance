#!/bin/bash

# ========== GLOBAL CONFIGURATION ==========
SERVER="computecanada"
CUDA_DEVICES="0,1"
SCRIPT="run_ours.sh"

 
#   "stanford-cars_processed" "cub-200-2011_processed" "artbench-10_processed" "food-101_processed" "caltech-101_processed"
declare -a TASKS=(
  "artbench-10_processed"
  "food-101_processed"
)

# ========== Define per-task (latestart, mghigh, experiment_prename) triples ==========
declare -A PAIR_MAP
# Format: "latestart,mghigh latestart,mghigh ..."
PAIR_MAP["artbench-10_processed"]="1.2,12000,1.0,ablation_guidance_strengths 2.4,12000,1.0,ablation_guidance_strengths 2.5,12000,1.0,ablation_guidance_strengths 2.7,12000,1.0,ablation_guidance_strengths 2.9,12000,1.0,ablation_guidance_strengths"

PAIR_MAP["food-101_processed"]="1.1,12000,1.0,ablation_guidance_strengths 1.2,12000,1.0,ablation_guidance_strengths 1.3,12000,1.0,ablation_guidance_strengths 1.4,12000,1.0,ablation_guidance_strengths 1.5,12000,1.0,ablation_guidance_strengths 1.6,12000,1.0,ablation_guidance_strengths 1.7,12000,1.0,ablation_guidance_strengths 1.8,12000,1.0,ablation_guidance_strengths 1.9,12000,1.0,ablation_guidance_strengths 2.0,12000,1.0,ablation_guidance_strengths 2.1,12000,1.0,ablation_guidance_strengths 2.2,12000,1.0,ablation_guidance_strengths 2.3,12000,1.0,ablation_guidance_strengths 2.4,12000,1.0,ablation_guidance_strengths 2.5,12000,1.0,ablation_guidance_strengths 2.6,12000,1.0,ablation_guidance_strengths 2.7,12000,1.0,ablation_guidance_strengths 2.8,12000,1.0,ablation_guidance_strengths 2.9,12000,1.0,ablation_guidance_strengths 3.0,12000,1.0,ablation_guidance_strengths"

# ========== EXECUTION LOOP ==========
for DATASET in "${TASKS[@]}"; do
  PAIRS=(${PAIR_MAP["$DATASET"]})
  for PAIR in "${PAIRS[@]}"; do
    IFS=',' read -r W_TRAIN_DOG LATESTART MGHIGH EXPERIMENT_PRENAME <<< "$PAIR"

    echo "=============================================="
    echo "Running $SCRIPT on $DATASET | w_train_dog: $W_TRAIN_DOG | latestart: $LATESTART | mghigh: $MGHIGH | prename: $EXPERIMENT_PRENAME"
    echo "Server: $SERVER | CUDA Devices: $CUDA_DEVICES"
    echo "----------------------------------------------"

    CMD="scripts/$SCRIPT \
      --dataset \"$DATASET\" \
      --server \"$SERVER\" \
      --cuda_devices \"$CUDA_DEVICES\" \
      --experiment_prename \"$EXPERIMENT_PRENAME\" \
      --latestart \"$LATESTART\" \
      --mghigh \"$MGHIGH\" \
      --wtraindog \"$W_TRAIN_DOG\""

    if [[ "$SERVER" == "computecanada" ]]; then
      eval "JOB_NAME=$EXPERIMENT_PRENAME sbatch $CMD"
    else
      eval "bash $CMD"
    fi
    
    echo "✅ Finished $SCRIPT on $DATASET | w_train_dog: $W_TRAIN_DOG | latestart: $LATESTART | mghigh: $MGHIGH | prename: $EXPERIMENT_PRENAME"
    echo "=============================================="
  done
done