#!/bin/bash

# ========== GLOBAL CONFIGURATION ==========
SERVER="taylor"
CUDA_DEVICES="2,3"
SCRIPT="run_ours.sh"

# "food-101_processed"
# "artbench-10_processed"
# "caltech-101_processed"
# "cub-200-2011_processed"
# "ffhq256"

declare -a TASKS=(
  "cub-200-2011_processed"
)

# ========== Define per-task (latestart, mghigh, experiment_prename) triples ==========
declare -A PAIR_MAP

PAIR_MAP["stanford-cars_processed"]="8000,1,1,3,0,50in1to1.25" 
PAIR_MAP["caltech-101_processed"]="8000,0.7,1,3,0,50in1to1.25"
PAIR_MAP["food-101_processed"]="8000,1,1,3,0,50in1to1.25"
PAIR_MAP["artbench-10_processed"]="8000,0.8,1,3,0,50in1to1.25"
PAIR_MAP["cub-200-2011_processed"]="6000,1,1,3,0,50in1to1.25"
PAIR_MAP["ffhq256"]="10000,0.6,1,3,0,50in1to1.25"

# ========== EXECUTION LOOP ==========
for DATASET in "${TASKS[@]}"; do
  PAIRS=(${PAIR_MAP["$DATASET"]})
  for PAIR in "${PAIRS[@]}"; do
    IFS=',' read -r LATESTART MGHIGH W_MIN W_MAX SAMPLE_GUIDANCE CONTROL_DISTRIBUTION <<< "$PAIR"

    echo "=============================================="
    echo "Running $SCRIPT on $DATASET | w_min: $W_MIN | w_max: $W_MAX | control_distribution: $CONTROL_DISTRIBUTION | sample_guidance $SAMPLE_GUIDANCE | latestart: $LATESTART | mghigh: $MGHIGH | prename: $EXPERIMENT_PRENAME"
    echo "Server: $SERVER | CUDA Devices: $CUDA_DEVICES"
    echo "----------------------------------------------"

    EXPERIMENT_PRENAME="DiT_dino_ours/control_normalizing_exponential_cutofflatestart/$CONTROL_DISTRIBUTION"

    CMD="scripts/$SCRIPT \
      --dataset \"$DATASET\" \
      --server \"$SERVER\" \
      --cuda_devices \"$CUDA_DEVICES\" \
      --experiment_prename \"$EXPERIMENT_PRENAME\" \
      --latestart \"$LATESTART\" \
      --mghigh \"$MGHIGH\" \
      --model_name \"DiT-XL/2\" \
      --guidance_control \"1\" \
      --w_max \"$W_MAX\" \
      --w_min \"$W_MIN\" \
      --sample_guidance \"$SAMPLE_GUIDANCE\" \
      --control_distribution \"$CONTROL_DISTRIBUTION\""

    if [[ "$SERVER" == "computecanada" ]]; then
      eval "JOB_NAME=$EXPERIMENT_PRENAME sbatch $CMD"
    else
      eval "bash $CMD"
    fi

    echo "✅ Finished $SCRIPT on $DATASET | latestart: $LATESTART | mghigh: $MGHIGH | prename: $EXPERIMENT_PRENAME"
    echo "=============================================="
  done
done
