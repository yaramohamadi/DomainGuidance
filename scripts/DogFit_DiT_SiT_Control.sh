#!/bin/bash

# ========== GLOBAL CONFIGURATION ==========
SERVER="bool"
CUDA_DEVICES="0,1"
SCRIPT="run_DogFit.sh"
MODEL_NAME="DiT-XL/2"  # Options: "DiT-XL/2" or "SiT-XL/2"
FOCUS_METRIC="FD_DINOV2"  # Options: "FID" or "FD_DINOV2"

declare -a TASKS=(
  "food-101_processed"
)

# ========== DEFINE CONFIGURATIONS ==========

# Format: latestart, mghigh, w_min, w_max, sample_guidance, control_distribution
declare -A CONFIG_FID
CONFIG_FID["food-101_processed"]="12000,0.5,0,50in1to1.25"

declare -A CONFIG_FD_DINOV2
CONFIG_FD_DINOV2["food-101_processed"]="12000,1,0,50in1to1.25"

# ========== EXECUTION LOOP ==========
for DATASET in "${TASKS[@]}"; do
  if [ "$FOCUS_METRIC" == "FID" ]; then
    CONFIG="${CONFIG_FID[$DATASET]}"
  elif [ "$FOCUS_METRIC" == "FD_DINOV2" ]; then
    CONFIG="${CONFIG_FD_DINOV2[$DATASET]}"
  else
    echo "Unsupported FOCUS_METRIC: $FOCUS_METRIC"
    exit 1
  fi

  IFS=',' read -r LATESTART MGHIGH W_MIN W_MAX SAMPLE_GUIDANCE CONTROL_DISTRIBUTION <<< "$CONFIG"

  EXPERIMENT_PRENAME="${MODEL_NAME%%/*}_${FOCUS_METRIC}/control_normalizing_exponential_cutofflatestart/$CONTROL_DISTRIBUTION"

  echo "=============================================="
  echo "Running $SCRIPT on $DATASET"
  echo "model: $MODEL_NAME | metric: $FOCUS_METRIC"
  echo "w_min: $W_MIN | w_max: $W_MAX | control_distribution: $CONTROL_DISTRIBUTION | sample_guidance: $SAMPLE_GUIDANCE"
  echo "latestart: $LATESTART | mghigh: $MGHIGH"
  echo "Server: $SERVER | CUDA Devices: $CUDA_DEVICES"
  echo "----------------------------------------------"

  bash scripts/$SCRIPT \
    --dataset "$DATASET" \
    --server "$SERVER" \
    --cuda_devices "$CUDA_DEVICES" \
    --experiment_prename "$EXPERIMENT_PRENAME" \
    --latestart "$LATESTART" \
    --mghigh "$MGHIGH" \
    --model_name "$MODEL_NAME" \
    --guidance_control "1" \
    --w_max "$W_MAX" \
    --w_min "$W_MIN" \
    --sample_guidance "$SAMPLE_GUIDANCE" \
    --control_distribution "$CONTROL_DISTRIBUTION"

  echo "Finished $SCRIPT on $DATASET"
  echo "=============================================="
done
