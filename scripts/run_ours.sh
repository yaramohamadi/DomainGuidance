#!/bin/bash
# set -e  # Exit on any error

# ====================== DEFAULT CONFIGURATION ======================

CUDA_DEVICES="0,1"
DATASET="cub-200-2011_processed"
SERVER="taylor"
EXPERIMENT_PRENAME=""

# ====================== ARGUMENT PARSING ======================

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --cuda_devices) CUDA_DEVICES="$2"; shift ;;
    --dataset) DATASET="$2"; shift ;;
    --server) SERVER="$2"; shift ;;
    --experiment_prename) EXPERIMENT_PRENAME="$2"; shift ;;
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
  shift
done

EXPERIMENT_NAME="$EXPERIMENT_PRENAME/dogfinetune_nodropout"

# Load all logic
source scripts/config.sh
resolve_server_paths
resolve_dataset_config

W_TRAIN_DOG=1.5
USE_GUIDANCE_CUTOFF=0
DROPOUT_RATIO=0.0

# Define any additional specific parameters here

train_model() {
    log_and_run "Training model..." \
    env CUDA_VISIBLE_DEVICES=$CUDA_DEVICES MASTER_PORT=$PORT torchrun --nproc_per_node=$NPROC_PER_NODE train_OURS.py \
        --data-path "$REAL_DATA_DIR" \
        --results-dir "$RESULTS_DIR" \
        --model "$MODEL" \
        --image-size "$IMAGE_SIZE" \
        --num-classes "$NUM_CLASSES" \
        --total-steps "$TOTAL_STEPS" \
        --log-every "$LOG_EVERY" \
        --ckpt-every "$CKPT_EVERY" \
        --global-batch-size "$BATCH_SIZE" \
        --vae "$VAE" \
        --num-workers "$NUM_WORKERS" \
        --w-dog "$W_TRAIN_DOG" \
        --guidance-cutoff "$USE_GUIDANCE_CUTOFF" \
        --dropout-ratio "$DROPOUT_RATIO" 
}

run_sampling() {
    log_and_run "Sampling images..." \
    env CUDA_VISIBLE_DEVICES=$CUDA_DEVICES MASTER_PORT=$PORT torchrun --nproc_per_node=$NPROC_PER_NODE sample_ddp.py \
        --model "$MODEL" \
        --vae "$VAE" \
        --sample-dir "$GENERATED_DIR" \
        --ckpt "$CHECKPOINT_DIR" \
        --per-proc-batch-size "$BATCH_SIZE" \
        --num-fid-samples "$NSAMPLE" \
        --image-size "$IMAGE_SIZE" \
        --num-classes "$NUM_CLASSES" \
        --cfg-scale "$CFG_SCALE" \
        --num-sampling-steps "$NUM_SAMPLE_STEPS" \
        --dropout-ratio "$DROPOUT_RATIO" 
}

calculate_fid() {
    log_and_run "Calculating FID..." \
    env CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m dgm_eval "$REAL_DATA_DIR" "$GENERATED_DIR" \
        --model inception \
        --device "$FID_DEVICE" \
        --nsample "$NSAMPLE" \
        --clean_resize \
        --metrics fd prdc \
        --save \
        --output_dir "$RESULTS_FILE"

    log_and_run "Calculating FID DINO..." \
    env CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m dgm_eval "$REAL_DATA_DIR" "$GENERATED_DIR" \
        --model dinov2 \
        --device "$FID_DEVICE" \
        --nsample "$NSAMPLE" \
        --clean_resize \
        --metrics fd prdc \
        --save \
        --output_dir "$RESULTS_FILE"
}

# ====================== MAIN ======================
echo ">>> Logging to: $LOG_FILE"
rm -f "$LOG_FILE"
mkdir -p "$(dirname "$LOG_FILE")"

create_environment
prepare_dataset
train_model
run_sampling
calculate_fid
cleanup_dataset

echo ">>> All tasks completed successfully!"
