#!/bin/bash
# set -e  # Exit on any error

# ====================== CONFIGURATION ======================

# Define CUDA devices here
CUDA_DEVICES="0,1"
EXPERIMENT_NAME="baseline_mgfinetune1_5_CUTOFF"
DATASET="stanford-cars_processed"  # Options: caltech, birds, etc.
SERVER="taylor"  # Options: taylor, bool, computecanada

# Load all logic
source scripts/config.sh
resolve_server_paths
resolve_dataset_config

W_TRAIN_CG=1.5
USE_GUIDANCE_CUTOFF=1

# Define any additional specific parameters here

train_model() {
    log_and_run "Training model..." \
    env CUDA_VISIBLE_DEVICES=$CUDA_DEVICES torchrun --nproc_per_node=$NPROC_PER_NODE train_MG.py \
        --data-path "$REAL_DATA_DIR" \
        --results-dir $RESULTS_DIR \
        --model $MODEL \
        --image-size $IMAGE_SIZE \
        --num-classes $NUM_CLASSES \
        --total-steps $TOTAL_STEPS \
        --log-every $LOG_EVERY \
        --ckpt-every $CKPT_EVERY \
        --global-batch-size $BATCH_SIZE \
        --vae $VAE \
        --num-workers $NUM_WORKERS \
        --w-cg $W_TRAIN_CG \
        --guidance-cutoff $USE_GUIDANCE_CUTOFF
}

run_sampling() {
    log_and_run "Sampling images..." \
    env CUDA_VISIBLE_DEVICES=$CUDA_DEVICES torchrun --nproc_per_node=$NPROC_PER_NODE sample_ddp.py \
        --model $MODEL \
        --vae $VAE \
        --sample-dir "$GENERATED_DIR" \
        --ckpt "$CHECKPOINT_DIR" \
        --per-proc-batch-size $BATCH_SIZE \
        --num-fid-samples $NSAMPLE \
        --image-size $IMAGE_SIZE \
        --num-classes $NUM_CLASSES \
        --cfg-scale $CFG_SCALE \
        --num-sampling-steps $NUM_SAMPLE_STEPS
}

calculate_fid() {
    log_and_run "Calculating FID..." \
    env CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m dgm_eval "$REAL_DATA_DIR" "$GENERATED_DIR" \
        --model inception \
        --device "$FID_DEVICE" \
        --nsample $NSAMPLE \
        --clean_resize \
        --metrics fd prdc \
        --save \
        --output_dir $RESULTS_FILE

    log_and_run "Calculating FID DINO..." \
    env CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m dgm_eval "$REAL_DATA_DIR" "$GENERATED_DIR" \
        --model dinov2 \
        --device "$FID_DEVICE" \
        --nsample $NSAMPLE \
        --clean_resize \
        --metrics fd prdc \
        --save \
        --output_dir $RESULTS_FILE
}

cleanup_dataset() {
    echo ">>> Cleaning up dataset..."
    rm -rf "$DATA_TARGET_DIR"
    echo ">>> Dataset removed."
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
