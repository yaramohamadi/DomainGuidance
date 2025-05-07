#!/bin/bash
# set -e  # Exit on any error

# ====================== CONFIGURATION ======================

CUDA_DEVICES="0,3"
EXPERIMENT_NAME="baselines_finetune"
DATASET="food-101_processed"  # Options: caltech, birds, etc.
SERVER="taylor"  # Options: taylor, bool, computecanada

# Load all logic
source scripts/config.sh
resolve_server_paths
resolve_dataset_config

# Define any additional specific parameters here
# ...

train_model() {
    log_and_run "Training model..." \
    env CUDA_VISIBLE_DEVICES=$CUDA_DEVICES torchrun --nproc_per_node=$NPROC_PER_NODE train.py \
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
        --num-workers $NUM_WORKERS 
}
sample_CG1() {
    log_and_run "Sampling images for CG1..." \
    env CUDA_VISIBLE_DEVICES=$CUDA_DEVICES torchrun --nproc_per_node=$NPROC_PER_NODE sample_ddp.py \
        --model "$MODEL" --vae "$VAE" \
        --sample-dir "$GENERATED_DIR_CG1" \
        --ckpt "$CHECKPOINT_DIR" \
        --per-proc-batch-size "$BATCH_SIZE" --num-fid-samples "$NSAMPLE" \
        --image-size "$IMAGE_SIZE" --num-classes "$NUM_CLASSES" \
        --cfg-scale 1 --num-sampling-steps "$NUM_SAMPLE_STEPS"
}

fid_CG1() {
    log_and_run "Calculating FID for CG1..." \
    env CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m dgm_eval "$REAL_DATA_DIR" "$GENERATED_DIR_CG1" \
        --model inception --device "$FID_DEVICE" --nsample $NSAMPLE --clean_resize \
        --metrics fd prdc --save --output_dir "$RESULTS_FILE_CG1"

    log_and_run "Calculating Dino FID for CG1..." \
    env CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m dgm_eval "$REAL_DATA_DIR" "$GENERATED_DIR_CG1" \
        --model dinov2 --device "$FID_DEVICE" --nsample $NSAMPLE --clean_resize \
        --metrics fd prdc --save --output_dir "$RESULTS_FILE_CG1"
}

sample_CG1_5() {
    log_and_run "Sampling images for CG1.5..." \
    env CUDA_VISIBLE_DEVICES=$CUDA_DEVICES torchrun --nproc_per_node=$NPROC_PER_NODE sample_ddp.py \
        --model "$MODEL" --vae "$VAE" \
        --sample-dir "$GENERATED_DIR_CG1_5" \
        --ckpt "$CHECKPOINT_DIR" \
        --per-proc-batch-size "$BATCH_SIZE" --num-fid-samples "$NSAMPLE" \
        --image-size "$IMAGE_SIZE" --num-classes "$NUM_CLASSES" \
        --cfg-scale 1.5 --num-sampling-steps "$NUM_SAMPLE_STEPS"
}

fid_CG1_5() {
    log_and_run "Calculating FID for CG1.5..." \
    env CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m dgm_eval "$REAL_DATA_DIR" "$GENERATED_DIR_CG1_5" \
        --model inception --device "$FID_DEVICE" --nsample $NSAMPLE --clean_resize \
        --metrics fd prdc --save --output_dir "$RESULTS_FILE_CG1_5"

    log_and_run "Calculating Dino FID for CG1.5..." \
    env CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m dgm_eval "$REAL_DATA_DIR" "$GENERATED_DIR_CG1_5" \
        --model dinov2 --device "$FID_DEVICE" --nsample $NSAMPLE --clean_resize \
        --metrics fd prdc --save --output_dir "$RESULTS_FILE_CG1_5"
}

sample_DoG1_5() {
    log_and_run "Sampling images for DoG1.5..." \
    env CUDA_VISIBLE_DEVICES=$CUDA_DEVICES torchrun --nproc_per_node=$NPROC_PER_NODE sample_dog_ddp.py \
        --model "$MODEL" --vae "$VAE" \
        --sample-dir "$GENERATED_DIR_DoG1_5" \
        --ckpt "$CHECKPOINT_DIR" \
        --per-proc-batch-size "$BATCH_SIZE" --num-fid-samples "$NSAMPLE" \
        --image-size "$IMAGE_SIZE" --num-classes "$NUM_CLASSES" \
        --cfg-scale 1.5 --num-sampling-steps "$NUM_SAMPLE_STEPS"
}

fid_DoG1_5() {
    log_and_run "Calculating FID for DoG1.5..." \
    env CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m dgm_eval "$REAL_DATA_DIR" "$GENERATED_DIR_DoG1_5" \
        --model inception --device "$FID_DEVICE" --nsample $NSAMPLE --clean_resize \
        --metrics fd prdc --save --output_dir "$RESULTS_FILE_DoG1_5"

    log_and_run "Calculating Dino FID for DoG1.5..." \
    env CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m dgm_eval "$REAL_DATA_DIR" "$GENERATED_DIR_DoG1_5" \
        --model dinov2 --device "$FID_DEVICE" --nsample $NSAMPLE --clean_resize \
        --metrics fd prdc --save --output_dir "$RESULTS_FILE_DoG1_5"
}

# ====================== MAIN ======================

echo ">>> Logging to: $LOG_FILE"
rm -f "$LOG_FILE"
mkdir -p "$(dirname "$LOG_FILE")"

create_environment
prepare_dataset
train_model

sample_CG1
fid_CG1

sample_CG1_5
fid_CG1_5

sample_DoG1_5
fid_DoG1_5

cleanup_dataset

echo ">>> All tasks completed successfully!"