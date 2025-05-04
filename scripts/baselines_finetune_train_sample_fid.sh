#!/bin/bash
# set -e  # Exit on any error

# ====================== CONFIGURATION ======================

# Define CUDA devices here
CUDA_DEVICES="0,1"
FID_DEVICE="cuda:0"
NPROC_PER_NODE=2

EXPERIMENT_NAME="baselines_finetune"
DATASET="cub-200-2011_processed"  # Options: caltech, birds, etc.

CODE_PRE_DIR="/projets/Ymohammadi/DomainGuidance"
RESULTS_PRE_DIR="/export/livia/home/vision/Ymohammadi/DoG"
DATA_TARGET_DIR="/projets/Ymohammadi/DomainGuidance/datasets"
ENV_PATH="/projets/Ymohammadi/envs/DiT"

GENERATED_DIR_CG1="$RESULTS_PRE_DIR/$DATASET/$EXPERIMENT_NAME/samples/0024000/samples_CFG1/"
GENERATED_DIR_CG1_5="$RESULTS_PRE_DIR/$DATASET/$EXPERIMENT_NAME/samples/0024000/samples_CFG1_5"
GENERATED_DIR_DoG1_5="$RESULTS_PRE_DIR/$DATASET/$EXPERIMENT_NAME/samples/0024000/samples_DOG1_5"
RESULTS_FILE_CG1="$RESULTS_PRE_DIR/$DATASET/$EXPERIMENT_NAME/results/CFG1_results"
RESULTS_FILE_CG1_5="$RESULTS_PRE_DIR/$DATASET/$EXPERIMENT_NAME/results/CFG1_5_results"
RESULTS_FILE_DoG1_5="$RESULTS_PRE_DIR/$DATASET/$EXPERIMENT_NAME/results/DoG1_5/results"

RESULTS_DIR="$RESULTS_PRE_DIR/$DATASET/$EXPERIMENT_NAME/"
CHECKPOINT_DIR="$RESULTS_PRE_DIR/$DATASET/$EXPERIMENT_NAME/checkpoints/0024000.pt"
LOG_FILE="$RESULTS_PRE_DIR/$DATASET/$EXPERIMENT_NAME/training_log.txt"

case "$DATASET" in
  caltech-101)
    DATA_DIR_ZIP="/export/datasets/public/diffusion_datasets/caltech-101_processed/$DATASET.zip"
    REAL_DATA_DIR="$DATA_TARGET_DIR/$DATASET"
    NUM_CLASSES=101
    ;;
  cub-200-2011_processed)
    DATA_DIR_ZIP="/export/datasets/public/diffusion_datasets/cub-200-2011_processed/$DATASET.zip"
    REAL_DATA_DIR="$DATA_TARGET_DIR/$DATASET"
    NUM_CLASSES=200
    ;;
  *)
    echo "Unknown dataset: $DATASET"
    exit 1
    ;;
esac

# CONSTANTS
IMAGE_SIZE=256
TOTAL_STEPS=24000
MODEL=DiT-XL/2
LOG_EVERY=1000
CKPT_EVERY=24000
BATCH_SIZE=32
VAE=ema
NUM_WORKERS=4

NSAMPLE=10000
CFG_SCALE=1.0
NUM_SAMPLE_STEPS=50


# ====================== HELPER ======================
log_and_run() {
    echo ">>> $1"
    shift
    "$@" 2>&1 | tee -a "$LOG_FILE"
}

mkdir -p "$(dirname "$LOG_FILE")"

# ====================== STEPS ======================
create_environment() {
    echo ">>> Setting up environment..."
    if [ -d "$ENV_PATH" ]; then
        echo "Using existing conda env at $ENV_PATH"
    else
        conda create --prefix "$ENV_PATH" python=3.11 -y
    fi
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_PATH"
    conda install pytorch torchvision pytorch-cuda=12.6 -c pytorch -c nvidia
    pip install timm diffusers accelerate pytorch-fid

    if [ ! -d "dgm-eval" ]; then
        git clone https://github.com/layer6ai-labs/dgm-eval.git
    fi
    pushd dgm-eval
    pip install -e .
    popd
}
prepare_dataset() {
    mkdir -p "$DATA_TARGET_DIR"
    cp "$DATA_DIR_ZIP" "$DATA_TARGET_DIR/"
    unzip -o "$DATA_DIR_ZIP" -d "$DATA_TARGET_DIR"
    echo ">>> Dataset prepared at: $REAL_DATA_DIR"
}
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

cleanup_dataset() {
    echo ">>> Cleaning up dataset..."
    rm -rf "$DATA_TARGET_DIR"
    echo ">>> Dataset removed."
}

# ====================== MAIN ======================
echo ">>> Logging to: $LOG_FILE"
rm -f "$LOG_FILE"

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