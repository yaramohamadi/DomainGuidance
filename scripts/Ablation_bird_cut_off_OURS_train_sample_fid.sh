#!/bin/bash
# set -e  # Exit on any error

# ====================== CONFIGURATION ======================

# Define CUDA devices here
CUDA_DEVICES="2,3"
FID_DEVICE="cuda:0"
NPROC_PER_NODE=2

EXPERIMENT_NAME="dogfinetune1_5_EMA_CUTOFF"
DATASET="cub-200-2011_processed"  # Options: caltech, birds, etc.

NSAMPLE=1000
W_TRAIN_DOG=1.5
USE_GUIDANCE_CUTOFF=1

CODE_PRE_DIR="/projets/Ymohammadi/DomainGuidance"
DATA_TARGET_DIR="/projets/Ymohammadi/DomainGuidance/datasets"
DATASETS_DIR="/export/datasets/public/diffusion_datasets"
RESULTS_PRE_DIR="/export/datasets/public/diffusion_datasets/tmp_weights"
ENV_PATH="/projets/Ymohammadi/envs/DiT"

case "$DATASET" in
  caltech-101)
    DATA_DIR_ZIP="$DATASETS_DIR/caltech-101_processed/$DATASET.zip"
    REAL_DATA_DIR="$DATA_TARGET_DIR/$DATASET"
    NUM_CLASSES=101
    ;;
  cub-200-2011_processed)
    DATA_DIR_ZIP="$DATASETS_DIR/cub-200-2011_processed/$DATASET.zip"
    REAL_DATA_DIR="$DATA_TARGET_DIR/$DATASET"
    NUM_CLASSES=200
    ;;
  stanford-cars_processed)
    DATA_DIR_ZIP="$DATASETS_DIR/stanford-cars_processed/$DATASET.zip"
    REAL_DATA_DIR="$DATA_TARGET_DIR/$DATASET"
    NUM_CLASSES=196
    ;;
  food-101_processed)
    DATA_DIR_ZIP="$DATASETS_DIR/food-101_processed/$DATASET.zip"
    REAL_DATA_DIR="$DATA_TARGET_DIR/$DATASET"
    NUM_CLASSES=101
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
CKPT_EVERY=4000
BATCH_SIZE=32
VAE=ema
NUM_WORKERS=4

# Sampling parameters
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
    env CUDA_VISIBLE_DEVICES=$CUDA_DEVICES torchrun --nproc_per_node=$NPROC_PER_NODE train_OURS.py \
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
        --mg-high "$MG_HIGH"
}

run_sampling() {
    log_and_run "Sampling images..." \
    env CUDA_VISIBLE_DEVICES=$CUDA_DEVICES torchrun --nproc_per_node=$NPROC_PER_NODE sample_ddp.py \
        --model "$MODEL" \
        --vae "$VAE" \
        --sample-dir "$GENERATED_DIR" \
        --ckpt "$CHECKPOINT_DIR" \
        --per-proc-batch-size "$BATCH_SIZE" \
        --num-fid-samples "$NSAMPLE" \
        --image-size "$IMAGE_SIZE" \
        --num-classes "$NUM_CLASSES" \
        --cfg-scale "$CFG_SCALE" \
        --num-sampling-steps "$NUM_SAMPLE_STEPS"
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

cleanup_dataset() {
    echo ">>> Cleaning up dataset..."
    rm -rf "$DATA_TARGET_DIR"
    echo ">>> Dataset removed."
}

# ====================== MAIN ======================

create_environment
prepare_dataset

MG_HIGH_VALUES=(0.6 0.7 0.8 0.9 1.0)

for MG_HIGH in "${MG_HIGH_VALUES[@]}"; do
    EXPERIMENT_NAME="ablation_mghigh/dogfinetune1_5_EMA_CUTOFF_MG${MG_HIGH}"

    RESULTS_DIR="$RESULTS_PRE_DIR/$DATASET/$EXPERIMENT_NAME/"
    GENERATED_DIR="$RESULTS_PRE_DIR/$DATASET/$EXPERIMENT_NAME/samples/0024000"
    CHECKPOINT_DIR="$RESULTS_PRE_DIR/$DATASET/$EXPERIMENT_NAME/checkpoints/0024000.pt"
    LOG_FILE="$RESULTS_PRE_DIR/$DATASET/$EXPERIMENT_NAME/training_log.txt"
    RESULTS_FILE="$RESULTS_PRE_DIR/$DATASET/$EXPERIMENT_NAME/results"

    echo ">>> Running experiment: $EXPERIMENT_NAME with MG_HIGH=$MG_HIGH"
    echo ">>> Logging to: $LOG_FILE"
    rm -f "$LOG_FILE"
    train_model
    run_sampling
    calculate_fid
    echo ">>> All tasks completed successfully!"
done

cleanup_dataset