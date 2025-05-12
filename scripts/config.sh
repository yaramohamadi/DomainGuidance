#!/bin/bash

# Global Constants
IMAGE_SIZE=256
TOTAL_STEPS=24000
MODEL="DiT-XL/2"
LOG_EVERY=1000
CKPT_EVERY=24000
BATCH_SIZE=32
VAE="ema"
NUM_WORKERS=4
CFG_SCALE=1.0
NUM_SAMPLE_STEPS=50
NSAMPLE=10000

FID_DEVICE="cuda:0"
NPROC_PER_NODE=2

PORT=$(shuf -i 20000-40000 -n 1)
MASTER_PORT=$PORT

# Set paths and dataset details
resolve_dataset_config() {
  case "$DATASET" in
    caltech-101_processed) NUM_CLASSES=101 ;;
    cub-200-2011_processed) NUM_CLASSES=200 ;;
    stanford-cars_processed) NUM_CLASSES=196 ;;
    food-101_processed) NUM_CLASSES=101 ;;
    df-20m_processed) NUM_CLASSES=1577 ;;
    artbench-10_processed) NUM_CLASSES=10 ;;
    *) echo "Unknown dataset: $DATASET"; exit 1 ;;
  esac
  DATA_DIR_ZIP="$DATASETS_DIR/$DATASET.zip"
  REAL_DATA_DIR="$DATA_TARGET_DIR/$DATASET"
  # Normal Generated Directory for Ours and MG
  RESULTS_DIR="$RESULTS_PRE_DIR/$DATASET/$EXPERIMENT_NAME/"
  GENERATED_DIR="$RESULTS_DIR/samples/0024000"
  # Generated Directory for baselines
  GENERATED_DIR_CG1="$RESULTS_DIR/samples/0024000_cg1"
  RESULTS_FILE_CG1="$RESULTS_DIR/results_cg1"
  GENERATED_DIR_CG1_5="$RESULTS_DIR/samples/0024000_cg1_5"
  RESULTS_FILE_CG1_5="$RESULTS_DIR/results_cg1_5"
  GENERATED_DIR_DoG1_5="$RESULTS_DIR/samples/0024000_dog1_5"
  RESULTS_FILE_DoG1_5="$RESULTS_DIR/results_dog1_5"
  # Till here...
  CHECKPOINT_DIR="$RESULTS_DIR/checkpoints/0024000.pt"
  LOG_FILE="$RESULTS_DIR/training_log.txt"
  RESULTS_FILE="$RESULTS_DIR/results"
}

# ==== PATHS ====
resolve_server_paths() {
    case "$SERVER" in
        taylor)
            export PATH="$HOME/.local/bin:$PATH"
            export PATH="$HOME/miniconda3/bin:$PATH"
            ~/miniconda3/bin/conda init bash
            source ~/.bashrc
            CODE_PRE_DIR="/home/ens/AT74470/DomainGuidance"
            DATA_TARGET_DIR="/home/ens/AT74470/datasets"
            DATASETS_DIR="/home/ens/AT74470/datasets"
            RESULTS_PRE_DIR="/home/ens/AT74470/results/DoG"
            ENV_PATH="/home/ens/AT74470/envs/DiT"
            ;;
        bool)
            conda init
            source ~/.bashrc
            CODE_PRE_DIR="/projets/Ymohammadi/DomainGuidance"
            DATA_TARGET_DIR="/projets/Ymohammadi/DomainGuidance/datasets"
            DATASETS_DIR="/export/datasets/public/diffusion_datasets"
            RESULTS_PRE_DIR="/export/datasets/public/diffusion_datasets/tmp_weights"
            ENV_PATH="/projets/Ymohammadi/envs/DiT"
            ;;
        computecanada) 
            CODE_PRE_DIR="/home/ens/AT74470/DomainGuidance" # TODO 
            DATA_TARGET_DIR="$SLURM_TMPDIR"
            DATASETS_DIR="/home/ymbahram/scratch/diffusion_datasets"
            RESULTS_PRE_DIR="/home/ymbahram/scratch/results/DoG"
            ENV_PATH="/home/ymbahram/projects/def-hadi87/ymbahram/envs/DiT"
            ;;
        *)
            echo "Unknown server: $SERVER" >&2
            exit 1
            ;;
    esac
}

create_environment() {
  echo ">>> Setting up environment..."

  if [[ "$SERVER" == "computecanada" ]]; then
    echo ">>> Detected Compute Canada: using virtualenv + module setup"

    # Load Compute Canada modules
    module load python/3.11 cuda/12.2

    # Create virtualenv if needed
    if [ -d "$ENV_PATH" ]; then
      echo "Using existing virtualenv at $ENV_PATH"
    else
      python -m venv "$ENV_PATH"
    fi
    
    # Activate and install packages
    source "$ENV_PATH/bin/activate"
    nvidia-smi
    pip install --upgrade pip --no-index
    pip install torch torchvision --no-index
    pip install timm diffusers accelerate pytorch-fid --no-index
    pip install numpy==1.23.2 --no-index

    # Install dgm-eval
    if [ ! -d "dgm-eval" ]; then
      git clone https://github.com/layer6ai-labs/dgm-eval.git
    fi
    pushd dgm-eval
    pip install --no-deps -e .
    popd

  else
    echo ">>> Detected local server: using conda env setup"

    if [ -d "$ENV_PATH" ]; then
      echo "Using existing conda env at $ENV_PATH"
      conda activate "$ENV_PATH"
    else
      conda create --prefix "$ENV_PATH" python=3.11 -y
      source "$(conda info --base)/etc/profile.d/conda.sh"
      conda activate "$ENV_PATH"
      conda install pytorch torchvision pytorch-cuda=12.6 -c pytorch -c nvidia
      pip install timm diffusers accelerate pytorch-fid
    fi

    if [ ! -d "dgm-eval" ]; then
      git clone https://github.com/layer6ai-labs/dgm-eval.git
    fi
    pushd dgm-eval
    pip install -e .
    popd
  fi
}

prepare_dataset() {
  find $REAL_DATA_DIR -name '._*' -delete # Delete metadata if exists in dataset (Exists for Artbench)
  if [ -d "$REAL_DATA_DIR" ] && [ "$(ls -A "$REAL_DATA_DIR")" ]; then
    echo ">>> Dataset already exists at: $REAL_DATA_DIR. Skipping extraction."
    return
  fi

  echo ">>> Preparing dataset..."
  mkdir -p "$DATA_TARGET_DIR"
  unzip -qn "$DATA_DIR_ZIP" -d "$DATA_TARGET_DIR"
  echo ">>> Dataset prepared at: $REAL_DATA_DIR"
}


# ====================== HELPER ======================
log_and_run() {
    echo ">>> $1"
    shift
    "$@" 2>&1 | tee -a "$LOG_FILE"
}

cleanup_dataset() {
    if [[ "$SERVER" == "bool" ]]; then
        echo ">>> Cleaning up dataset..."
        rm -rf "$DATA_TARGET_DIR"
        echo ">>> Dataset removed."
    else
        echo ">>> Skipping dataset cleanup (SERVER = $SERVER)"
    fi
}