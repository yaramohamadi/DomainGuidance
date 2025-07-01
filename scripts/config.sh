#!/bin/bash

# Global Constants
TOTAL_STEPS=40000
LOG_EVERY=1000
CKPT_EVERY=8000
SKIP_FIRST_CKPT=1

IMAGE_SIZE=256
MODEL="DiT-XL/2"
BATCH_SIZE=32
VAE="ema"
NUM_WORKERS=4
CFG_SCALE=1.0
NUM_SAMPLE_STEPS=50
NSAMPLE=10000

FID_DEVICE="cuda:0"
NPROC_PER_NODE=2
is_port_in_use() {
  local port=$1
  (echo >/dev/tcp/127.0.0.1/$port) &>/dev/null
}

PORT=$(shuf -i 20000-40000 -n 1)  # 🔧 Initialize first
while is_port_in_use $PORT; do
  PORT=$(shuf -i 20000-40000 -n 1)
done
export PORT
echo "Using MASTER_PORT=$PORT"

# Set paths and dataset details
resolve_dataset_config() {
  case "$DATASET" in
    caltech-101_processed) NUM_CLASSES=101 ;;
    cub-200-2011_processed) NUM_CLASSES=200 ;;
    stanford-cars_processed) NUM_CLASSES=196 ;;
    food-101_processed) NUM_CLASSES=101 ;;
    df-20m_processed) NUM_CLASSES=1577 ;;
    artbench-10_processed) NUM_CLASSES=10 ;;
    ffhq256) NUM_CLASSES=1 ;;
    *) echo "Unknown dataset: $DATASET"; exit 1 ;;
  esac
  DATA_DIR_ZIP="$DATASETS_DIR/$DATASET.zip"
  REAL_DATA_DIR="$DATA_TARGET_DIR/$DATASET"
  # Normal Generated Directory for Ours and MG
  RESULTS_DIR="$RESULTS_PRE_DIR/$DATASET/$EXPERIMENT_NAME/"
  GENERATED_DIR="$RESULTS_DIR/samples"
  # Generated Directory for baselines
  GENERATED_DIR_CG1="$RESULTS_DIR/samples"
  RESULTS_FILE_CG1="$RESULTS_DIR/results_cg1"
  GENERATED_DIR_CG1_5="$RESULTS_DIR/samples"
  RESULTS_FILE_CG1_5="$RESULTS_DIR/results_cg1_5"
  GENERATED_DIR_DoG1_5="$RESULTS_DIR/samples"
  RESULTS_FILE_DoG1_5="$RESULTS_DIR/results_dog1_5"
  # Till here...
  CHECKPOINT_DIR="$RESULTS_DIR/checkpoints"
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
            DATA_TARGET_DIR="/home/ymbahram/scratch/diffusion_datasets"
            DATASETS_DIR="/home/ymbahram/scratch/diffusion_datasets"
            RESULTS_PRE_DIR="/home/ymbahram/scratch/results/DoG"
            ENV_PATH="/home/ymbahram/projects/def-hadi87/ymbahram/envs/DiT"
            python download.py
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

    module load httpproxy #beluga and narval
    module load StdEnv/2023 intel/2023.2.1
    module load cuda/11.8
    module load StdEnv/2023  gcc/12.3
    module load opencv/4.9.0

    module load StdEnv/2023 python/3.11.5
    source $ENV_PATH/bin/activate

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
      pip install timm diffusers accelerate pytorch-fid torchdiffeq
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
  # Clean up metadata if exists
  find "$REAL_DATA_DIR" -name '._*' -delete

  if [[ "$SERVER" == "computecanada" ]]; then
    echo ">>> Running on Compute Canada"
    if [ -d "$REAL_DATA_DIR" ] && [ "$(ls -A "$REAL_DATA_DIR")" ]; then
      echo ">>> Dataset already exists at: $REAL_DATA_DIR. Proceeding..."
    else
      echo ">>> ERROR: Dataset not found at $REAL_DATA_DIR."
      echo ">>> Dataset preparation must be done before SLURM job submission on Compute Canada."
      exit 1
    fi
    return
  fi

  # For other servers, proceed with normal extraction
  if [ -d "$REAL_DATA_DIR" ] && [ "$(ls -A "$REAL_DATA_DIR")" ]; then
    echo ">>> Dataset already exists at: $REAL_DATA_DIR. Skipping extraction."
  fi

  echo ">>> Preparing dataset..."
  mkdir -p "$DATA_TARGET_DIR"
  unzip -qn "$DATA_DIR_ZIP" -d "$DATA_TARGET_DIR"
  find "$REAL_DATA_DIR" -name '._*' -delete
  echo ">>> Dataset prepared at: $REAL_DATA_DIR"

  # Special case: ffhq256 needs images inside a dummy class folder
  if [[ "$DATASET" == "ffhq256" ]]; then
    echo ">>> Detected ffhq256: moving images to dummy class folder..."
    mkdir -p "$REAL_DATA_DIR/dummy_class"
    find "$REAL_DATA_DIR" -maxdepth 1 -type f -iname '*.png' -exec mv {} "$REAL_DATA_DIR/dummy_class/" \;
  fi

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