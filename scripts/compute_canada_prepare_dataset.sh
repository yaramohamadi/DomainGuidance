#!/bin/bash

# ====== CONFIGURATION ======
SERVER="computecanada"
DATASET="stanford-cars_processed"  # <-- change this to the dataset you want to prepare

# ====== PATHS ======
resolve_paths_and_dataset() {
    case "$SERVER" in
        computecanada)
            CODE_PRE_DIR="/home/ens/AT74470/DomainGuidance"
            DATA_TARGET_DIR="/home/ymbahram/scratch/diffusion_datasets"
            DATASETS_DIR="/home/ymbahram/scratch/diffusion_datasets"
            RESULTS_PRE_DIR="/home/ymbahram/scratch/results/DoG"
            ENV_PATH="/home/ymbahram/projects/def-hadi87/ymbahram/envs/DiT"
            ;;
        *)
            echo "Unknown server: $SERVER" >&2
            exit 1
            ;;
    esac

    DATA_DIR_ZIP="$DATASETS_DIR/$DATASET.zip"
    REAL_DATA_DIR="$DATA_TARGET_DIR/$DATASET"
}

# ====== PREPARE DATASET ======
prepare_dataset() {
  find "$REAL_DATA_DIR" -name '._*' -delete 2>/dev/null
  if [ -d "$REAL_DATA_DIR" ] && [ "$(ls -A "$REAL_DATA_DIR")" ]; then
    echo ">>> Dataset already exists at: $REAL_DATA_DIR. Skipping extraction."
    return
  fi
  echo ">>> Preparing dataset: $DATASET"
  mkdir -p "$DATA_TARGET_DIR"
  unzip -qn "$DATA_DIR_ZIP" -d "$DATA_TARGET_DIR"
  find "$REAL_DATA_DIR" -name '._*' -delete 2>/dev/null
  echo ">>> Dataset prepared at: $REAL_DATA_DIR"
}

# ====== MAIN ======
resolve_paths_and_dataset
prepare_dataset
