#!/bin/bash
#SBATCH --account=def-hadi87
#SBATCH --job-name=${JOB_NAME:-Ours}  # Use $JOB_NAME if defined, else 'myjob'
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err          
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:2              
#SBATCH --mem=80G                        
#SBATCH --mail-user=yara.mohammadi-bahram.1@ens.etsmtl.ca 
#SBATCH --mail-type=ALL           


# def-hadi87
# ====================== DEFAULT CONFIGURATION ======================

CUDA_DEVICES="0,1"
DATASET="cub-200-2011_processed"
SERVER="taylor"
EXPERIMENT_PRENAME=""
USE_GUIDANCE_CUTOFF=1
MG_HIGH=1
LATE_START=0

W_TRAIN_DOG=1.5
DROPOUT_RATIO=0 # TODO Change this back to 0   
DIFFFIT=0

# Load all logic
source scripts/config.sh

# ====================== ARGUMENT PARSING ======================

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --cuda_devices) CUDA_DEVICES="$2"; shift ;;
    --dataset) DATASET="$2"; shift ;;
    --server) SERVER="$2"; shift ;;
    --experiment_prename) EXPERIMENT_PRENAME="$2"; shift ;;
    --latestart) LATE_START="$2"; shift ;;
    --mghigh) MG_HIGH="$2"; shift ;;
    --wtraindog) W_TRAIN_DOG="$2"; shift ;;
    --model_name) MODEL="$2"; shift ;;
    --difffit) DIFFFIT="$2"; shift ;;
    --num_sample_steps) NUM_SAMPLE_STEPS="$2"; shift ;;
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
  shift
done

EXPERIMENT_NAME="$EXPERIMENT_PRENAME/dogfinetune_LATE_START_ITER${LATE_START}_MG${MG_HIGH}_W_TRAIN_DOG${W_TRAIN_DOG}"

resolve_server_paths
resolve_dataset_config

# GENERATED_DIR="$SLURM_TMPDIR/samples" # TODO -> Change for qualitative experiments in which you need samples
GENERATED_DIR=$GENERATED_DIR/$NUM_SAMPLE_STEPS 
RESULTS_FILE=$RESULTS_FILE/$NUM_SAMPLE_STEPS 
# Define any additional specific parameters here

train_model() {
    log_and_run "Training model..." \
    env CUDA_VISIBLE_DEVICES=$CUDA_DEVICES torchrun --master_port=$PORT --nproc_per_node=$NPROC_PER_NODE train_OURS.py \
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
        --mg-high "$MG_HIGH" \
        --dropout-ratio "$DROPOUT_RATIO" \
        --late-start-iter "$LATE_START" \
        --difffit "$DIFFFIT"
}

run_sampling() {
    log_and_run "Sampling images..." \
    env CUDA_VISIBLE_DEVICES=$CUDA_DEVICES torchrun --master_port=$PORT --nproc_per_node=$NPROC_PER_NODE sample_ddp.py \
        --model "$MODEL" \
        --vae "$VAE" \
        --sample-dir "$GENERATED_DIR/$PADDED_STEP" \
        --ckpt "$CHECKPOINT_DIR/$PADDED_CKPT" \
        --per-proc-batch-size "$BATCH_SIZE" \
        --num-fid-samples "$NSAMPLE" \
        --image-size "$IMAGE_SIZE" \
        --num-classes "$NUM_CLASSES" \
        --cfg-scale "$CFG_SCALE" \
        --num-sampling-steps "$NUM_SAMPLE_STEPS" \
        --dropout-ratio "$DROPOUT_RATIO" \
        --difffit "$DIFFFIT"
}

calculate_fid() {
    # log_and_run "Calculating FID DINO..." \
    # env CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m dgm_eval "$REAL_DATA_DIR" "$GENERATED_DIR/$PADDED_STEP" \
    #     --model dinov2 \
    #     --device "$FID_DEVICE" \
    #     --nsample "$NSAMPLE" \
    #     --clean_resize \
    #     --metrics fd prdc \
    #     --save \
    #     --output_dir "$RESULTS_FILE"

    log_and_run "Calculating FID..." \
    env CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m dgm_eval "$REAL_DATA_DIR" "$GENERATED_DIR/$PADDED_STEP" \
        --model inception \
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
# train_model

for ((i=0; i<=TOTAL_STEPS; i+=CKPT_EVERY)); do
  if [[ $i -eq 0 && "$SKIP_FIRST_CKPT" -eq 1 ]]; then
    continue
  fi
  printf -v PADDED_STEP "%07d" "$i"
  printf -v PADDED_CKPT "%07d.pt" "$i"
  run_sampling
  calculate_fid
done

# cleanup_dataset

echo ">>> All tasks completed successfully!"