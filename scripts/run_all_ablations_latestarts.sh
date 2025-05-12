#!/bin/bash

# ========== GLOBAL CONFIGURATION ==========
SERVER="computecanada"
CUDA_DEVICES="0,1"
SCRIPT="run_ablation_latestart.sh"

declare -a TASKS=(
  "stanford-cars_processed"
  "caltech-101_processed"
)

declare -a LATESTARTS=(0 2000 4000 6000 8000 10000)
declare -a MGHIGHS=(0.5 0.6 0.7 0.8 0.9 1)  # Add as needed

# ========== EXECUTION LOOP ==========
for DATASET in "${TASKS[@]}"; do
  for LATESTART in "${LATESTARTS[@]}"; do
    for MGHIGH in "${MGHIGHS[@]}"; do

      echo "=============================================="
      echo "Running $SCRIPT on $DATASET | latestart: $LATESTART | mghigh: $MGHIGH"
      echo "Server: $SERVER | CUDA Devices: $CUDA_DEVICES"
      echo "----------------------------------------------"

      EXPERIMENT_PRENAME="ablation_latestart"

      CMD="scripts/$SCRIPT \
        --dataset \"$DATASET\" \
        --server \"$SERVER\" \
        --cuda_devices \"$CUDA_DEVICES\" \
        --experiment_prename \"$EXPERIMENT_PRENAME\" \
        --latestart \"$LATESTART\" \
        --mghigh \"$MGHIGH\""

      if [[ "$SERVER" == "computecanada" ]]; then
        eval "sbatch $CMD"
      else
        eval "bash $CMD"
      fi

      echo "✅ Finished $SCRIPT on $DATASET | latestart: $LATESTART | mghigh: $MGHIGH"
      echo "=============================================="

    done
  done
done
