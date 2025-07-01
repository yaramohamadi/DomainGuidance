ROOT_DIR="/home/ymbahram/scratch/results/DoG/"  # 🔁 Replace this with your actual root directory

# Find all directories named "samples"
find "$ROOT_DIR" -type d -name "samples" | while read -r samples_dir; do
    echo "[INFO] Clearing contents of: $samples_dir"
    
    # Remove all contents inside the "samples" directory
    find "$samples_dir" -mindepth 1 -exec rm -rf {} +
done