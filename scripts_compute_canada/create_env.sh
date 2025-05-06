create_environment() {
    echo ">>> Setting up Python virtual environment..."

    # Load required modules (adjust CUDA version if needed)
    module load python/3.11 cuda/12.2  # Match CUDA to your desired version

    # Create the environment directory if it doesn't exist
    if [ -d "$ENV_PATH" ]; then
        echo "Using existing virtualenv at $ENV_PATH"
    else
        python -m venv "$ENV_PATH"
    fi

    # Activate the virtual environment
    source "$ENV_PATH/bin/activate"

    # Upgrade pip and install packages
    pip install --upgrade pip
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install timm diffusers accelerate pytorch-fid

    # Clone and install dgm-eval if not present
    if [ ! -d "dgm-eval" ]; then
        git clone https://github.com/layer6ai-labs/dgm-eval.git
    fi
    pushd dgm-eval
    pip install -e .
    popd
}

create_environment