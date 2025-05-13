unset PIP_FIND_LINKS
unset PIP_NO_INDEX

ENV_PATH="/home/ymbahram/projects/def-hadi87/ymbahram/envs/DiT"


echo ">>> Setting up environment..."

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

# Install dgm-eval
if [ ! -d "dgm-eval" ]; then
    git clone https://github.com/layer6ai-labs/dgm-eval.git
fi
pushd dgm-eval
pip install -e . --no-index
popd