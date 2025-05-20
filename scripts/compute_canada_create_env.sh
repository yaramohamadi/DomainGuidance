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

# Load required system-level modules before activating the virtualenv
module load StdEnv/2020 gcc/9.3.0 cuda/11.7 opencv/4.7.0

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


# Patch numpy requirement to avoid strict 1.23.3 version
sed -i 's/numpy==1\.23\.3/numpy>=1.23/' setup.py

# Remove OpenCV from setup.py entirely (Compute Canada provides it via modules)
sed -i '/opencv-python/d' setup.py

# Patch open_clip_torch version to compatible one
sed -i 's/open_clip_torch==2\.19\.0/open_clip_torch>=2.0/' setup.py

# Patch pillow version to compatible one
sed -i 's/pillow==9\.2\.0/pillow>=10.0/' setup.py

# Patch scikit-image version to compatible one
sed -i 's/scikit-image==0\.19\.3/scikit-image>=0.20/' setup.py

# Patch scikit-learn version to compatible one
sed -i 's/scikit-learn==1\.1\.3/scikit-learn>=1.2/' setup.py

# Patch scipy version to compatible one
sed -i 's/scipy==1\.9\.3/scipy>=1.10/' setup.py

# Patch timm version to compatible one
sed -i 's/timm==0\.8\.19\.dev0/timm>=0.9/' setup.py


pip install -e . --no-index
popd
