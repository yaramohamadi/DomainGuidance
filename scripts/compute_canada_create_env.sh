#!/bin/bash

ENV_PATH="/home/ymbahram/projects/def-hadi87/ymbahram/envs/DiT"

echo ">>> Setting up environment..."

module load httpproxy #beluga and narval
module load StdEnv/2023 intel/2023.2.1
module load cuda/11.8
module load StdEnv/2023  gcc/12.3
module load opencv/4.9.0

module load StdEnv/2023 python/3.11.5

python -m venv $ENV_PATH
source $ENV_PATH/bin/activate

pip install --no-index --ignore-installed torch==2.6.0 timm==1.0.15 diffusers==0.32.2 accelerate==1.6.0 pytorch-fid==0.3.0  scikit-image==0.25.1 scikit-learn==1.6.1 transformers==4.52.3 xformers==0.0.29.post2 scipy==1.15.1 open_clip_torch==2.29.0 pandas==2.2.3 pillow==11.1.0
pip install triton

# Install dgm-eval
if [ ! -d "dgm-eval" ]; then
    git clone https://github.com/layer6ai-labs/dgm-eval.git
fi

cp scripts/compute_canada_create_env.py dgm-eval/setup.py
pushd dgm-eval
pip install --no-index -e .
popd
