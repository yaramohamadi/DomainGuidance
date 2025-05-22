# Get Inception

# wget -P /home/ymbahram/projects/def-hadi87/ymbahram/DomainGuidance/pretrained_models/ https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth

wget -O /home/ymbahram/projects/def-hadi87/ymbahram/DomainGuidance/pretrained_models/dinov2_vitl14.pth https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth
\
# git clone https://github.com/facebookresearch/dinov2.git code


# module load python/3.11
# source /home/ymbahram/projects/def-hadi87/ymbahram/envs/DiT/bin/activate
# 
# # Save the current directory
# START_DIR=$(pwd)
# 
# # === Install dinov2 as editable package ===
# cd /home/ymbahram/projects/def-hadi87/ymbahram/DomainGuidance/code
# pip install -e .
# 
# # === Return to original working directory ===
# cd "$START_DIR"