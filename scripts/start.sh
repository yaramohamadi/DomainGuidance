CUDA_VISIBLE_DEVICES=1,2 python --nproc_per_node=1 train.py \
  --data-path ./my_target_few_shot/ \
  --results-dir ./results/ \
  --model DiT-XL/2 \
  --image-size 256 \
  --num-classes 1000 \
  --epochs 1400 \
  --global-batch-size 32 \
  --vae ema \
  --num-workers 2