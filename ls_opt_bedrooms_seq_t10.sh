#!/bin/bash
for seed in $(seq 63 69)
  do
    python main.py --config bedroom_ls_opt_seq.yml --model Diffusion --exp lsun_orig --image_folder lsun_bedroom_images_t50 --doc lsun_orig --ls_opt --timesteps 10 --ni --method default --lambda1 1 --lambda2 0 --lambda3 0 --seed $seed --tau 0.1 --no_augmentation --pg_steps 1 --use_pretrained --use_wandb
  done
