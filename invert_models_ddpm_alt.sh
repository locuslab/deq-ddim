#!/bin/bash

#python main.py --config celeba_ls_opt_ddpm.yml --model Diffusion --exp celeba-orig --image_folder celeba-ddim-eta-0.1-t-20 --doc celeba-orig-ddim --ls_opt --timesteps 20 --ni --method ddpm --seed 11 --eta 1 --lambda1 1 --lambda2 0 --lambda3 0 --seed 11 --tau 0.1 --no_augmentation --pg_steps 1 --use_wandb
# pg_steps=(3)
# tau=(0.1 0.25 0.5 0.75 0.9)
# lambda1_vals=(0.1 0.5 0.75)
# for tau in ${tau[@]}; do
#   for pg_step in ${pg_steps[@]}; do
#     for l1 in ${lambda1_vals[@]}; do
#       python main.py --config celeba_ls_opt_ddpm.yml --model Diffusion --exp celeba-orig --image_folder celeba-ddim-eta-0.1-t-20 --doc celeba-orig-ddim --ls_opt --timesteps 20 --ni --method ddpm --seed 11 --eta 0.5 --lambda1 $l1 --lambda2 1 --lambda3 0 --seed 11 --tau $tau --no_augmentation --pg_steps $pg_step --use_wandb
#     done
#   done
# done

python main.py --config celeba_ls_opt_dense_ddim.yml --model Diffusion --exp celeba-orig --image_folder celeba-ddim-dense-eta-0-t-20 --doc celeba-orig-ddim --ls_opt --timesteps 20 --ni --method simple-seq --seed 11 --eta 0 --lambda1 1 --lambda2 0 --lambda3 0 --seed 1 --tau 0.1 --no_augmentation --pg_steps 1 --use_wandb