#!/bin/bash

########### CIFAR10 ##################
#### DEQ-sDDIM #############################

for i in $(seq 1 25)
    do
    python main.py --config cifar10_ls_opt_ddpm.yml --model DiffusionInversion --exp cifar10-orig-exp --image_folder cifar10-ddim-eta-0.5-t20-seq --doc cifar10-orig --ls_opt --timesteps 10 --ni --method ddpm --lambda1 1 --lambda2 0 --lambda3 0 --seed $i --no_augmentation --tau 0.1 --pg_steps 1 --eta 1 --use_wandb --use_pretrained
    done

########### CelebA ###################

for seed in $(seq 1 25)
   do
       python main.py --config celeba_ls_opt_ddpm.yml --model DiffusionInversion --exp celeba-orig --image_folder celeba-ddim-eta-0.1-t-20 --doc celeba-orig-ddim --ls_opt --timesteps 20 --ni --method ddpm --seed $seed --eta 0.1 --lambda1 1 --lambda2 0 --lambda3 0 --seed $seed --tau 0.1 --no_augmentation --pg_steps 1 --use_wandb
   done

############ LSUN Bedroom ############# 
for seed in $(seq 1 25)
   do
       python main.py --config bedroom_ls_opt_ddpm.yml --model DiffusionInversion --exp lsun_orig --image_folder lsun_bedroom_images --doc lsun_orig --ls_opt --timesteps 10 --ni --method ddpm --eta 0.5 --lambda1 1 --lambda2 0 --lambda3 0 --seed $seed --tau 0.1 --no_augmentation --pg_steps 1 --use_pretrained --use_wandb
   done

############ LSUN Church ###############

for seed in $(seq 1 25)
    do
        python main.py --config church_ls_opt_ddpm.yml --model DiffusionInversion --exp lsun_orig --image_folder lsun_church_images --doc lsun_orig --ls_opt --timesteps 10 --ni --method ddpm --eta 0.5 --lambda1 1 --lambda2 0 --lambda3 0 --seed $seed --tau 0.1 --no_augmentation --pg_steps 1 --use_pretrained --use_wandb
    done