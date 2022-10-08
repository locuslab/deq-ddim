#!/bin/bash

########### CIFAR10 ##################
#### DEQ #############################

for i in $(seq 1 100)
do
    python main.py --config cifar10_ls_opt.yml --model DiffusionInversion --exp cifar10-orig-exp --image_folder cifar10-inverse-t1000-parallel --doc cifar10-orig --ls_opt --timesteps 1000 --ni --method default --lambda1 1 --lambda2 0 --lambda3 0 --seed $i --tau 0.1 --use_wandb --no_augmentation --pg_steps 1 --use_pretrained
done
#### Baseline ########################
for i in $(seq 1 100)
do
    python main.py --config cifar10_ls_opt_seq.yml --model DiffusionInversion --exp cifar10-orig-exp --image_folder cifar10-inverse-t1000-seq --doc cifar10-orig --ls_opt --timesteps 1000 --ni --method default --lambda1 1 --lambda2 0 --lambda3 0 --seed $i --use_wandb --no_augmentation --use_pretrained
done

########### CelebA ###################
#### DEQ #############################
for seed in $(seq 1 100)
   do
       python main.py --config celeba_ls_opt.yml --model DiffusionInversion --exp celeba-orig --image_folder celeba-inverse-t20-parallel --doc celeba-orig-ddim --ls_opt --timesteps 20 --ni --method default --lambda1 1 --lambda2 0 --lambda3 0 --seed $seed --tau 0.1 --use_wandb --no_augmentation --pg_steps 1 
   done

#### Baseline ########################
for seed in $(seq 1 100)
   do
       python main.py --config celeba_ls_opt_seq.yml --model DiffusionInversion --exp celeba-orig --image_folder celeba-inverse-t20-seq --doc celeba-orig-ddim --ls_opt --timesteps 20 --ni --method default --lambda1 1 --lambda2 0 --lambda3 0 --seed $seed --use_wandb --no_augmentation
   done

############ LSUN Bedroom ############# (Use similar command for LSUN Church)
#### DEQ ##############################
for seed in $(seq 1 100)
    do
        python main.py --config bedroom_ls_opt.yml --model DiffusionInversion --exp lsun_orig --image_folder lsun_bedroom_images --doc lsun_orig --ls_opt --timesteps 10 --ni --method default --lambda1 1 --lambda2 0 --lambda3 0 --seed $seed --tau 0.1 --no_augmentation --pg_steps 1 --use_pretrained --use_wandb
    done

#### Baseline #########################
for seed in $(seq 1 100)
    do
        python main.py --config bedroom_ls_opt_seq.yml --model DiffusionInversion --exp lsun_orig --image_folder lsun_bedroom_images_t10 --doc lsun_orig --ls_opt --timesteps 10 --ni --method default --seed $seed --no_augmentation --use_pretrained --use_wandb
    done
