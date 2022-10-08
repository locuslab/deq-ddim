#!/bin/bash

########### CIFAR10 ##################
#### DEQ #############################

#for i in $(seq 1 100)
#do
#    python main.py --config cifar10_ls_opt.yml --model Diffusion --exp cifar10-orig-exp --image_folder cifar10-inverse-t1000-parallel --doc cifar10-orig --ls_opt --timesteps 1000 --ni --method default --lambda1 1 --lambda2 0 --lambda3 0 --seed $i --tau 0.1 --use_wandb --no_augmentation --pg_steps 1
#done
#### Baseline ########################
#for i in $(seq 1 100)
#do
#    python main.py --config cifar10_ls_opt_seq.yml --model Diffusion --exp cifar10-orig-exp --image_folder cifar10-inverse-t1000-seq --doc cifar10-orig --ls_opt --timesteps 1000 --ni --method default --lambda1 1 --lambda2 0 --lambda3 0 --seed $i --use_wandb --no_augmentation
#done

########### CelebA ###################
#### DEQ #############################
#for seed in $(seq 1 10)
#   do
#       python main.py --config celeba_ls_opt.yml --model Diffusion --exp celeba-orig --image_folder celeba-ift-t20 --doc celeba-orig-ddim --ls_opt --timesteps 20 --ni --method default --lambda1 1 --lambda2 0 --lambda3 0 --seed $seed --tau 0.1 --use_wandb --no_augmentation --pg_steps 1 
#   done

python main.py --config celeba.yml --model Diffusion --exp celeba-orig --image_folder xt-init-deq-trial --doc celeba-orig-ddim --sample --fid --timesteps 50 --ni --method anderson --use_wandb
#for seed in $(seq 1 100)
#   do
#        python main.py --config celeba_ls_opt.yml --model Diffusion --exp celeba-orig --image_folder celeba-recons-t-1000 --doc celeba-orig-ddim --recons --timesteps 1000 --ni --method default --seed $seed --use_wandb
#   done

#for seed in $(seq 1 10)
#   do
#       python main.py --config celeba_ls_opt2.yml --model Diffusion --exp celeba-orig --image_folder celeba-ift-t20-lr-5e-3 --doc celeba-orig-ddim --ls_opt --timesteps 20 --ni --method default --lambda1 1 --lambda2 0 --lambda3 0 --seed $seed --tau 0.1 --use_wandb --no_augmentation --pg_steps 1 
#   done

#### Baseline ########################
#for seed in $(seq 1 100)
#   do
#       python main.py --config celeba_ls_opt_seq.yml --model Diffusion --exp celeba-orig --image_folder celeba-inverse-t20-seq --doc celeba-orig-ddim --ls_opt --timesteps 20 --ni --method default --lambda1 1 --lambda2 0 --lambda3 0 --seed $seed --use_wandb --no_augmentation
#   done

############ LSUN Bedroom ############# (Use similar command for LSUN Church)
#### DEQ ##############################
#for seed in $(seq 1 100)
#    do
#        python main.py --config bedroom_ls_opt.yml --model Diffusion --exp lsun_orig --image_folder lsun_bedroom_images --doc lsun_orig --ls_opt --timesteps 10 --ni --method default --lambda1 1 --lambda2 0 --lambda3 0 --seed $seed --tau 0.1 --no_augmentation --pg_steps 1 --use_pretrained --use_wandb
#    done

#### Baseline #########################
#for seed in $(seq 1 100)
#    do
#        python main.py --config bedroom_ls_opt_seq.yml --model Diffusion --exp lsun_orig --image_folder lsun_bedroom_images_t10 --doc lsun_orig --ls_opt --timesteps 10 --ni --method default --seed $seed --no_augmentation --use_pretrained --use_wandb
#    done
