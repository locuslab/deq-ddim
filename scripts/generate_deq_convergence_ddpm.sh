#!/bin/bash
### Generate samples with DDIM
# CIFAR10
python main.py --config cifar10.yml --model Diffusion --exp cifar10-orig-fid --image_folder fid-cifar10-seq-t20-eta=0.2 --doc cifar10 --sample --fid --timesteps 20 --eta 0.2 --ni --method simple-seq --use_pretrained --skip_type quad
python main.py --config cifar10.yml --model Diffusion --exp cifar10-orig-fid --image_folder fid-cifar10-seq-t20-eta-0.5 --doc cifar10 --sample --fid --timesteps 20 --eta 0.5 --ni --method simple-seq --use_pretrained --skip_type quad
python main.py --config cifar10.yml --model Diffusion --exp cifar10-orig-fid --image_folder fid-cifar10-seq-t20-eta-1 --doc cifar10 --sample --fid --timesteps 20 --eta 1 --ni --method simple-seq --use_pretrained --skip_type quad

python main.py --config cifar10.yml --model Diffusion --exp cifar10-orig-fid --image_folder fid-cifar10-seq-t50-eta=0.2 --doc cifar10 --sample --fid --timesteps 50 --eta 0.2 --ni --method simple-seq --use_pretrained --skip_type quad
python main.py --config cifar10.yml --model Diffusion --exp cifar10-orig-fid --image_folder fid-cifar10-seq-t50-eta-0.5 --doc cifar10 --sample --fid --timesteps 50 --eta 0.5 --ni --method simple-seq --use_pretrained --skip_type quad
python main.py --config cifar10.yml --model Diffusion --exp cifar10-orig-fid --image_folder fid-cifar10-seq-t50-eta-1 --doc cifar10 --sample --fid --timesteps 50 --eta 1 --ni --method simple-seq --use_pretrained --skip_type quad

# CelebA
python main.py --config celeba.yml --model Diffusion --exp celeba-orig --image_folder fid-celeba-seq-t20-eta-0.2 --doc celeba-orig-ddim --sample --fid --timesteps 20 --eta 0.2 --ni --method simple-seq
python main.py --config celeba.yml --model Diffusion --exp celeba-orig --image_folder fid-celeba-seq-t20-eta-0.5 --doc celeba-orig-ddim --sample --fid --timesteps 20 --eta 0.5 --ni --method simple-seq
python main.py --config celeba.yml --model Diffusion --exp celeba-orig --image_folder fid-celeba-seq-t20-eta-1 --doc celeba-orig-ddim --sample --fid --timesteps 20 --eta 1 --ni --method simple-seq

python main.py --config celeba.yml --model Diffusion --exp celeba-orig --image_folder fid-celeba-seq-t50-eta-0.2 --doc celeba-orig-ddim --sample --fid --timesteps 50 --eta 0.2 --ni --method simple-seq
python main.py --config celeba.yml --model Diffusion --exp celeba-orig --image_folder fid-celeba-seq-t50-eta-0.5 --doc celeba-orig-ddim --sample --fid --timesteps 50 --eta 0.5 --ni --method simple-seq
python main.py --config celeba.yml --model Diffusion --exp celeba-orig --image_folder fid-celeba-seq-t50-eta-1 --doc celeba-orig-ddim --sample --fid --timesteps 50 --eta 1 --ni --method simple-seq

# LSUN Bedroom
python main.py --config bedroom.yml --model Diffusion --exp bedroom-orig-fid --image_folder fid-bedroom-t20-seq-eta-0.2 --doc bedroom --sample --fid --timesteps 20 --eta 0.2 --ni --method simple-seq --use_pretrained

# LSUN Church
python main.py --config church.yml --model Diffusion --exp church-orig-fid --image_folder fid-church-t20-seq-eta-0.2 --doc church --sample --fid --timesteps 20 --eta 0.2 --ni --method simple-seq --use_pretrained

### Generate samples with DEQ-DDIM + Anderson

python main.py --config celeba.yml --model Diffusion --exp celeba-orig --image_folder ddim-eta-0.2-celeba-seq-t50 --doc celeba-orig-ddim --sample --fid --timesteps 50 --eta 0.2 --ni --method ddpm
python main.py --config celeba.yml --model Diffusion --exp celeba-orig --image_folder ddim-eta-0.5-celeba-seq-t50 --doc celeba-orig-ddim --sample --fid --timesteps 50 --eta 0.5 --ni --method ddpm
python main.py --config celeba.yml --model Diffusion --exp celeba-orig --image_folder ddim-eta-1-celeba-seq-t50 --doc celeba-orig-ddim --sample --fid --timesteps 50 --eta 1 --ni --method ddpm

python main.py --config celeba.yml --model Diffusion --exp celeba-orig --image_folder ddim-eta-0.2-celeba-seq-t20 --doc celeba-orig-ddim --sample --fid --timesteps 20 --eta 0.2 --ni --method ddpm
python main.py --config celeba.yml --model Diffusion --exp celeba-orig --image_folder ddim-eta-0.5-celeba-seq-t20 --doc celeba-orig-ddim --sample --fid --timesteps 20 --eta 0.5 --ni --method ddpm
python main.py --config celeba.yml --model Diffusion --exp celeba-orig --image_folder ddim-eta-1-celeba-seq-t20 --doc celeba-orig-ddim --sample --fid --timesteps 20 --eta 1 --ni --method ddpm

python main.py --config cifar10.yml --model Diffusion --exp cifar10-orig-fid --image_folder ddim-eta-0.2-cifar10-t20 --doc cifar10 --sample --fid --timesteps 20 --eta 0.2 --ni --method ddpm --use_pretrained --skip_type quad
python main.py --config cifar10.yml --model Diffusion --exp cifar10-orig-fid --image_folder ddim-eta-0.5-cifar10-t20 --doc cifar10 --sample --fid --timesteps 20 --eta 0.5 --ni --method ddpm --use_pretrained --skip_type quad
python main.py --config cifar10.yml --model Diffusion --exp cifar10-orig-fid --image_folder ddim-eta-1-cifar10-t20 --doc cifar10 --sample --fid --timesteps 20 --eta 1 --ni --method ddpm --use_pretrained --skip_type quad

python main.py --config cifar10.yml --model Diffusion --exp cifar10-orig-fid --image_folder ddim-eta-1-cifar10-t50 --doc cifar10 --sample --fid --timesteps 50 --eta 1 --ni --method ddpm --use_pretrained --skip_type quad
python main.py --config cifar10.yml --model Diffusion --exp cifar10-orig-fid --image_folder ddim-eta-0.5-cifar10-t50 --doc cifar10 --sample --fid --timesteps 50 --eta 0.5 --ni --method ddpm --use_pretrained --skip_type quad
python main.py --config cifar10.yml --model Diffusion --exp cifar10-orig-fid --image_folder ddim-eta-0.2-cifar10-t50 --doc cifar10 --sample --fid --timesteps 50 --eta 0.2 --ni --method ddpm --use_pretrained --skip_type quad

python main.py --config bedroom.yml --model Diffusion --exp bedroom-orig-fid --image_folder ddim-eta-0.2-bedroom-seq-t20 --doc bedroom --sample --fid --timesteps 20 --eta 0.2 --ni --method ddpm --use_pretrained
python main.py --config church.yml --model Diffusion --exp church-orig-fid --image_folder ddim-eta-0.2-church-t20 --doc church --sample --fid --timesteps 20 --eta 0.2 --ni --method ddpm --use_pretrained
