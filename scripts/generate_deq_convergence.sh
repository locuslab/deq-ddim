#!/bin/bash

### Generate samples with DEQ + Anderson solver
python main.py --config cifar10.yml --model Diffusion --exp cifar10-orig-fid --image_folder samples-cifar10-and-t1000-long-new --doc cifar10 --sample --fid --timesteps 1000 --eta 0 --ni --method anderson --use_pretrained
python main.py --config celeba.yml --model Diffusion --exp celeba-orig --image_folder time-celeba-and-t500 --doc celeba-orig-ddim --sample --fid --timesteps 500 --eta 0 --ni --method anderson
python main.py --config bedroom.yml --model Diffusion --exp bedroom-orig-fid --image_folder time-bedroom-and-t25 --doc bedroom --sample --fid --timesteps 25 --eta 0 --ni --method anderson --use_pretrained
python main.py --config church.yml --model Diffusion --exp church-orig-fid --image_folder time-church-and-t25 --doc church --sample --fid --timesteps 25 --eta 0 --ni --method anderson --use_pretrained

### Generate samples with DDIM
python main.py --config cifar10.yml --model Diffusion --exp cifar10-orig-fid --image_folder fid-cifar10-seq-t1000-v2 --doc cifar10 --sample --fid --timesteps 1000 --eta 0 --ni --method simple-seq --use_pretrained
python main.py --config celeba.yml --model Diffusion --exp celeba-orig --image_folder fid-celeba-seq-t500 --doc celeba-orig-ddim --sample --fid --timesteps 500 --eta 0 --ni --method simple-seq
python main.py --config bedroom.yml --model Diffusion --exp bedroom-orig-fid --image_folder fid-bedroom-seq-t25 --doc bedroom --sample --fid --timesteps 25 --eta 0 --ni --method simple-seq --use_pretrained
python main.py --config church.yml --model Diffusion --exp church-orig-fid --image_folder fid-church-seq-batch-t25 --doc church --sample --fid --timesteps 25 --eta 0 --ni --method simple-seq --use_pretrained