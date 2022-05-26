#!/bin/bash
python main.py --config cifar10.yml --model Diffusion --exp cifar10-orig-fid --image_folder /project_data/datasets/apokle_exp/ddim-sampling-images-05-11/cifar10-orig-fid/image_samples/time-cifar10-seq-t1000-v2 --doc cifar10 --sample --fid --timesteps 1000 --eta 0 --ni --method simple-seq --use_pretrained
#python main.py --config cifar10.yml --model Diffusion --exp cifar10-orig-fid --image_folder samples-cifar10-and-t1000 --doc cifar10 --sample --fid --timesteps 1000 --eta 0 --ni --method anderson --use_pretrained
#python main.py --config cifar10.yml --model Diffusion --exp cifar10-orig-fid --image_folder time-cifar10-and-t1000-v2 --doc cifar10 --sample --fid --timesteps 1000 --eta 0 --ni --method anderson --use_pretrained
#python main.py --config cifar10_batch.yml --model Diffusion --exp cifar10-orig-fid --image_folder fid-cifar10-seq-batch-t1000 --doc cifar10 --sample --fid --timesteps 1000 --eta 0 --ni --method simple-seq --use_pretrained
