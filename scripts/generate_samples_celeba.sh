#!/bin/bash
python main.py --config celeba.yml --model Diffusion --exp celeba-orig --image_folder time-celeba-and-t500 --doc celeba-orig-ddim --sample --fid --timesteps 500 --eta 0 --ni --method anderson
#python main.py --config celeba_batch.yml --model Diffusion --exp celeba-orig --image_folder fid-celeba-seq-batch-t500 --doc celeba-orig-ddim --sample --fid --timesteps 500 --eta 0 --ni --method simple-seq
#python main.py --config celeba.yml --model Diffusion --exp celeba-orig --image_folder /project_data/datasets/apokle_exp/ddim-sampling-images-05-11/celeba-orig/image_samples/time-celeba-seq-t500 --doc celeba-orig-ddim --sample --fid --timesteps 500 --eta 0 --ni --method simple-seq

#python main.py --config celeba_batch.yml --model Diffusion --exp celeba-orig --image_folder fid-celeba-seq-t50 --doc celeba-orig-ddim --sample --fid --timesteps 50 --eta 0 --ni --method simple-seq

#python main.py --config celeba.yml --model Diffusion --exp celeba-orig --image_folder sample-celeba-and-t500-long --doc celeba-orig-ddim --sample --fid --timesteps 500 --eta 0 --ni --method anderson
#python main.py --config celeba.yml --model Diffusion --exp celeba-orig --image_folder sample-celeba-and-t50-long --doc celeba-orig-ddim --sample --fid --timesteps 50 --eta 0 --ni --method anderson
#python main.py --config celeba.yml --model Diffusion --exp celeba-orig --image_folder sample-celeba-and-t10-long --doc celeba-orig-ddim --sample --fid --timesteps 10 --eta 0 --ni --method anderson

#python main.py --config church.yml --model Diffusion --exp church-orig-fid --image_folder sample-church-seq-t10-long --doc church --sample --fid --timesteps 10 --eta 0 --ni --method anderson --use_pretrained
#python main.py --config church.yml --model Diffusion --exp church-orig-fid --image_folder sample-church-seq-t50-long --doc church --sample --fid --timesteps 50 --eta 0 --ni --method anderson --use_pretrained

#python main.py --config bedroom.yml --model Diffusion --exp bedroom-orig-fid --image_folder sample-bedroom-and-t500-long --doc bedroom --sample --fid --timesteps 500 --eta 0 --ni --method anderson --use_pretrained
#python main.py --config bedroom.yml --model Diffusion --exp bedroom-orig-fid --image_folder sample-bedroom-and-t50-long --doc bedroom --sample --fid --timesteps 50 --eta 0 --ni --method anderson --use_pretrained
#python main.py --config bedroom.yml --model Diffusion --exp bedroom-orig-fid --image_folder sample-bedroom-and-t10-long --doc bedroom --sample --fid --timesteps 10 --eta 0 --ni --method anderson --use_pretrained

