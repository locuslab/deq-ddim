#!/bin/bash
#python main.py --config bedroom.yml --model Diffusion --exp /project_data/datasets/apokle_exp/ddim-sampling-images-05-11/bedroom-orig-fid --image_folder time-bedroom-seq-t25 --doc bedroom --sample --fid --timesteps 25 --eta 0 --ni --method simple-seq --use_pretrained
#python main.py --config church_batch.yml --model Diffusion --exp church-orig-fid --image_folder fid-church-seq-batch-t25 --doc church --sample --fid --timesteps 50 --eta 0 --ni --method simple-seq --use_pretrained
#python main.py --config bedroom.yml --model Diffusion --exp /project_data/datasets/apokle_exp/ddim-sampling-images-05-11/bedroom-orig-fid --image_folder fid-bedroom-and-t25-new --doc bedroom --sample --fid --timesteps 25 --eta 0 --ni --method anderson --use_pretrained
python main.py --config bedroom.yml --model Diffusion --exp bedroom-orig-fid --image_folder time-bedroom-and-t25 --doc bedroom --sample --fid --timesteps 25 --eta 0 --ni --method anderson --use_pretrained
