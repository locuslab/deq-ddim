#!/bin/bash
#python main.py --config church.yml --model Diffusion --exp church-orig-fid --image_folder /project_data/datasets/apokle_exp/ddim-sampling-images-05-11/church-orig-fid/image_samples/time-church-seq-t50/ --doc church --sample --fid --timesteps 25 --eta 0 --ni --method simple-seq --use_pretrained
#python main.py --config church_batch.yml --model Diffusion --exp church-orig-fid --image_folder fid-church-seq-batch-t25 --doc church --sample --fid --timesteps 50 --eta 0 --ni --method simple-seq --use_pretrained
#python main.py --config church.yml --model Diffusion --exp church-orig-fid --image_folder fid-church-and-t25 --doc church --sample --fid --timesteps 25 --eta 0 --ni --method anderson --use_pretrained
python main.py --config church.yml --model Diffusion --exp church-orig-fid --image_folder time-church-and-t25 --doc church --sample --fid --timesteps 25 --eta 0 --ni --method anderson --use_pretrained
