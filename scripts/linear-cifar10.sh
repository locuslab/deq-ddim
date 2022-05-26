#!/bin/bash
#SBATCH --job-name=const-0.01-rand-ncunet-cifar10
#SBATCH --output=cifar10-cunet-modified-const-0.01-schedule-rand-xT.out
#SBATCH --gres=gpu:1
#SBATCH --mem=12GB
#SBATCH --exclude=locus-1-1,locus-1-9,locus-0-25
#SBATCH --time=7-12:00:00
#SBATCH -p bigmem
source ~/miniconda3/etc/profile.d/conda.sh
conda activate smldenv
python main.py --config cifar10_modified_copy.yml --exp cifar10-pair-modified-linear-schedule --doc const-rand-e-ncunet --ni --model Diffusion --resume_training
#python main.py --config cifar10_modifiedv3.yml --exp cifar10-pair-modified-const-0.005-schedule --doc v3-const-0.005-rand-e-ncunet-sp --ni --model Diffusion --resume_training
#python main.py --config cifar10_modified.yml --exp cifar10-pair-modified-const-0.005-schedule --doc cifar10-pair-modified-const-0.005-schedule-rand-xT --resume_training --ni
