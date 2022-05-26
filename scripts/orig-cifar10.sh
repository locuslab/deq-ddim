#!/bin/bash
#SBATCH --job-name=orig-cifar10
#SBATCH --output=cifar10-orig.out
#SBATCH --gres=gpu:1
#SBATCH --mem=12GB
#SBATCH --exclude=locus-1-1,locus-0-25
#SBATCH --time=72:00:00
#SBATCH --oversubscribe
source ~/miniconda3/etc/profile.d/conda.sh
conda activate smldenv
python main.py --config cifar10.yml --model Diffusion --exp cifar10-orig-exp --doc cifar10-orig --ni --resume_training
