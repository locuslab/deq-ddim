#!/bin/bash
#SBATCH --job-name=geom-cifar10
#SBATCH --output=cifar10-ddim-geom.out
#SBATCH --gres=gpu:4
#SBATCH --mem=10GB
#SBATCH --exclude=locus-1-1,locus-0-25
#SBATCH --time=72:00:00
#SBATCH -p short
#SBATCH --oversubscribe
source ~/miniconda3/etc/profile.d/conda.sh
conda activate smldenv
python main.py --config cifar10_geometric.yml --exp exp --doc cifar10-pair-geometric --ni --model GeometricDiffusion
