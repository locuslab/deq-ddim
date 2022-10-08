# Deep Equilibrium Approaches to Diffusion Models

This codebase has been adapted largely from the repository of Denoising Diffusion Implicit Models (DDIM) by Song. et. al. 2020 (https://arxiv.org/abs/2010.02502) (Note: we include the original MIT license that belongs to the authors of prior work (Song. et. al.) in this codebase.)

## Getting Started 

Create conda environment and install all packages from `requirements.txt`
```
conda create --name <environment_name> --file requirements.txt
conda activate <environment_name>
```

If you are working with CelebA 64x64 dataset, please download pretrained checkpoint from https://github.com/ermongroup/ddim

## Running the Experiments
The code has been tested on PyTorch 1.11.

### Sampling

#### Sampling for FID evaluation

General command to sample with a DEQ or DDIM is:
```
python main.py --config {DATASET}.yml --model Diffusion --exp {PROJECT_PATH} --image_folder {IMG_FOLDER} --doc {DOCUMENTATIOIN_FOLDER} --sample --fid --timesteps 1000 --eta 0 --ni --method {METHOD} --use_pretrained
```
where 
- `ETA` controls the scale of the variance (0 is DDIM, and 1 is one type of DDPM). (We use 0 for all examples)
- `STEPS` controls how many timesteps used in the process.
- `MODEL_NAME` finds the pre-trained checkpoint according to its inferred path.
- `METHOD` Use 'anderson' for DEQ and 'simple-seq' for DDIM

Please check [generate_deq_convergence.sh](scripts/generate_deq_convergence.sh) for sampling commands for all the datasets.

Example command for sampling with DEQ from CIFAR10
```
python main.py --config cifar10.yml --model Diffusion --exp cifar10-orig-fid --image_folder samples-cifar10-and-t1000-long-new --doc cifar10 --sample --fid --timesteps 1000 --eta 0 --ni --method anderson --use_pretrained
```
The `--use_pretrained` option will automatically load the model according to the dataset for CIFAR10, LSUN Bedrooms and Churches. We use DDPM models for all datasets except CelebA. Please download CelebA 64x64 pretrained model from https://github.com/ermongroup/ddim

### Training DEQ for Model Inversion
```
 python main.py --config {DATASET}_ls_opt.yml --model DiffusionInversion --exp {PROJECT_PATH} --image_folder {IMAGE_FOLDER} --doc {MODEL_NAME} --ls_opt --timesteps {STEPS} --ni --method {METHOD} --lambda1 1 --lambda2 0 --lambda3 0 --seed $i --tau {DAMPING_FACTOR} --use_wandb --no_augmentation --pg_steps {PG_STEPS}
```
where
- `ETA` controls the scale of the variance (0 is DDIM, and 1 is one type of DDPM). (We use 0 for all examples)
- `STEPS` controls how many timesteps used in the process.
- `MODEL_NAME` finds the pre-trained checkpoint according to its inferred path.
- `METHOD` Use 'anderson' for DEQ and 'simple-seq' for DDIM
- `PG_STEPS` is the number of iterations while computing phantom gradients. We set this value to 1.
- `DAMPING_FACTOR` is the value of damping used in phantom gradients. We set this to 0.1.

Please check [invert_models.sh](scripts/invert_models.sh) for sampling commands for all the datasets.
