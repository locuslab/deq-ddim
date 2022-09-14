from logging import log
import pdb
from turtle import pd
from functions.ddim_anderson import anderson
import torch
import wandb
import numpy as np
import torch.autograd as autograd
from functions.latent_space_opt_anderson import anderson

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    if beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    return betas

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def compute_multi_step_ddpm(xt, model, all_xT, et_coeff, et_prevsum_coeff, noise_t, T, t, xT, image_dim, **kwargs):
    xt_in = xt[kwargs['next_idx']]

    et = model(xt_in, t)
    
    et_updated = et_coeff * et + noise_t ### Additional DDPM noise
    et_cumsum_all = et_updated.cumsum(dim=0)
    
    et_prevsum = et_cumsum_all

    idx = torch.arange(T-1, et_cumsum_all.shape[0]-1, T)
    if len(idx) > 0:
        prev_cumsum = et_cumsum_all[idx]
        et_prevsum[T:] -= torch.repeat_interleave(prev_cumsum, T,  dim=0)

    xt_next = all_xT + et_prevsum_coeff * et_prevsum
    xt_all = torch.zeros_like(xt)
    xt_all[kwargs['xT_idx']] = xT
    xt_all[kwargs['prev_idx']] = xt_next
    xt_all = xt_all.cuda()

    return xt_all

# x: source image
# model: diffusion model 
# args: required args
def find_source_noise(x, model, args, logger=None):
    T = args['T']
    at = args['at']
    at_next = args['at_next']
    alpha_ratio = args['alpha_ratio']
    _,C,H,W = x.shape

    xT = x[0].view(1, C, H, W)
    all_xT = alpha_ratio * torch.repeat_interleave(xT, T, dim=0).to(x.device)
    

    sigma_t = args['eta'] * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
    et_coeff2 = (1 - at_next - sigma_t**2).sqrt() - (((1 - at)*at_next)/at).sqrt()
    et_coeff = (1 / at_next.sqrt()) * et_coeff2 

    all_noiset = torch.randn_like(all_xT).to(all_xT.device)       
    noise_t = (1 / at_next.sqrt()) * sigma_t * all_noiset

    et_prevsum_coeff = at_next.sqrt()
    with torch.no_grad():
        for _ in range(T, -1, -1):
            all_xt = compute_multi_step_ddpm(xt=x, model=model, et_coeff=et_coeff,
                            et_prevsum_coeff=et_prevsum_coeff, noise_t=noise_t, image_dim=x.shape, all_xT=all_xT, xT=xT, **args)
            if logger is not None:
                logger({"generated images": [wandb.Image(all_xt[i].view((3, 32, 32))) for i in list(range(0, T, T//10)) + [-5, -4, -3, -2, -1]]})
            x = all_xt
    return all_xt

class DEQDDPMLatentSpaceOpt(object):
    def __init__(self):
        self.hook =  None

    def find_source_noise_deq(self, x, model, args, anderson_params=None, tau=0.5, pg_steps=5, logger=None):
        T = args['T']
        at = args['at']
        at_next = args['at_next']
        alpha_ratio = args['alpha_ratio']
        all_noiset = args['all_noiset']
        _,C,H,W = x.shape

        if anderson_params is None:
            anderson_params = {
                "m": 3,
                "lambda": 1e-3,
                "max_anderson_iters": 15,
                "tol": 0.01,
                "beta": 1
            }
        xT = x[0].view(1, C, H, W)
        all_xT = alpha_ratio * torch.repeat_interleave(xT, T, dim=0).to(x.device)

        if args['eta'] == 0:
            raise ValueError("Running in DDPM mode but eta is 0")

        sigma_t = args['eta'] * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()

        et_coeff2 = (1 - at_next - sigma_t**2).sqrt() - (((1 - at)*at_next)/at).sqrt()
        
        noise_t = (1 / at_next.sqrt()) * sigma_t * all_noiset

        et_coeff = (1 / at_next.sqrt()) * et_coeff2

        et_prevsum_coeff = at_next.sqrt()
        
        args['model'] = model
        args['xT'] = xT
        args['all_xT'] = all_xT
        args['et_coeff'] = et_coeff
        args['et_prevsum_coeff'] = et_prevsum_coeff
        args['image_dim'] = x.shape
        args['noise_t'] = noise_t

        with torch.no_grad():
            x_eq = anderson(compute_multi_step_ddpm, x, args, m=3, lam=1e-3, max_iter=50, tol=1e-3, beta = 1.0, logger=None)
            if logger is not None:
                logger({"generated images": [wandb.Image(x_eq[i].view((3, H, W))) for i in list(range(0, T, T//10)) + [-5, -4, -3, -2, -1]]})
        
        torch.cuda.empty_cache()

        x_eq = x_eq.requires_grad_()
        for _ in range(pg_steps):
            x_eq = (1 - tau) * x_eq + tau * compute_multi_step_ddpm(xt=x_eq, **args)
        return x_eq

def get_additional_lt_opt_args(all_xt, seq, betas, batch_size):
    from functions.ddim_anderson import compute_alpha

    cur_seq = list(seq)
    seq_next = [-1] + list(seq[:-1])

    gather_idx = [idx for idx in range(len(cur_seq), len(all_xt), len(cur_seq)+2)]
    xT_idx = [idx for idx in range(0, len(all_xt), len(cur_seq)+1)]
    next_idx = [idx for idx in range(len(all_xt)) if idx not in gather_idx]
    prev_idx = [idx + 1 for idx in next_idx]

    T = len(cur_seq)
    t = torch.tensor(cur_seq[::-1]).repeat(batch_size).to(all_xt.device)
    next_t = torch.tensor(seq_next[::-1]).repeat(batch_size).to(all_xt.device)

    at = compute_alpha(betas, t.long())
    at_next = compute_alpha(betas, next_t.long())

    alpha_ratio = (at_next/at[0]).sqrt() 
    
    additional_args = {
        "T" : T, 
        "t" : t,
        "bz": batch_size,
        "gather_idx": gather_idx,
        "xT_idx": xT_idx,
        "prev_idx": prev_idx,
        "next_idx": next_idx,
        "alpha_ratio": alpha_ratio,
        'at_next': at_next,
        'at': at
    }
    return additional_args

### Rest of the code is just for the sake of validation
def fp_validity_check(all_xt, model, additional_args, max_steps=100, logger=None):
    with torch.no_grad():
        all_xt = find_source_noise(all_xt, model, additional_args, logger=logger)
    return all_xt

def anderson_validity_check(x, model, args, additional_args, max_steps=100, logger=None):
    deq_ls_opt = DEQDDPMLatentSpaceOpt()
    with torch.no_grad():
        all_xt = deq_ls_opt.find_source_noise_deq(args['all_xt'], model, additional_args, logger=logger)
    return all_xt
