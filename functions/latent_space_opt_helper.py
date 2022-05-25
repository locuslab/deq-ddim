from logging import log
import torch
import wandb
import time 
import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import numpy as np

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

def compute_multi_step(xt, all_xT, model, et_coeff, et_prevsum_coeff, T, t, image_dim, xT, **kwargs):

    et = model(xt, t)
    
    et_updated = et_coeff * et
    et_cumsum_all = et_updated.cumsum(dim=0)
    
    et_prevsum = et_cumsum_all

    idx = torch.arange(T-1, et_cumsum_all.shape[0]-1, T)
    if len(idx) > 0:
        prev_cumsum = et_cumsum_all[idx]
        et_prevsum[T:] -= torch.repeat_interleave(prev_cumsum, T,  dim=0)

    xt_next = all_xT + et_prevsum_coeff * et_prevsum
    log_dict = {
        "xt": torch.norm(xt.reshape(image_dim[0], -1), -1).mean(),
        "xt_next": torch.norm(xt_next.reshape(image_dim[0], -1), -1).mean(),
        "prediction et": torch.norm(et.reshape(image_dim[0], -1), -1).mean(),
    }
    xt_all = torch.zeros_like(all_xT)
    xt_all[kwargs['xT_idx']] = xT
    xt_all[kwargs['prev_idx']] = xt_next[kwargs['next_idx']]
    xt_all = xt_all.cuda()

    return xt_next[-1], xt_all, log_dict

# x: source image
# model: diffusion model 
# args: required arfs like 
# def find_source_noise(x, model, args):
#     return compute_multi_step(xt=x, model=model, image_dim=x.shape, **args)

def find_source_noise(x, model, args, logger=None):
    T = args['T']
    B, C, H, W = x.shape
    at = args['at']
    at_next = args['at_next']
    alpha_ratio = args['alpha_ratio']

    xT = x[0].view(1, C, H, W)

    all_xT = alpha_ratio * torch.repeat_interleave(xT, T, dim=0).to(x.device)

    et_coeff2 = (1 - at_next).sqrt() - (((1 - at)*at_next)/at).sqrt()

    et_coeff = (1 / at_next.sqrt()) * et_coeff2

    et_prevsum_coeff = at_next.sqrt()

    # with torch.no_grad():
    for _ in range(T, -1, -1):
        xt_next, all_xt, log_dict = compute_multi_step(xt=x, all_xT=all_xT, model=model, et_coeff=et_coeff,
                        et_prevsum_coeff=et_prevsum_coeff, image_dim=x.shape, xT=xT, **args)
        if logger is not None:
            logger({"generated images": [wandb.Image(all_xt[i].view((3, 32, 32))) for i in list(range(0, T, T//10)) + [-5, -4, -3, -2, -1]]})
        x = all_xt
    return xt_next, all_xt, log_dict

def get_additional_lt_opt_args(all_xt, seq, betas, batch_size):
    from functions.ddim_anderson import compute_alpha

    cur_seq = list(seq)
    seq_next = [-1] + list(seq[:-1])

    gather_idx = [idx for idx in range(len(cur_seq) - 1, len(all_xt), len(cur_seq))]
    xT_idx = [idx for idx in range(0, len(all_xt), len(cur_seq))]
    next_idx = [idx for idx in range(len(all_xt)) if idx not in range(len(cur_seq)-1, len(all_xt), len(cur_seq))]
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
        # for _ in range(max_steps, -1, -1):
        xt_next, all_xt, log_dict = find_source_noise(all_xt, model, additional_args, logger=logger)
            # if logger is not None:
            #     logger({"generated images": [wandb.Image(all_xt[i].view((3, 32, 32))) for i in list(range(0, 100, 10)) + [95, 96, 97, 98, 99]]})
    return xt_next, all_xt