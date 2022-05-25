from logging import DEBUG
import torch
from functions.utils import get_ortho_mat
import wandb
import numpy as np

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def compute_multi_step(model, all_xT, et_coeff, et_prevsum_coeff, T, t, xs, image_dim, bz):
    # import pdb; pdb.set_trace()
    with torch.no_grad():
        xt = xs[-1].to('cuda')
        et = model(xt, t)

        et_updated = et_coeff * et

        # et_split = torch.split(et_updated, T)
        # et_cumsum = torch.stack(et_split).cumsum(dim=1)
        # et_prevsum = torch.cat(torch.unbind(et_cumsum))

        #et_prevsum_faster = et_updated.reshape(bz, T, image_dim[1], image_dim[2], image_dim[3]).cumsum(dim=1).view(-1, image_dim[1], image_dim[2], image_dim[3])

        #import pdb; pdb.set_trace()
        # Idea 2
        et_cumsum_all = et_updated.cumsum(dim=0)
        idx = torch.arange(T-1, et_cumsum_all.shape[0]-1, T)
        prev_cumsum = et_cumsum_all[idx]
        # diff = torch.cat((torch.zeros((T, image_dim[1], image_dim[2], image_dim[3]), device=xt.device), torch.repeat_interleave(prev_cumsum, T,  dim=0)), dim=0)
        # et_prevsum = et_cumsum_all - diff

        # Idea 3
        et_prevsum = et_cumsum_all
        # for i in range(T, et_prevsum.shape[0], T):
        #     et_prevsum[i:i+T] -= prev_cumsum[i//T-1]
        
        # idea 4
        et_prevsum[T:] -= torch.repeat_interleave(prev_cumsum, T,  dim=0)

        #et_prevsum = et_updated.cumsum(dim=0)

        xt_next = all_xT + et_prevsum_coeff * et_prevsum

        log_dict = {
                    "xt": torch.norm(xt.reshape(image_dim[0], -1), -1).mean(),
                    "xt_next": torch.norm(xt_next.reshape(image_dim[0], -1), -1).mean(),
                    "prediction et": torch.norm(et.reshape(image_dim[0], -1), -1).mean(),
                }
    return xt_next, log_dict

def fp_implicit_iters_parallel(x, seq, model, b, logger=None, print_logs=False, save_last=True, **kwargs):
    with torch.no_grad():
        bz = x.size(0)
        cur_seq = list(seq)
        seq_next = [-1] + list(seq[:-1])

        x0_preds = []
        image_dim = x.shape

        x_final = {}
        xT = x

        # first stack xs on top of each other. Should be size of max_steps
        all_xt = torch.repeat_interleave(x, len(cur_seq), dim=0).to(x.device)
        xs = [all_xt]

        #save_timesteps = [0] + [idx for idx in range(9, 100, 20)]
        #save_timesteps = [0, 5]
        save_timesteps = [0]
        for tstep in save_timesteps:
            x_final[tstep] = []
        
        gather_idx = [idx for idx in range(len(cur_seq) - 1, len(all_xt), len(cur_seq))]
        xT_idx = [idx for idx in range(0, len(all_xt), len(cur_seq))]
        next_idx = [idx for idx in range(len(all_xt)) if idx not in range(len(cur_seq)-1, len(all_xt), len(cur_seq))]
        prev_idx = [idx + 1 for idx in next_idx]

        plot_timesteps = []
        for batch_idx in range(bz):
            plot_timesteps += [n + batch_idx * len(cur_seq) for n in range(0, len(seq), 10)] + [(batch_idx + 1) * len(cur_seq) - n for n in range(5, 0, -1)] 
        
        T = len(cur_seq)
        t = torch.tensor(cur_seq[::-1]).repeat(bz).to(x.device)
        next_t = torch.tensor(seq_next[::-1]).repeat(bz).to(x.device)

        at = compute_alpha(b, t.long())
        at_next = compute_alpha(b, next_t.long())

        alpha_ratio = (at_next/at[0]).sqrt() 
        xT_coeff = alpha_ratio
        all_xT = xT_coeff * torch.repeat_interleave(xT, T, dim=0).to(x.device)

        et_coeff2 = (1 - at_next).sqrt() - (((1 - at)*at_next)/at).sqrt()

        et_coeff = (1 / at_next.sqrt()) * et_coeff2

        et_prevsum_coeff = at_next.sqrt()

        max_steps = 40
        #for i in range(len(seq)-1, -1, -1):
        for i in range(max_steps, -1, -1):
            # import pdb; pdb.set_trace()
            x_next_, log_dict = compute_multi_step(model, all_xT, et_coeff, et_prevsum_coeff, T, t, xs, image_dim, bz)
            
            x_prev = torch.zeros_like(all_xt)
            x_prev[xT_idx] = xT
            x_prev[prev_idx] = x_next_[next_idx] 

            xs.append(x_prev.to('cpu'))
            if i in save_timesteps and save_last:
                x_final[i].append(x_next_[gather_idx].to('cpu'))

            if logger is not None:
                #if len(xs) >= 2:
                log_dict["residual"] = (xs[-1].to('cpu') - xs[-2].to('cpu')).norm().item()
                #    log_dict["residual"] = (xs[-1].to('cpu') - xs[-2].to('cpu')).norm().item()
                if i % 20 == 0 or i < 10:
                    log_dict["samples"] = [wandb.Image(x_next_.to('cpu')[ts]) for ts in plot_timesteps]
                logger(log_dict)

    if len(save_timesteps) == 1:
        return x_final[0], x0_preds

    return x_final, x0_preds
