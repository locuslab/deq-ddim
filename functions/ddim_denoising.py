from logging import DEBUG
import torch
from functions.utils import get_ortho_mat
import wandb
import numpy as np

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def fp_implicit_iters(x, seq, model, b, logger=None, print_logs=False, **kwargs):
    with torch.no_grad():
        bsz = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        image_dim = x.shape

        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(bsz) * i).to(x.device)
            next_t = (torch.ones(bsz) * j).to(x.device)

            beta_t = b.index_select(0, t.long()).view(-1, 1, 1, 1)

            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())

            # import pdb; pdb.set_trace()
            # print(1 - beta_t, at/at_next)

            xt = xs[-1].to('cuda')
            et = model(xt, t)
            xt_coeff = 1/(1 - beta_t).sqrt()
            
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))

            et_coeff = (1 - at_next).sqrt() - ((1 - at)/(1 - beta_t)).sqrt()

            xt_next = xt_coeff * xt + et_coeff * et
            xs.append(xt_next.to('cpu'))

            log_dict = {
                    "alpha at": torch.mean(at),
                    "alpha at_next": torch.mean(at_next),
                    "xt": torch.norm(xt.reshape(image_dim[0], -1), -1).mean(),
                    "xt_next": torch.norm(xt_next.reshape(image_dim[0], -1), -1).mean(),
                    #"coeff x0": c1.squeeze().mean(),
                    "coeff et": et_coeff.squeeze().mean(),
                    "prediction et": torch.norm(et.reshape(image_dim[0], -1), -1).mean(),
                }
            
            if logger is not None:
                if i % 50 == 0 or i < 50:
                    log_dict["samples"] = [wandb.Image(xt_next[i]) for i in range(10)]
                logger(log_dict)
            if print_logs:
                print(i, j, log_dict)
    return xs, x0_preds

def compute_single_step(bsz, x, b, model, xs, image_dim, x0_preds, cur_iter):
    i = cur_iter
    j = cur_iter - 1

    t = (torch.ones(bsz) * i).to(x.device)
    next_t = (torch.ones(bsz) * j).to(x.device)

    beta_t = b.index_select(0, t.long()).view(-1, 1, 1, 1)

    at = compute_alpha(b, t.long())
    at_next = compute_alpha(b, next_t.long())

    # import pdb; pdb.set_trace()
    # print(1 - beta_t, at/at_next)

    xt = xs[-1].to('cuda')
    et = model(xt, t)
    xt_coeff = 1/(1 - beta_t).sqrt()
    
    x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
    x0_preds.append(x0_t.to('cpu'))

    et_coeff = (1 - at_next).sqrt() - ((1 - at)/(1 - beta_t)).sqrt()

    xt_next = xt_coeff * xt + et_coeff * et
    xs.append(xt_next.to('cpu'))

    log_dict = {
            "alpha at": torch.mean(at),
            "alpha at_next": torch.mean(at_next),
            "xt": torch.norm(xt.reshape(image_dim[0], -1), -1).mean(),
            "xt_next": torch.norm(xt_next.reshape(image_dim[0], -1), -1).mean(),
            #"coeff x0": c1.squeeze().mean(),
            "coeff et": et_coeff.squeeze().mean(),
            "prediction et": torch.norm(et.reshape(image_dim[0], -1), -1).mean(),
        }
    return xt_next, log_dict


def fp_implicit_iters_v2(x, seq, model, b, logger=None, print_logs=False, **kwargs):
    with torch.no_grad():
        bsz = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        image_dim = x.shape

        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(bsz) * i).to(x.device)
            next_t = (torch.ones(bsz) * j).to(x.device)

            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())

            xt = xs[-1].to('cuda')
            et = model(xt, t)

            et_coeff = (1 - at_next).sqrt() - ((at_next * (1 - at))/at).sqrt()
            xt_coeff = (at_next/at).sqrt()

            xt_next = xt_coeff * xt + et_coeff * et
            xs.append(xt_next.to('cpu'))

            # import pdb; pdb.set_trace()
            log_dict = {
                    "alpha at": torch.mean(at),
                    "alpha at_next": torch.mean(at_next),
                    "xt": torch.norm(xt.reshape(image_dim[0], -1), -1).mean(),
                    "xt_next": torch.norm(xt_next.reshape(image_dim[0], -1), -1).mean(),
                    #"coeff x0": c1.squeeze().mean(),
                    "coeff et": et_coeff.squeeze().mean(),
                    "prediction et": torch.norm(et.reshape(image_dim[0], -1), -1).mean(),
                }
            
            if logger is not None:
                if i % 50 == 0 or i < 50:
                    log_dict["samples"] = [wandb.Image(xt_next[i]) for i in range(min(len(xt_next), 10))]
                logger(log_dict)
            if print_logs:
                print(i, j, log_dict)
    return xs, x0_preds

def compute_single_step_v2(bsz, x, b, model, xs, image_dim, x0_preds, cur_iter):
    i = cur_iter
    j = cur_iter - 1

    t = (torch.ones(bsz) * i).to(x.device)
    next_t = (torch.ones(bsz) * j).to(x.device)

    at = compute_alpha(b, t.long())
    at_next = compute_alpha(b, next_t.long())

    # import pdb; pdb.set_trace()
    # print(1 - beta_t, at/at_next)

    xt = xs[-1].to('cuda')
    et = model(xt, t)

    et_coeff = (1 - at_next).sqrt() - ((at_next * (1 - at))/at).sqrt()
    xt_coeff = (at_next/at).sqrt()

    xt_next = xt_coeff * xt + et_coeff * et
    xs.append(xt_next.to('cpu'))

    log_dict = {
            "alpha at": torch.mean(at),
            "alpha at_next": torch.mean(at_next),
            "xt": torch.norm(xt.reshape(image_dim[0], -1), -1).mean(),
            "xt_next": torch.norm(xt_next.reshape(image_dim[0], -1), -1).mean(),
            #"coeff x0": c1.squeeze().mean(),
            "coeff et": et_coeff.squeeze().mean(),
            "prediction et": torch.norm(et.reshape(image_dim[0], -1), -1).mean(),
        }
    return xt_next, log_dict

def compute_multi_step(cur_seq, seq_next, seq, bz, x, b, model, xs, image_dim, x0_preds, xT, cur_iter, xst):
    #import pdb; pdb.set_trace()
    T = len(cur_seq)
    t = torch.tensor(cur_seq[::-1]).repeat(bz).to(x.device)
    next_t = torch.tensor(seq_next[::-1]).repeat(bz).to(x.device)

    #all_t = torch.tensor(list(seq)[::-1] + [-1]).to(x.device)

    at = compute_alpha(b, t.long())
    at_next = compute_alpha(b, next_t.long())

    #all_at =  compute_alpha(b, all_t.long())

    alpha_ratio = (at_next/at[0]).sqrt() #(all_at/all_at[0]).sqrt()
    xT_coeff = alpha_ratio #[1:]

    xt = xs[-1].to('cuda')
    et = model(xt, t)
    
    et_coeff2 = (1 - at_next).sqrt() - (((1 - at)*at_next)/at).sqrt()

    et_updated = (1 / at_next.sqrt()) * et_coeff2 * et

    et_split = torch.split(et_updated, T)
    et_cumsum = torch.stack(et_split).cumsum(dim=1)
    et_prevsum = torch.cat(torch.unbind(et_cumsum))
    #et_prevsum = et_updated.cumsum(dim=0)

    # if cur_iter == 0:
    #     import pdb; pdb.set_trace()
    xt_next = xT_coeff * torch.repeat_interleave(xT, T, dim=0).to(x.device) + at_next.sqrt() * et_prevsum
    #x_exp, _ = compute_single_step_v2(bz, x, b, model, xst, image_dim, x0_preds, cur_iter=cur_seq[cur_iter])

    log_dict = {
                "alpha at": torch.mean(at),
                "alpha at_next": torch.mean(at_next),
                "xt": torch.norm(xt.reshape(image_dim[0], -1), -1).mean(),
                "xt_next": torch.norm(xt_next.reshape(image_dim[0], -1), -1).mean(),
                #"coeff x0": c1.squeeze().mean(),
                #"coeff et": et_coeff.squeeze().mean(),
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

        # max_steps = seq[-1]
        # diff = seq[-1] - seq[-2]

        x_final = {}

        xT = x
        xst = [x]

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
        
        #for i in range(max_steps, -1, -diff):
        #for i in range(len(seq)-1, -1, -1):
        for i in range(20, -1, -1):
            x_next_, log_dict = compute_multi_step(cur_seq, seq_next, seq, bz, x, b, model, xs, image_dim, x0_preds, xT, cur_iter=i, xst=xst)
            
            #x_prev = torch.cat((xT, x_next_[:-1]), dim=0)
            x_prev = torch.zeros_like(all_xt)
            x_prev[xT_idx] = xT
            x_prev[prev_idx] = x_next_[next_idx] 

            xs.append(x_prev.to('cpu'))
            if i in save_timesteps and save_last:
                x_final[i].append(x_next_[gather_idx].to('cpu'))

            if logger is not None:
                if i % 5 == 0 or i < 5:
                    log_dict["samples"] = [wandb.Image(x_next_.to('cpu')[ts]) for ts in plot_timesteps]
                logger(log_dict)
    if len(save_timesteps) == 1:
        return x_final[0], x0_preds

    return x_final, x0_preds
