from logging import log
from turtle import pd
import torch
import wandb
import numpy as np
import torch.autograd as autograd

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

def compute_multi_step(xt, model, all_xT, et_coeff, et_prevsum_coeff, T, t, xT, image_dim, **kwargs):
    xt_in = xt[kwargs['next_idx']]
    et = model(xt_in, t)
    
    et_updated = et_coeff * et
    et_cumsum_all = et_updated.cumsum(dim=0)
    
    et_prevsum = et_cumsum_all

    idx = torch.arange(T-1, et_cumsum_all.shape[0]-1, T)
    if len(idx) > 0:
        prev_cumsum = et_cumsum_all[idx]
        et_prevsum[T:] -= torch.repeat_interleave(prev_cumsum, T,  dim=0)

    xt_next = all_xT + et_prevsum_coeff * et_prevsum
    # log_dict = {
    #     "xt": torch.norm(xt.reshape(image_dim[0], -1), -1).mean(),
    #     "xt_next": torch.norm(xt_next.reshape(image_dim[0], -1), -1).mean(),
    #     "prediction et": torch.norm(et.reshape(image_dim[0], -1), -1).mean(),
    # }
    xt_all = torch.zeros_like(xt)
    xt_all[kwargs['xT_idx']] = xT
    xt_all[kwargs['prev_idx']] = xt_next
    xt_all = xt_all.cuda()

    return xt_all

def simple_anderson(f, x0, m=3, lam=1e-3, threshold=30, eps=1e-3, stop_mode='rel', beta=1.0, **kwargs):
    """ Anderson acceleration for fixed point iteration. """
    bsz, ch, h0, w0 = x0.shape
    alternative_mode = 'rel' if stop_mode == 'abs' else 'abs'
    X = torch.zeros(bsz, m, ch * h0 * w0, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, ch * h0 * w0, dtype=x0.dtype, device=x0.device)

    X[:,0], F[:,0] = x0.reshape(bsz, -1), f(x0).reshape(bsz, -1)
    X[:,1], F[:,1] = F[:,0], f(F[:,0].reshape_as(x0)).reshape(bsz, -1)
    
    H = torch.zeros(bsz, m+1, m+1, dtype=x0.dtype, device=x0.device)
    H[:,0,1:] = H[:,1:,0] = 1
    y = torch.zeros(bsz, m+1, 1, dtype=x0.dtype, device=x0.device)
    y[:,0] = 1

    trace_dict = {'abs': [],
                  'rel': []}
    lowest_dict = {'abs': 1e8,
                   'rel': 1e8}
    lowest_step_dict = {'abs': 0,
                        'rel': 0}

    for k in range(2, threshold):
        n = min(k, m)
        G = F[:,:n]-X[:,:n]
        H[:,1:n+1,1:n+1] = torch.bmm(G,G.transpose(1,2)) + lam*torch.eye(n, dtype=x0.dtype,device=x0.device)[None]
        alpha = torch.solve(y[:,:n+1], H[:,:n+1,:n+1])[0][:, 1:n+1, 0]   # (bsz x n)
        
        X[:,k%m] = beta * (alpha[:,None] @ F[:,:n])[:,0] + (1-beta)*(alpha[:,None] @ X[:,:n])[:,0]
        F[:,k%m] = f(X[:,k%m].reshape_as(x0)).reshape(bsz, -1)
        gx = (F[:,k%m] - X[:,k%m]).view_as(x0)
        abs_diff = gx.norm().item()
        rel_diff = abs_diff / (1e-5 + F[:,k%m].norm().item())
        diff_dict = {'abs': abs_diff,
                     'rel': rel_diff}
        trace_dict['abs'].append(abs_diff)
        trace_dict['rel'].append(rel_diff)
        
        for mode in ['rel', 'abs']:
            if diff_dict[mode] < lowest_dict[mode]:
                if mode == stop_mode: 
                    lowest_xest, lowest_gx =  X[:,k%m].view_as(x0).clone().detach(), gx.clone().detach()
                lowest_dict[mode] = diff_dict[mode]
                lowest_step_dict[mode] = k

        if trace_dict[stop_mode][-1] < eps:
            for _ in range(threshold-1-k):
                trace_dict[stop_mode].append(lowest_dict[stop_mode])
                trace_dict[alternative_mode].append(lowest_dict[alternative_mode])
            break
            
    # out = {"result": lowest_xest,
    #        "lowest": lowest_dict[stop_mode],
    #        "nstep": lowest_step_dict[stop_mode],
    #        "prot_break": False,
    #        "abs_trace": trace_dict['abs'],
    #        "rel_trace": trace_dict['rel'],
    #        "eps": eps,
    #        "threshold": threshold}
    X = F = None
    return lowest_xest

@torch.no_grad()
def anderson(f, x0, args, m=3, lam=1e-3, max_iter=50, tol=1e-3, beta = 1.0, logger=None):
    """ Anderson acceleration for fixed point iteration. """
    with torch.no_grad():
        bsz, ch, h0, w0 = x0.shape
        
        X = torch.zeros(bsz, m, ch * h0 * w0, dtype=x0.dtype, device=x0.device)
        F = torch.zeros(bsz, m, ch * h0 * w0, dtype=x0.dtype, device=x0.device)

        X[:,0] = x0.view(bsz, -1)
        F[:,0] = f(xt=x0.view(x0.shape), **args).view(bsz, -1)

        X[:,1] = F[:,0].view(bsz, -1)
        F[:,1] = f(xt=F[:,0].view(x0.shape), **args).view(bsz, -1)

        H = torch.zeros(bsz, m+1, m+1, dtype=x0.dtype, device=x0.device)
        H[:,0,1:] = H[:,1:,0] = 1
        y = torch.zeros(bsz, m+1, 1, dtype=x0.dtype, device=x0.device)
        y[:,0] = 1

        iter_count = 0
        log_metrics = {}
        res = []
        norm_res = []
        for k in range(2, max_iter):
            n_ = min(k, m)
            G = F[:,:n_]-X[:,:n_]
            
            H[:,1:n_+1,1:n_+1] = torch.bmm(G,G.transpose(1,2)) + lam*torch.eye(n_, dtype=x0.dtype,device=x0.device)[None]
            alpha = torch.solve(y[:,:n_+1], H[:,:n_+1,:n_+1])[0][:, 1:n_+1, 0]   # (bsz x n)
            X[:,k%m] = beta * (alpha[:,None] @ F[:,:n_])[:,0] + (1-beta)*(alpha[:,None] @ X[:,:n_])[:,0]
            F[:,k%m] = f(xt=X[:,k%m].view(x0.shape), **args).view(bsz, -1)

            residual = (F[:,k%m] - X[:,k%m]).norm().item()
            normalized_residual = (F[:,k%m] - X[:,k%m]).norm().item()/(1e-5 + F[:,k%m].norm()).item()

            res.append(residual)
            norm_res.append(normalized_residual)
            iter_count += 1

            if (norm_res[-1] < tol):
                print("Breaking out early at {}".format(k))
                break

            #print("{}/{} Residual {} tol {} ".format(k, max_iter, res[-1], tol))
            if logger is not None:
                log_metrics["residual"] = residual
                log_metrics["normalized_residual"] = normalized_residual

                log_metrics["alpha"] = torch.norm(alpha, dim=-1).mean()
                #if k % 20 == 0 or k < 10:
                log_metrics["samples"] = [wandb.Image(X[:, k%m].view_as(x0).to('cpu')[ts]) for ts in args['plot_timesteps']]
                logger(log_metrics)
    x_eq = X[:,k%m].view_as(x0)#[args['gather_idx']].to('cpu')
    X = F = None
    print("Abs residual ", min(res), " Rel residual ", min(norm_res))
    return x_eq

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
    
    et_coeff2 = (1 - at_next).sqrt() - (((1 - at)*at_next)/at).sqrt()

    et_coeff = (1 / at_next.sqrt()) * et_coeff2

    et_prevsum_coeff = at_next.sqrt()
    with torch.no_grad():
        for _ in range(T, -1, -1):
            all_xt = compute_multi_step(xt=x, model=model, et_coeff=et_coeff,
                            et_prevsum_coeff=et_prevsum_coeff, image_dim=x.shape, all_xT=all_xT, xT=xT, **args)
            if logger is not None:
                logger({"generated images": [wandb.Image(all_xt[i].view((3, 32, 32))) for i in list(range(0, T, T//10)) + [-5, -4, -3, -2, -1]]})
            x = all_xt
    return all_xt

class DEQLatentSpaceOpt(object):
    def __init__(self):
        self.hook =  None

    def find_source_noise_deq(self, x, model, args, anderson_params=None, tau=0.5, pg_steps=5, logger=None):
        T = args['T']
        at = args['at']
        at_next = args['at_next']
        alpha_ratio = args['alpha_ratio']
        _,C,H,W = x.shape

        if anderson_params is None:
            anderson_params = {
                "m": 3,
                "lambda": 1e-3,
                "max_anderson_iters": 30,
                "tol": 0.01,
                "beta": 1
            }
        xT = x[0].view(1, C, H, W)
        all_xT = alpha_ratio * torch.repeat_interleave(xT, T, dim=0).to(x.device)

        et_coeff2 = (1 - at_next).sqrt() - (((1 - at)*at_next)/at).sqrt()

        et_coeff = (1 / at_next.sqrt()) * et_coeff2

        et_prevsum_coeff = at_next.sqrt()
        
        args['model'] = model
        args['xT'] = xT
        args['all_xT'] = all_xT
        args['et_coeff'] = et_coeff
        args['et_prevsum_coeff'] = et_prevsum_coeff
        args['image_dim'] = x.shape

        with torch.no_grad():
            x_eq = anderson(compute_multi_step, x, args, m=3, lam=1e-3, max_iter=50, tol=1e-2, beta = 0.9, logger=None)
            if logger is not None:
                logger({"generated images": [wandb.Image(x_eq[i].view((3, 32, 32))) for i in list(range(0, T, T//10)) + [-5, -4, -3, -2, -1]]})
        
        torch.cuda.empty_cache()

        x_eq.requires_grad_()
        for _ in range(pg_steps):
            x_eq = (1 - tau) * x_eq + tau * compute_multi_step(xt=x_eq, **args)
        return x_eq

        # x_eq.requires_grad_()
        # new_z1 = compute_multi_step(x_eq.view(x.shape), **args)
        
        # def backward_hook(grad):
        #     if self.hook is not None:
        #         self.hook.remove()
        #         torch.cuda.synchronize()
        #     x_back = simple_anderson(lambda y: autograd.grad(new_z1, x_eq, y, retain_graph=True)[0] + grad, torch.zeros_like(grad), 
        #                 m=3, lam=1e-3, max_iter=50, tol=1e-2, beta = 0.8, logger=None)
        #     return x_back
        # self.hook = new_z1.register_hook(backward_hook)
        # return new_z1

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

def anderson_validity_check(all_xt, model, additional_args, max_steps=100, logger=None):
    deq_ls_opt = DEQLatentSpaceOpt()
    with torch.no_grad():
        all_xt = deq_ls_opt.find_source_noise_deq(all_xt, model, additional_args, logger=logger)
    return all_xt