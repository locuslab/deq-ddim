from logging import log
import torch
import wandb
import time 

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def compute_multi_step(xt, all_xT, model, et_coeff, et_prevsum_coeff, T, t, image_dim, xT, **kwargs):
    with torch.no_grad():
        xt_all = torch.zeros_like(all_xT)
        xt_all[kwargs['xT_idx']] = xT
        xt_all[kwargs['prev_idx']] = xt[kwargs['next_idx']]

        xt = xt_all.to('cuda')

        et = model(xt, t)
        et_updated = et_coeff * et
        et_cumsum_all = et_updated.cumsum(dim=0)
        idx = torch.arange(T-1, et_cumsum_all.shape[0]-1, T)
        prev_cumsum = et_cumsum_all[idx]

        et_prevsum = et_cumsum_all
        et_prevsum[T:] -= torch.repeat_interleave(prev_cumsum, T,  dim=0)

        xt_next = all_xT + et_prevsum_coeff * et_prevsum
        log_dict = {
            "xt": torch.norm(xt.reshape(image_dim[0], -1), -1).mean(),
            "xt_next": torch.norm(xt_next.reshape(image_dim[0], -1), -1).mean(),
            "prediction et": torch.norm(et.reshape(image_dim[0], -1), -1).mean(),
        }
    return xt_next.view(xt.shape[0], -1), log_dict

@torch.no_grad()
def anderson(f, x0, X, F, H, y, args, m=3, lam=1e-3, max_iter=15, tol=1e-2, beta = 1.0, logger=None):
    """ Anderson acceleration for fixed point iteration. """
    with torch.no_grad():
        bsz, ch, h0, w0 = x0.shape
        
        t1 = time.time()
        #X = torch.zeros(bsz, m, ch * h0 * w0, dtype=x0.dtype, device=x0.device)
        #F = torch.zeros(bsz, m, ch * h0 * w0, dtype=x0.dtype, device=x0.device)

        X[:,0] = x0.view(bsz, -1)
        F[:,0], _ = f(xt=x0.view(x0.shape), **args)

        X[:,1] = F[:,0].view(bsz, -1)
        F[:,1], _ = f(xt=F[:,0].view(x0.shape), **args)

        #H = torch.zeros(bsz, m+1, m+1, dtype=x0.dtype, device=x0.device)
        H[:,0,1:] = H[:,1:,0] = 1
        #y = torch.zeros(bsz, m+1, 1, dtype=x0.dtype, device=x0.device)
        y[:,0] = 1
        
        t2 = time.time()
        print("Intial set up", t2-t1)
        time_logger = {
            "setup": t2 - t1,
            "bmm": 0,
            "solve": 0,
            "forward call-unet": 0,
            "total_time_per_iter": 0
        }

        iter_count = 0
        log_metrics = {}
        res = []
        norm_res = []
        for k in range(2, max_iter):
            n_ = min(k, m)
            G = F[:,:n_]-X[:,:n_]
            
            t3 = time.time()
            H[:,1:n_+1,1:n_+1] = torch.bmm(G,G.transpose(1,2)) + lam*torch.eye(n_, dtype=x0.dtype,device=x0.device)[None]
            t4 = time.time()
            alpha = torch.solve(y[:,:n_+1], H[:,:n_+1,:n_+1])[0][:, 1:n_+1, 0]   # (bsz x n)
            t5 = time.time()
            X[:,k%m] = beta * (alpha[:,None] @ F[:,:n_])[:,0] + (1-beta)*(alpha[:,None] @ X[:,:n_])[:,0]
            F[:,k%m], log_metrics = f(xt=X[:,k%m].view(x0.shape), **args)
            t6 = time.time()

            residual = (F[:,k%m] - X[:,k%m]).norm().item()
            normalized_residual = (F[:,k%m] - X[:,k%m]).norm().item()/(1e-5 + F[:,k%m].norm()).item()

            t7 = time.time()
            res.append(residual)
            norm_res.append(normalized_residual)
            
            time_logger["bmm"] += (t4 - t3)
            time_logger["solve"] += (t5 - t4)
            time_logger["forward call-unet"] += (t6 - t5)
            time_logger["total_time_per_iter"] += (t7 - t3)
            iter_count += 1

            ### TODO: break out early for norm_res
            if (norm_res[-1] < tol):
                print("Breaking out early at {}".format(k))
                break

            #print("{}/{} Residual {} tol {} ".format(k, max_iter, res[-1], tol))
            if logger is not None:
                log_metrics["residual"] = residual
                log_metrics["normalized_residual"] = normalized_residual

                log_metrics["alpha"] = torch.norm(alpha, dim=-1).mean()
                log_metrics["samples"] = [wandb.Image(X[:, k%m].view_as(x0).to('cpu')[ts]) for ts in args['plot_timesteps']]
                log_metrics["setup"] = time_logger['setup']
                log_metrics["total_time_per_iter"] = time_logger['total_time_per_iter'] / iter_count
                log_metrics["total_time"] = t7 - t1
                for key, val in time_logger.items():
                    if key not in log_metrics: 
                        log_metrics[f"avg-{key}"] = val / iter_count
                        log_metrics[key] = val
                log_metrics["perc_time_forward_call"] = time_logger["forward call-unet"] * 100 / time_logger["total_time_per_iter"]
                logger(log_metrics)
    x_eq = X[:,k%m].view_as(x0)[args['gather_idx']].to('cpu')
    return x_eq

# @torch.no_grad()
# def anderson_single(f, x0, X, F, H, y, args, m=1, lam=1e-3, max_iter=15, tol=1e-2, beta = 1.0, logger=None):
#     """ Anderson acceleration for fixed point iteration. """
#     with torch.no_grad():
#         #import pdb; pdb.set_trace()
#         bsz, ch, h0, w0 = x0.shape
        
#         X[:,0] = x0.view(bsz, -1)
#         F[:,0], _ = f(xt=x0.view((bsz, ch, h0, w0)), **args)

#         H[:,0,1:] = H[:,1:,0] = 1
#         y[:,0] = 1
        
#         #import pdb; pdb.set_trace()

#         log_metrics = {}
#         res = []
#         for k in range(1, max_iter):
#             n_ = min(k, m)
#             G = F[:,:n_]-X[:,:n_]
#             H[:,1:n_+1,1:n_+1] = torch.bmm(G,G.transpose(1,2)) + lam*torch.eye(n_, dtype=x0.dtype,device=x0.device)[None]
#             alpha = torch.solve(y[:,:n_+1], H[:,:n_+1,:n_+1])[0][:, 1:n_+1, 0]   # (bsz x n)
            
#             X[:,k%m] = beta * (alpha[:,None] @ F[:,:n_])[:,0] + (1-beta)*(alpha[:,None] @ X[:,:n_])[:,0]
#             F[:,k%m], log_metrics = f(xt=X[:,k%m].view((bsz, ch, h0, w0)), **args)

#             residual = (F[:,k%m] - X[:,k%m]).norm().item()
#             normalized_residual = (F[:,k%m] - X[:,k%m]).norm().item()/(1e-5 + F[:,k%m].norm()).item()

#             res.append(residual)

#             if (res[-1] < tol):
#                 print("Breaking out early at {}".format(k))
#                 break

#             #print("{}/{} Residual {} tol {} ".format(k, max_iter, res[-1], tol))
#             if logger is not None:
#                 log_metrics["residual"] = residual
#                 log_metrics["normalized_residual"] = normalized_residual

#                 log_metrics["alpha"] = torch.norm(alpha, dim=-1).mean()
#                 #if k % 20 == 0 or k < 10:
#                 log_metrics["samples"] = [wandb.Image(X[:, k%m].view_as(x0).to('cpu')[ts]) for ts in args['plot_timesteps']]
#                 logger(log_metrics)
#     x_eq = X[:,k%m].view_as(x0)[args['gather_idx']].to('cpu')
#     return x_eq

def fp_implicit_iters_anderson(x, model, b, args=None, additional_args=None, logger=None, print_logs=False, save_last=True, **kwargs):
    with torch.no_grad():
        # bz = x.size(0)
        # cur_seq = list(seq)
        # seq_next = [-1] + list(seq[:-1])

        x0_preds = []
        image_dim = x.shape

        #x_final = {}
        #xT = x

        # first stack xs on top of each other. Should be size of max_steps
        
        #save_timesteps = [0] + [idx for idx in range(9, 100, 20)]
        #save_timesteps = [0, 5]
        # save_timesteps = [0]
        # for tstep in save_timesteps:
        #     x_final[tstep] = []
        
        all_xt = args['all_xt']
        # gather_idx = [idx for idx in range(len(cur_seq) - 1, len(all_xt), len(cur_seq))]
        # xT_idx = [idx for idx in range(0, len(all_xt), len(cur_seq))]
        # next_idx = [idx for idx in range(len(all_xt)) if idx not in range(len(cur_seq)-1, len(all_xt), len(cur_seq))]
        # prev_idx = [idx + 1 for idx in next_idx]

        # plot_timesteps = []
        # for batch_idx in range(bz):
        #     plot_timesteps += [n + batch_idx * len(cur_seq) for n in range(0, len(seq), 10)] + [(batch_idx + 1) * len(cur_seq) - n for n in range(5, 0, -1)] 
        
        # T = len(cur_seq)
        # t = torch.tensor(cur_seq[::-1]).repeat(bz).to(x.device)
        # next_t = torch.tensor(seq_next[::-1]).repeat(bz).to(x.device)

        # at = compute_alpha(b, t.long())
        # at_next = compute_alpha(b, next_t.long())

        # alpha_ratio = (at_next/at[0]).sqrt() 
        # all_xT = alpha_ratio * torch.repeat_interleave(xT, T, dim=0).to(x.device)

        # et_coeff2 = (1 - at_next).sqrt() - (((1 - at)*at_next)/at).sqrt()

        # et_coeff = (1 / at_next.sqrt()) * et_coeff2

        # et_prevsum_coeff = at_next.sqrt()

        additional_args["model"] = model
        additional_args["image_dim"] = image_dim

        # additional_args = {
        #     "model" : model,
        #     "all_xT": all_xT, 
        #     "et_coeff": et_coeff,
        #     "et_prevsum_coeff": et_prevsum_coeff, 
        #     "T" : T, 
        #     "t" : t,
        #     "image_dim": image_dim, 
        #     "bz": bz,
        #     "plot_timesteps": plot_timesteps,
        #     "gather_idx": gather_idx,
        #     "xT_idx": xT_idx,
        #     "prev_idx": prev_idx,
        #     "next_idx": next_idx,
        #     "xT": xT
        # }
        # x_final = anderson(compute_multi_step, all_xt, args['X'], args['F'], args['H'], args['y'], 
        #                         additional_args, m=args['m'], lam=1e-3, max_iter=500, tol=1e-3, beta = 1.0, logger=logger)
        x_final = anderson(compute_multi_step, all_xt, args['X'], args['F'], args['H'], args['y'], 
                                 additional_args, m=args['m'], lam=1e-3, max_iter=15, tol=1e-3, beta = 1.0, logger=logger)
    return x_final, x0_preds

