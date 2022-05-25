import torch
from functions.utils import get_ortho_mat
import wandb
import time 

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def compute_cumprod_alpha(beta):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0)
    return a

def get_alpha_at_index(a, t):
    a_val = a.index_select(0, t + 1).view(-1, 1, 1, 1)
    return a_val

def generalized_steps(x, seq, model, b, logger=None, print_logs=False, **kwargs):
    #with torch.no_grad():
    bsz = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]
    image_dim = x.shape
    
    #iter_count = 0
    
    # time_logger = {
    #     "setup": 0,
    #     "forward call-unet": 0,
    #     "post-processing" : 0,
    #     "total_time_per_iter": 0,
    # }

    # t0 = time.time()

    alpha = compute_cumprod_alpha(b)

    for i, j in zip(reversed(seq), reversed(seq_next)):
        #t1 = time.time()
        t = (torch.ones(bsz) * i).to(x.device)
        next_t = (torch.ones(bsz) * j).to(x.device)
        
        #at = compute_alpha(b, t.long())
        #at_next = compute_alpha(b, next_t.long())
        
        at = get_alpha_at_index(alpha, t.long())
        at_next = get_alpha_at_index(alpha, next_t.long())
        
        xt = xs[-1].to('cuda')
        #t2 = time.time()
        et = model(xt, t)
        #t3 = time.time()
        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
        #x0_preds.append(x0_t.to('cpu'))
        x0_preds.append(x0_t)
        c1 = (
            kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
        )
        c2 = ((1 - at_next) - c1 ** 2).sqrt()
        noise_t = torch.randn_like(x)
        xt_next = at_next.sqrt() * x0_t + c1 * noise_t + c2 * et
        #xs.append(xt_next.to('cpu'))
        xs.append(xt_next)
        #t4 = time.time()

        # time_logger["setup"] += (t2 - t1)
        # time_logger["forward call-unet"] += (t3 - t2)
        # time_logger["post-processing"] += (t4 - t3)
        # time_logger["total_time_per_iter"] += (t4 - t1)
        # iter_count += 1

        # log_dict = {
        #         "alpha at": torch.mean(at),
        #         "alpha at_next": torch.mean(at_next),
        #         "xt": torch.norm(xt.reshape(image_dim[0], -1), -1).mean(),
        #         "xt_next": torch.norm(xt_next.reshape(image_dim[0], -1), -1).mean(),
        #         "coeff x0": c1.squeeze().mean(),
        #         "coeff et": c2.squeeze().mean(),
        #         "prediction et": torch.norm(et.reshape(image_dim[0], -1), -1).mean(),
        #         "noise": torch.norm(noise_t.reshape(image_dim[0], -1), -1).mean(),
        #         "setup": time_logger["setup"]/iter_count,
        #         "forward call-unet": time_logger["forward call-unet"],
        #         "avg-forward call-unet": time_logger["forward call-unet"]/iter_count,
        #         "avg-post-processing": time_logger["post-processing"]/iter_count,
        #         "total_time_per_iter": time_logger['total_time_per_iter'] / iter_count,
        #         "perc_time_forward_call": time_logger["forward call-unet"] * 100 / time_logger["total_time_per_iter"],
        #         "total_time": t4 - t0
        #     }
        
        # if logger is not None:
        #     if i % 50 == 0 or i < 50:
        #         log_dict["samples"] = [wandb.Image(xt_next[i]) for i in range(min(xt_next.shape[0], 10))]
        #     logger(log_dict)
        # elif print_logs:
        #     print(i, j, log_dict)
    return xs, x0_preds

def generalized_steps_fp_ddim(x, seq, model, b, logger=None, print_logs=False, **kwargs):
    with torch.no_grad():
        B = x.size(0)

        x0_preds = []
        xs = [x]

        image_dim = x.shape
        T = seq[-1]
        diff = seq[-1] - seq[-2]

        for i in range(T, -1, -diff):
            t = (torch.ones(B) * i).to(x.device)

            next_t = max(-1, i-diff)
            next_t = (torch.ones(B) * next_t).to(x.device)

            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            
            xt = xs[-1].to('cuda')
            et = model(xt, t)
            
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            
            et_coeff = ((1 - at_next)).sqrt()

            xt_next = at_next.sqrt() * x0_t + et_coeff * et

            log_dict = {
                    "alpha at": torch.mean(at),
                    "alpha at_next": torch.mean(at_next),
                    "xt": torch.norm(xt.reshape(image_dim[0], -1), -1).mean(),
                    "xt_next": torch.norm(xt_next.reshape(image_dim[0], -1), -1).mean(),
                    "coeff x0": at_next.sqrt().squeeze().mean(),
                    "coeff et": et_coeff.squeeze().mean(),
                    "prediction et": torch.norm(et.reshape(image_dim[0], -1), -1).mean(),
                }
            
            if logger is not None:
                if i % 50 == 0 or i < 50:
                    log_dict["samples"] = [wandb.Image(xt_next[i]) for i in range(10)]
                logger(log_dict)
            elif print_logs:
                print(t, max(-1, i-diff), log_dict)

            xs.append(xt_next.view(image_dim).to('cpu'))

    return xs, x0_preds

def generalized_steps_fixed_U_ddim(x, seq, model, b, logger=None, print_logs=False, **kwargs):
    with torch.no_grad():
        B = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]

        image_dim = x.shape
        U = get_ortho_mat(x.dtype, x.shape, x.device, method='bases')
        noise_t = torch.randn_like(x.view(image_dim[0], -1))

        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(B) * i).to(x.device)
            next_t = (torch.ones(B) * j).to(x.device)

            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            et = model(xt, t)
            
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            
            sigma_t = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            et_coeff = ((1 - at_next) - sigma_t ** 2).sqrt()

            #xt_next = (at_next.sqrt() * x0_t).view(image_dim[0], -1) + c1.squeeze().unsqueeze(1) * noise + c2.squeeze().unsqueeze(1) * et.view(image_dim[0], -1)
            
            xt_next = at_next.sqrt() * x0_t + et_coeff * et + sigma_t * noise_t.view(image_dim)
            noise_t = torch.matmul(noise_t, U)

            log_dict = {
                    "alpha at": torch.mean(at),
                    "alpha at_next": torch.mean(at_next),
                    "xt": torch.norm(xt.reshape(image_dim[0], -1), -1).mean(),
                    "xt_next": torch.norm(xt_next.reshape(image_dim[0], -1), -1).mean(),
                    "coeff x0": at_next.sqrt().squeeze().mean(),
                    "coeff et": et_coeff.squeeze().mean(),
                    "prediction et": torch.norm(et.reshape(image_dim[0], -1), -1).mean(),
                    "noise": torch.norm(noise_t.reshape(image_dim[0], -1), -1).mean(),
                }
            
            if logger is not None:
                if i % 50 == 0 or i < 50:
                    log_dict["samples"] = [wandb.Image(xt_next[i]) for i in range(10)]
                logger(log_dict)
            elif print_logs:
                print(i, j, log_dict)

            xs.append(xt_next.view(image_dim).to('cpu'))

    return xs, x0_preds

def ddpm_steps(x, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        betas = b
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            beta_t = 1 - at / atm1
            x = xs[-1].to('cuda')

            output = model(x, t.float())
            e = output

            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e.to('cpu'))
            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)

            mean = mean_eps
            noise = torch.randn_like(x)
            mask = 1 - (t == 0).float()
            mask = mask.view(-1, 1, 1, 1)
            logvar = beta_t.log()
            sample = mean + mask * torch.exp(0.5 * logvar) * noise
            xs.append(sample.to('cpu'))
    return xs, x0_preds

def generalized_steps_modified(x, seq, model, b, logger=None, scale_xT=False, print_logs=False, **kwargs):
    with torch.no_grad():
        B = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        xT = x
        image_dim = x.shape
        
        last_T = len(seq)-1
        T = (torch.ones(B)*last_T).to(x.device)
        aT = compute_alpha(b, T.long())

        for i, j in zip(reversed(seq), reversed(seq_next)):
            if i == last_T:
                continue
            # import pdb; pdb.set_trace()
            t = (torch.ones(B) * i).to(x.device)
            next_t = (torch.ones(B) * j).to(x.device)

            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())

            at_aT_ = (1 - at)/(1 - aT)
            ataT = aT / at
            
            xt = xs[-1].to('cuda')
            if scale_xT:
                et = model(xt, (at_aT_) * (ataT).sqrt() * xT, t)
            else:
                et = model(xt, xT, t)

            # predicted x_0t
            x0_t = xt - (at_aT_) * (ataT).sqrt() * xT - et * (((at - aT)/at) * at_aT_).sqrt()
            x0_t = x0_t / ((at - aT)/(at.sqrt()*(1 - aT)))

            x0_preds.append(x0_t.to('cpu'))
            x0_coeff = (at_next - aT) / ((1 - aT) * at_next.sqrt())
            xT_coeff = ((1 - at_next)/(1 - aT)) * (aT/at_next).sqrt()

            sigma_t = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )

            et_coeff = (((1 - at_next)*(at_next - aT))/(at_next * (1 - aT)) - sigma_t ** 2).sqrt()

            noise_t = torch.randn_like(x)
            xt_next = x0_coeff * x0_t + xT_coeff * xT + et_coeff * et + sigma_t * noise_t
            xs.append(xt_next.to('cpu'))

            log_dict = {
                "alpha at": torch.mean(at).item(),
                "alpha at_next": torch.mean(at_next).item(),
                "xt": torch.norm(xt.reshape(image_dim[0], -1), -1).mean(),
                "xt_next": torch.norm(xt_next.reshape(image_dim[0], -1), -1).mean(),
                "coeff x0": x0_coeff.squeeze().mean(),
                "coeff xT": xT_coeff.squeeze().mean(),
                "coeff et": et_coeff.squeeze().mean(),
                "sigma_t": sigma_t.squeeze().mean(),
                "prediction et": torch.norm(et.reshape(image_dim[0], -1), -1).mean(),
                "noise": torch.norm(noise_t.reshape(image_dim[0], -1), -1).mean(),
            }

            if logger is not None:
                if i % 50 == 0 or i < 50:
                    log_dict["samples"] = [wandb.Image(xt_next[i]) for i in range(10)]
                logger(log_dict)
            elif print_logs:
                print(i, j, log_dict)
                                    
    return xs, x0_preds


def generalized_steps_modified_fixed_U(x, seq, model, b, logger=None, scale_xT=False, print_logs=False, **kwargs):
    with torch.no_grad():
        B = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        xT = x
        image_dim = x.shape
        U = get_ortho_mat(x.dtype, x.shape, x.device, method='bases')
        noise_t = torch.randn_like(x.view(image_dim[0], -1))

        last_T = len(seq)-1
        T = (torch.ones(B)*last_T).to(x.device)
        aT = compute_alpha(b, T.long())

        for i, j in zip(reversed(seq), reversed(seq_next)):
            if i == last_T:
                continue

            t = (torch.ones(B) * i).to(x.device)
            next_t = (torch.ones(B) * j).to(x.device)

            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())

            at_aT_ = (1 - at)/(1 - aT)
            ataT = aT / at
            
            xt = xs[-1].to('cuda')
            if scale_xT:
                et = model(xt, (at_aT_) * (ataT).sqrt() * xT, t)
            else:
                et = model(xt, xT, t)

            # predicted x_0t
            x0_t = xt - (at_aT_) * (ataT).sqrt() * xT - et * (((at - aT)/at) * at_aT_).sqrt()
            x0_t = x0_t / ((at - aT)/(at.sqrt()*(1 - aT)))

            x0_preds.append(x0_t.to('cpu'))
            
            x0_coeff = (at_next - aT) / ((1 - aT) * at_next.sqrt())
            xT_coeff = ((1 - at_next)/(1 - aT)) * (aT/at_next).sqrt()

            sigma_t = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )

            et_coeff = (((1 - at_next)*(at_next - aT))/(at_next * (1 - aT)) - sigma_t ** 2).sqrt()

            # xt_next = (x0_coeff * x0_t).view(image_dim[0], -1) + (xT_coeff * xT).view(image_dim[0], -1) + \
            #            (et_coeff * et).view(image_dim[0], -1) + sigma_t.squeeze().unsqueeze(1) * noise_t

            xt_next = x0_coeff * x0_t + xT_coeff * xT + \
                        et_coeff * et + sigma_t * noise_t.view(image_dim)

            noise_t = torch.matmul(noise_t, U)

            xs.append(xt_next.view(image_dim).to('cpu'))

            log_dict = {
                "alpha at": torch.mean(at).item(),
                "alpha at_next": torch.mean(at_next).item(),
                "xt": torch.norm(xt.reshape(image_dim[0], -1), -1).mean(),
                "xt_next": torch.norm(xt_next.reshape(image_dim[0], -1), -1).mean(),
                "coeff x0": x0_coeff.squeeze().mean(),
                "coeff xT": xT_coeff.squeeze().mean(),
                "coeff et": et_coeff.squeeze().mean(),
                "sigma_t": sigma_t.squeeze().mean(),
                "prediction et": torch.norm(et.reshape(image_dim[0], -1), -1).mean(),
                "noise": torch.norm(noise_t.reshape(image_dim[0], -1), -1).mean(),
            }

            if logger is not None:
                if i % 50 == 0 or i < 50:
                    log_dict["samples"] = [wandb.Image(xt_next.view(image_dim)[i]) for i in range(10)]
                logger(log_dict)
            elif print_logs:
                print(i, j, log_dict)
                                    
    return xs, x0_preds