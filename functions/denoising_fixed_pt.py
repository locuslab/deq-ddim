import torch
from functions.utils import get_ortho_mat
import wandb
import numpy

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = beta.index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def generalized_steps(x, seq, model, b, logger=None, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        image_dim = x.shape
        if isinstance(b, numpy.ndarray):
            b = torch.from_numpy(b).float().cuda()
        
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            et = model(xt, t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            noise_t = torch.randn_like(x)
            xt_next = at_next.sqrt() * x0_t + c1 * noise_t + c2 * et
            xs.append(xt_next.to('cpu'))

            if logger is not None:
                logger({
                    "alpha at": torch.mean(at),
                    "alpha at_next": torch.mean(at_next),
                    "xt": torch.norm(xt.reshape(image_dim[0], -1), -1).mean(),
                    "xt_next": torch.norm(xt_next.reshape(image_dim[0], -1), -1).mean(),
                    "c1": c1,
                    "c2": c2,
                    "prediction et": torch.norm(et.reshape(image_dim[0], -1), -1).mean(),
                    "noise": torch.norm(noise_t.reshape(image_dim[0], -1), -1).mean(),
                    #"samples": [wandb.Image(xt_next[i]) for i in range(10)]
                })
            else:
                print({
                    "alpha at": torch.mean(at),
                    "alpha at_next": torch.mean(at_next),
                    "xt": torch.norm(xt.reshape(image_dim[0], -1), -1).mean(),
                    "xt_next": torch.norm(xt_next.reshape(image_dim[0], -1), -1).mean(),
                    "c1": torch.mean(c1),
                    "c2": torch.mean(c2),
                    "prediction et": torch.norm(et.reshape(image_dim[0], -1), -1).mean(),
                    "noise": torch.norm(noise_t.reshape(image_dim[0], -1), -1).mean(),
                    #"samples": [wandb.Image(xt_next[i]) for i in range(10)]
                })

    return xs, x0_preds

def generalized_steps_modified(x, seq, model, b, logger=None, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        xT = x
        image_dim = x.shape
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            et = model(xt, t)
            x0_t = (xt - et * (1 - at).sqrt() - (1 - at).sqrt() * xT) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            noise_t = torch.randn_like(x)
            xt_next = at_next.sqrt() * x0_t + c1 * noise_t + c2 * et
            xs.append(xt_next.to('cpu'))

            if logger is not None:
                logger({
                    "alpha at": torch.mean(at),
                    "alpha at_next": torch.mean(at_next),
                    "xt": torch.norm(xt.reshape(image_dim[0], -1), -1).mean(),
                    "xt_next": torch.norm(xt_next.reshape(image_dim[0], -1), -1).mean(),
                    "c1": c1,
                    "c2": c2,
                    "prediction et": torch.norm(et.reshape(image_dim[0], -1), -1).mean(),
                    "noise": torch.norm(noise_t.reshape(image_dim[0], -1), -1).mean(),
                    "samples": [wandb.Image(xt_next[i]) for i in range(10)]
                })
    return xs, x0_preds
