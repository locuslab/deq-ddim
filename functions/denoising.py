import torch
from functions.utils import get_ortho_mat
import wandb

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def generalized_steps(x, seq, model, b, logger=None, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        image_dim = x.shape
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
                    "samples": [wandb.Image(xt_next[i]) for i in range(10)]
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

def generalized_steps_fixed_U(x, seq, model, b, **kwargs):
    with torch.no_grad():
        B = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]

        image_dim = x.shape
        U = get_ortho_mat(x.dtype, x.shape, x.device, method='bases')
        noise = torch.randn_like(x.view(image_dim[0], -1))

        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(B) * i).to(x.device)
            next_t = (torch.ones(B) * j).to(x.device)
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

            xt_next = (at_next.sqrt() * x0_t).view(image_dim[0], -1) + c1.squeeze().unsqueeze(1) * noise + c2.squeeze().unsqueeze(1) * et.view(image_dim[0], -1)
            noise = torch.matmul(noise, U)

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

def ddpm_steps_fixed_U(x, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        betas = b

        image_dim = x.shape
        U = get_ortho_mat(x.dtype, x.shape, x.device, method='bases')
        noise = torch.randn_like(x.view(image_dim[0], -1))
        import pdb; pdb.set_trace()
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
            mask = mask.view(-1, 1)
            logvar = beta_t.log()
            sample = mean.view(image_dim[0], -1) + mask * torch.exp(0.5 * logvar) * noise
            noise = torch.matmul(noise, U)
            xs.append(sample.view(image_dim[0], -1).to('cpu'))
    return xs, x0_preds
