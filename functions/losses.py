import torch
import numpy as np
def noise_estimation_loss(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, keepdim=False):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, t.float())
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)

def modified_noise_estimation_loss(model,
                          x0: torch.Tensor,
                          xT: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, keepdim=False):

    a_t = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    T = b.shape[0] - 1
    B = t.shape[0]
    T = torch.tensor([T]*B, dtype=t.dtype, device=t.device)
    a_T = (1-b).cumprod(dim=0).index_select(0, T).view(-1, 1, 1, 1)

    x0_coeff = (1.0/a_t.sqrt()) * (a_t - a_T)/(1 - a_T)
    xT_coeff = ((1 - a_t) / (1 - a_T)) * (a_T / a_t).sqrt()
    e_coeff = (((1 - a_t)*(a_t - a_T))/(a_t * (1 - a_T))).sqrt()

    x = x0_coeff * x0 + xT_coeff * xT + e_coeff * e

    output = model(x, xT, t.float())
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)

def modified_noise_estimation_loss_v2(model,
                          x0: torch.Tensor,
                          xT: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, keepdim=False):

    a_t = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    T = b.shape[0] - 1
    B = t.shape[0]
    T = torch.tensor([T]*B, dtype=t.dtype, device=t.device)
    a_T = (1-b).cumprod(dim=0).index_select(0, T).view(-1, 1, 1, 1)

    x0_coeff = (1.0/a_t.sqrt()) * (a_t - a_T)/(1 - a_T)
    xT_coeff = ((1 - a_t) / (1 - a_T)) * (a_T / a_t).sqrt()
    e_coeff = (((1 - a_t)*(a_t - a_T))/(a_t * (1 - a_T))).sqrt()

    x = x0_coeff * x0 + xT_coeff * xT + e_coeff * e

    output = model(x, xT_coeff * xT, t.float())
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)

def geometric_noise_estimation_loss(model,
                          x0: torch.Tensor,
                          xT: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, keepdim=False):

    a_t = b.index_select(0, t).view(-1, 1, 1, 1)
    T = b.shape[0] - 1
    B = t.shape[0]
    T = torch.tensor([T]*B, dtype=t.dtype, device=t.device)
    a_T = b.index_select(0, T).view(-1, 1, 1, 1)

    x0_coeff = (1.0/a_t.sqrt()) * (a_t - a_T)/(1 - a_T)
    xT_coeff = ((1 - a_t) / (1 - a_T)) * (a_T / a_t).sqrt()
    e_coeff = (((1 - a_t)*(a_t - a_T))/(a_t * (1 - a_T))).sqrt()

    x = x0_coeff * x0 + xT_coeff * xT + e_coeff * e

    output = model(x.float(), xT.float(), t.float())
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)

# def t_est_noise_estimation_loss(model,
#                           x0: torch.Tensor,
#                           xT: torch.Tensor,
#                           t: torch.LongTensor,
#                           e: torch.Tensor,
#                           b: torch.Tensor, 
#                           r: torch.Tensor, 
#                           keepdim=False):
#     a = b.index_select(0, t).view(-1, 1, 1, 1)
#     x = x0 * a.sqrt()  + xT * (1.0 - a).sqrt() + e * (1.0 - a).sqrt()
#     t_est = torch.ceil(torch.log(b.index_select(0, t))/torch.log(r))
#     output = model(x, t_est.float())
#     if keepdim:
#         return (e - output).square().sum(dim=(1, 2, 3))
#     else:
#         return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)

loss_registry = {
    'simple': noise_estimation_loss,
    'modified': modified_noise_estimation_loss,
    'modifiedv2': modified_noise_estimation_loss_v2,
    'geometric': geometric_noise_estimation_loss,
#    't_est': t_est_noise_estimation_loss
}
