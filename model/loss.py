import torch
import numpy as np
import torch.nn.functional as F

# def nll_loss(output, target):
#     return F.nll_loss(output, target)

def log_normal_pdf(x, mean, logvar):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar))


def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.
    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl

def latent_ode_loss(output, target, std=0.01):
    means, logvars, pred = output['mean'], output['logvar'], output['pred_x']
    logpx = torch.sum(log_normal_pdf(target, pred, torch.Tensor([std])))
    kl_loss = 0.0
    normal_mean = torch.zeros(means[0].shape)
    normal_logvar = torch.zeros(logvars[0].shape)
    for mean, logvar in zip(means, logvars):
        kl_loss += torch.sum(
            normal_kl(mean, logvar, normal_mean, normal_logvar))
    return kl_loss - logpx
