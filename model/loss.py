import torch
import numpy as np
import torch.nn.functional as F

# def nll_loss(output, target):
#     return F.nll_loss(output, target)

def log_normal_pdf(x, mean, logvar):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    val = torch.sum(-.5 * (const + logvar + ((x - mean) ** 2) / torch.exp(logvar)), dim=[1,2])
    assert bool((val<=0.0).all()), "Invalid log-likelihood value " + str(val) 
    return val.mean()


def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.
    kl = torch.sum(lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2) / (2. * v2)) - .5, dim=1)
    assert bool((kl>=0.0).all()), "Invalid KL divergence value " + str(kl)
    return kl.mean()

def latent_ode_loss(output, target):
    # KL divergence between 
    means, logvars = output['z0_means'], output['z0_logvars']
    # kl_loss = 0.0
    # normal_mean = normal_logvar = torch.zeros(means[0].shape)
    # for mean, logvar in zip(means, logvars):
    #     kl_loss += normal_kl(mean, logvar, normal_mean, normal_logvar)
    k = len(means)
    prior_z0_mean = torch.zeros(means[0].size())
    prior_z0_logvar = torch.log(torch.ones(logvars[0].size())/k)
    reciprocal_sum = sum([1./(1e-9 + torch.exp(lv)) for lv in logvars]) + 1e-9 # for stability, add epsilon
    posterior_z0_mean = sum([mu/torch.exp(lv) for mu,lv in zip(means, logvars)])/reciprocal_sum
    posterior_z0_logvar = torch.log(1./reciprocal_sum)
    kl_loss = normal_kl(posterior_z0_mean, posterior_z0_logvar, prior_z0_mean, prior_z0_logvar)

    pred_mean, pred_logvar = output['pred_mean'], output['pred_logvar']
    logpx = log_normal_pdf(target, pred_mean, pred_logvar)
    return kl_loss - logpx
