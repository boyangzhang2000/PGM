"""Dimension-normalized Gaussian-kernel matching and annealing rules."""
import math
import torch


def bandwidth(config, step, total_steps):
    progress = min(1., step/max(total_steps-1, 1))
    if config.schedule == 'geometric':
        return config.start*(config.end/config.start)**progress
    return config.start + (config.end-config.start)*progress


def learning_rate(config, step, total_steps):
    if step < config.warmup_steps:
        return config.lr*(step+1)/config.warmup_steps
    if config.lr_schedule == 'cosine':
        progress = (step-config.warmup_steps)/max(total_steps-config.warmup_steps, 1)
        return config.lr*(1+math.cos(math.pi*progress))/2
    return config.lr


def gaussian_matching(prediction, target, zeta):
    if not math.isfinite(zeta) or zeta <= 0:
        raise ValueError("Kernel bandwidth must be finite and positive")
    error = (prediction.float()-target.float()).square().flatten(1).mean(1)
    exponent = -error/zeta**2
    # This form preserves the original kernel objective near exact agreement.
    losses = -torch.expm1(exponent)
    return losses, error, torch.exp(exponent)


def perturb(clean, timesteps, noise, mu, sigma):
    shape = (-1,) + (1,)*(clean.ndim-1)
    return mu[timesteps].view(shape)*clean + sigma[timesteps].view(shape)*noise
