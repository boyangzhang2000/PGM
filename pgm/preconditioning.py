"""Fixed Fourier metric estimated from observation-only Jacobian probes.

The metric is independent of the chain state. Drift and noise both use it;
for a gradient drift this preserves the stage's invariant density before any
image clipping or approximate learned-prox effects. It is not an inverse solver.
"""
import hashlib

import torch
import torch.nn.functional as F


@torch.no_grad()
def curvature_spectrum(op, initial, y, cfg):
    seed = int.from_bytes(hashlib.sha256(f'{cfg.seed}:curvature'.encode()).digest()[:8], 'little')
    generator = torch.Generator(device=initial.device).manual_seed(seed)
    power = torch.zeros_like(initial)
    for _ in range(cfg.sampler.curvature_probes):
        probe = torch.randn(y.shape, device=y.device, generator=generator)
        jt_probe = op.vjp(initial, probe)
        power += torch.fft.fft2(jt_probe, norm='ortho').abs().square()
    power /= cfg.sampler.curvature_probes
    width = cfg.sampler.curvature_smoothing
    if width > 1:
        radius = width // 2
        power = F.avg_pool2d(F.pad(power, (radius, radius, radius, radius), mode='circular'), width, stride=1)
    # Exact conjugate symmetry gives a real, self-adjoint positive metric.
    return (power + power.flip((-2,-1)).roll((1,1),(-2,-1))) / 2


def spectral_coefficients(power, *, a, d, gamma, grad_scale, scheme):
    metric = a / (a + gamma * power / grad_scale**2)
    ad = a * metric * d
    if scheme == 'em':
        w, h, std = 1-ad, metric*d, (2*metric*d).sqrt()
    elif scheme == 'exp':
        w, h, std = (-ad).exp(), -torch.expm1(-ad)/a, (-torch.expm1(-2*ad)/a).sqrt()
    elif scheme == 'imp':
        w = (1+ad).reciprocal()
        h, std = metric*d*w, (2*metric*d).sqrt()*w
    else:
        raise ValueError(f'Unknown spectral integrator: {scheme}')
    return w, h, std


def spectral_step(x, forcing, noise, weights):
    w, h, std = weights
    transform = lambda u: torch.fft.fft2(u, norm='ortho')
    return torch.fft.ifft2(w*transform(x)+h*transform(forcing)+std*transform(noise), norm='ortho').real
