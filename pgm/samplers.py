"""Annealed proximal Langevin sampling with Joint and Partial updates.

Joint evaluates the proximal map after the likelihood step. Partial keeps
the likelihood gradient outside the map and includes its confinement term.
"""
from collections import deque
from dataclasses import dataclass
import math
import torch
from .consistency import adam_fit, refine_prox
from .accounting import CountedOperator
from .directions import LikelihoodDirection, limit_guidance
from .integrators import coefficients, step_size
from .preconditioning import curvature_spectrum, spectral_coefficients, spectral_step
from .initialization import initialization_generator, transform_initial_state


@dataclass
class SampleResult:
    estimate: torch.Tensor
    state: torch.Tensor
    nfe: int
    trace: list[dict]
    evaluations: dict


def _base_initial_state(op, y, cfg, generator):
    s = cfg.sampler
    shape = (y.shape[0], y.shape[1], *op.shape[-2:])
    if s.init == "zero":
        return torch.zeros(shape, device=y.device)
    if s.init == "gauss":
        return s.gauss_scale * torch.randn(shape, device=y.device, generator=generator)
    if s.init == "backproj":
        return op.backproject(y, generator).clone()
    if y.shape != shape:
        raise ValueError("y_plus initialization needs an image-shaped observation")
    return (y + cfg.operator.noise_std * torch.randn(shape, device=y.device, generator=generator)).clamp(-1, 1)


def initial_state(op,y,cfg,generator,initial_lambda=None):
    x = _base_initial_state(op, y, cfg, generator)
    # Initialization noise uses a separate seed domain from the chain noise.
    init_rng = initialization_generator(generator) if cfg.sampler.init_noise else generator
    return transform_initial_state(x, cfg.sampler, init_rng, initial_lambda)


@torch.no_grad()
def sample(model, op, y, cfg, seed=None, progress=None):
    cfg.validate()
    op = CountedOperator(op)
    s, c = cfg.sampler, cfg.consistency
    if max(s.timesteps + ([s.final_t] if s.final_prox and s.readout == 'prox' else [])) >= len(model.lambdas):
        raise ValueError('Sampling times exceed the model diffusion schedule')
    generator = torch.Generator(device=y.device).manual_seed(cfg.seed if seed is None else seed)
    x = initial_state(op, y, cfg, generator, float(model.lambdas[s.timesteps[0]]))
    power = curvature_spectrum(op, x, y, cfg) if s.preconditioner == 'spectral' else None
    before = model.nfe
    buffer = deque(maxlen=s.average_window)
    trace = []
    joint = cfg.method == "joint"
    direction = LikelihoodDirection(s)
    for stage, (t, k, gamma) in enumerate(zip(s.timesteps, s.steps, s.gamma)):
        lam = float(model.lambdas[t])
        gamma = gamma / lam if s.likelihood_mode == "inverse_lambda" else gamma
        d = step_size(lam, s)
        a = 1 / lam + (0 if joint else s.confinement * lam)
        temperature = s.temperature * (a * lam if s.temperature_mode == "matched" else 1)
        if s.temperature_power:
            temperature *= (lam/float(model.lambdas[s.timesteps[0]]))**s.temperature_power
        w, h, std = coefficients(s.scheme, a, d)
        if s.likelihood_mode == "step":
            gamma = gamma * s.grad_scale**2 / (lam if joint else h)
        weights = spectral_coefficients(power, a=a, d=d, gamma=gamma, grad_scale=s.grad_scale,
                                        scheme=s.scheme) if power is not None else None
        for _ in range(k):
            if s.data_location == "denoised":
                p, grad = model.prox_likelihood_gradient(x, t, op, y)
                op.counts['denoised_likelihood_vjps'] = op.counts.get('denoised_likelihood_vjps', 0) + 1
                grad = direction(grad, x) / s.grad_scale**2
                p = refine_prox(op, p, y, cfg, stage)
            elif s.data_location == "prox":
                p = refine_prox(op, model.prox(x, t), y, cfg, stage)
                grad = direction(op.gradient(p, y), p) / s.grad_scale**2
            else:
                grad = direction(op.gradient(x, y), x) / s.grad_scale**2
                if joint:
                    grad = limit_guidance(grad, gamma*lam, s.guidance_rms_cap, s.guidance_pixel_cap)
                arg = x - gamma * lam * grad if joint else x
                p = refine_prox(op, model.prox(arg, t), y, cfg, stage)
            if not joint:
                grad = limit_guidance(grad, gamma*h, s.guidance_rms_cap, s.guidance_pixel_cap)
            forcing = p / lam if joint else p / lam - gamma * grad
            noise = torch.randn(x.shape, device=x.device, generator=generator)
            if weights is None:
                x = w * x + h * forcing + std * math.sqrt(temperature) * noise
            else:
                x = spectral_step(x, forcing, math.sqrt(temperature)*noise, weights)
            # Check BEFORE clipping, so infinities cannot masquerade as valid output.
            if not torch.isfinite(x).all():
                raise FloatingPointError(f"Nonfinite chain at stage={stage}, t={t}")
            if s.clamp is not None:
                x = x.clamp(*s.clamp)
            if s.readout == "average":
                buffer.append(p)
        trace.append({"stage": stage, "t": t, "lambda": lam, "step_size": d,
                      "temperature": temperature,
                      "gamma": gamma, "steps": k, "state_rms": float(x.square().mean().sqrt()),
                      "linear_state_weight": w if weights is None else None,
                      "linear_spectral_radius": abs(w) if weights is None else float(weights[0].abs().max()),
                      "linear_stable": abs(w) < 1 if weights is None else bool((weights[0].abs() < 1).all())})
        if progress:
            progress(trace[-1])
    if s.readout == "average":
        out = torch.stack(list(buffer)).mean(0)
    elif s.readout == "prox" and s.final_prox:
        out = model.prox(x, s.final_t)
    else:
        out = x
    if c.final_amplitude:
        out = op.project_amplitude(out, y, phase_coupling=c.phase_coupling)
    if c.final_adam_steps:
        out = adam_fit(op, out, y, c.final_adam_steps, c.final_adam_lr, cfg.operator.noise_std)
    if not torch.isfinite(out).all():
        raise FloatingPointError("Nonfinite reconstruction")
    nfe = model.nfe - before
    return SampleResult(out.detach(), x.detach(), nfe, trace, dict(prior=nfe, **op.counts))
