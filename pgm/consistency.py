"""Optional measurement refinements, separate from the Langevin integrator."""
import torch


@torch.no_grad()
def polish(op, x, y, steps, lr, grad_scale):
    for _ in range(steps):
        x = (x - lr * op.gradient(x, y) / grad_scale**2).clamp(-1, 1)
    return x.detach()


def adam_fit(op, x, y, steps, lr, noise_std):
    with torch.enable_grad():
        u = x.detach().clamp(-1, 1).requires_grad_(True)
        optimizer = torch.optim.Adam([u], lr=lr)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = (op.forward(u) - y).square().sum()
            if float(loss.detach()) <= (noise_std + 0.001)**2 * y.numel():
                break
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                u.clamp_(-1, 1)
    return u.detach()


@torch.no_grad()
def finalize_reconstruction(op, estimate, y, cfg):
    """Apply the same output constraints during calibration and normal runs."""
    if cfg.task == 'inpaint' and cfg.consistency.paste_observed:
        return estimate * (1-op.mask) + y * op.mask
    return estimate


def refine_prox(op, p, y, cfg, stage):
    c = cfg.consistency
    if c.amplitude and stage >= c.amplitude_from:
        p = op.project_amplitude(p, y, c.amplitude_keep, phase_coupling=c.phase_coupling)
    if c.adam_steps and stage >= c.adam_from:
        p = adam_fit(op, p, y, c.adam_steps, c.adam_lr, cfg.operator.noise_std)
    elif c.polish_steps:
        p = polish(op, p, y, c.polish_steps, c.polish_lr, cfg.sampler.grad_scale)
    return p
