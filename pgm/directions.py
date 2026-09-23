"""Optional adaptive likelihood directions; these alter the standard APL drift.

Adam here is one persistent update per outer iteration, never an inner solver.
Its uncorrected, adaptive drift is a reconstruction heuristic, not exact posterior
Langevin sampling. The default identity direction preserves the original SDE.
"""
import torch


def limit_guidance(gradient, coefficient, rms_cap, pixel_cap=None):
    """Bound each image's data displacement without rotating the direction.
    
    Joint bounds the input shift to prox; Partial bounds its explicit data
    displacement. Optional pixel caps preserve each RGB vector's direction,
    but alter the spatial gradient direction and the standard drift.
    """
    if pixel_cap is not None:
        rms = (coefficient*gradient).square().mean(dim=1, keepdim=True).sqrt()
        factor = (pixel_cap/rms.clamp_min(torch.finfo(gradient.dtype).tiny)).clamp_max(1)
        gradient = gradient*factor
    if rms_cap is None:
        return gradient
    rms = (coefficient*gradient).square().flatten(1).mean(1).sqrt()
    factor = (rms_cap/rms.clamp_min(torch.finfo(gradient.dtype).tiny)).clamp_max(1)
    return gradient*factor.view(-1, *([1]*(gradient.ndim-1)))


class LikelihoodDirection:
    def __init__(self, cfg):
        self.cfg = cfg
        self.m = self.v = None
        self.count = 0
        self.previous = None
        self.history = []

    def __call__(self, gradient, point=None):
        if self.cfg.data_direction == 'gradient':
            return gradient
        if self.cfg.data_direction == 'lbfgs':
            return self._lbfgs(gradient, point)
        beta1, beta2 = self.cfg.adam_betas
        if self.m is None:
            self.m, self.v = torch.zeros_like(gradient), torch.zeros_like(gradient)
        self.count += 1
        self.m.mul_(beta1).add_(gradient, alpha=1 - beta1)
        self.v.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
        corrected_m = self.m / (1 - beta1 ** self.count)
        corrected_v = self.v / (1 - beta2 ** self.count)
        return corrected_m / (corrected_v.sqrt() + self.cfg.adam_epsilon)

    def _lbfgs(self, gradient, point):
        """Limited-memory inverse-Hessian action without an inner line search."""
        if point is None:
            raise ValueError('L-BFGS requires the gradient evaluation point')
        dot = lambda a, b: torch.dot(a.flatten(), b.flatten())
        if self.previous is not None:
            previous_point, previous_gradient = self.previous
            displacement, difference = point-previous_point, gradient-previous_gradient
            curvature = dot(displacement, difference)
            if float(curvature) > torch.finfo(gradient.dtype).eps * float(displacement.norm()*difference.norm()):
                self.history.append((displacement, difference, curvature.reciprocal()))
                self.history = self.history[-self.cfg.history_size:]
        self.previous = (point.clone(), gradient.clone())
        q, alphas = gradient.clone(), []
        for displacement, difference, rho in reversed(self.history):
            alpha = rho * dot(displacement, q)
            alphas.append(alpha)
            q = q - alpha * difference
        if self.history:
            displacement, difference, rho = self.history[-1]
            q = q * dot(displacement, difference) / dot(difference, difference)
        else:
            q = q * self.cfg.initial_inverse_curvature
        for (displacement, difference, rho), alpha in zip(self.history, reversed(alphas)):
            q = q + displacement * (alpha - rho * dot(difference, q))
        return q
