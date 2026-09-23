"""EM, frozen-forcing exponential interpolation and linearly implicit Euler.

IMEX treats the linear drift implicitly while the nonlinear forcing remains explicit.
"""
import math


def step_size(lam, cfg):
    values = {"lambda": cfg.delta_scale * lam,
              "quad": cfg.delta_scale * cfg.delta_c * lam * lam,
              "abs": cfg.delta_scale,
              "ad": cfg.delta_scale / (lam + 1 / lam)}
    return min(values[cfg.delta_mode], cfg.delta_cap)


def coefficients(scheme, a, d):
    """Return state multiplier, forcing multiplier, noise standard deviation."""
    if a <= 0 or d < 0 or not math.isfinite(a + d):
        raise ValueError("a must be positive and d nonnegative, both finite")
    if scheme == "em":
        return 1 - a * d, d, math.sqrt(2 * d)
    if scheme == "exp":
        return math.exp(-a * d), -math.expm1(-a * d) / a, math.sqrt(-math.expm1(-2 * a * d) / a)
    if scheme == "imp":
        den = 1 + a * d
        return 1 / den, d / den, math.sqrt(2 * d) / den
    raise ValueError(f"Unknown scheme: {scheme}")


def step(x, forcing, noise, *, a, d, scheme):
    w, h, std = coefficients(scheme, a, d)
    return w * x + h * forcing + std * noise
