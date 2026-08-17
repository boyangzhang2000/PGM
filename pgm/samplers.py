"""PGM samplers -- Annealed Proximal Langevin (APL) as defined in the paper.

partial (Algorithm 3 "guided sampling", the paper's main method)
----------------------------------------------------------------
    pi_t^lambda(x) = exp(-[f(x) + lambda/2 ||x||^2 + M_g^lambda(x)])     (with the
                                                          explicit quadratic term)
    x <- (d/lambda) Prox_g^lambda(x) + (1 - d/lambda - d*lambda) x
         - d * grad f(x) + sqrt(2 d) z
with d = delta_t (per-stage step size) and lambda = lam_t[t].  This is a proximal
Langevin step (Langevin noise sqrt(2 d) z).  The -d*lambda*x coefficient comes from
the explicit quadratic term, which enforces the universal Dirac initialization.

Notation
--------
- x        : chain state in IMAGE space (clean image, values in [-1, 1])
- mu(t), sigma(t): diffusion schedule (used only to evaluate Prox_g^lambda via Tweedie)
- lam_t[t] : lambda(t) = sigma(t)^2 / mu(t)^2
- K_t      : per-stage inner iterations (constant or U-shaped via k_us_seq)
- gamma    : likelihood weight in the explicit-gradient update (paper: gamma=1;
             in practice rescaled per task)
- grad_ref : reference scale for the explicit gradient.  The paper uses
             grad_f = A^T(A x - y) / sigma_n^2; we expose grad_ref so that one can
             set grad_ref = sigma_n (paper) or a fixed 0.1 (engineering default,
             which avoids the 1/sigma_n^2 = 400 blow-up at sigma_n = 0.05).
- delta_mode / delta_c : step-size schedule.  "lambda": d = min(lambda, dmax)
             (P-ULA, delta=lambda); "quad": d = min(delta_c*lambda^2, dmax)
             (the paper's decoupled schedule delta = Theta(lambda^2)).
- final_prox : if True, output Prox_g^{lam_min}(x) (Tweedie of the last state)
             instead of the raw last Langevin state.  The last state carries an
             O(sqrt(lam_min)) Langevin noise floor; projecting it out gives the
             clean point estimate, which is the standard output for restoration.
- clamp_x   : hard clamp on the chain state each inner step (safety; prevents
             NaN blow-ups when the explicit gradient is large).
"""
import torch as th
from tqdm import tqdm
from .model import model_x0


def _prox(model, x, t, mu_t, sigma_t):
    """Prox_g^{lambda(t)}(x) evaluated with the epsilon-prediction network:
    feed mu(t)*x, read the Tweedie x0 = (mu x - sigma eps)/mu, clipped to [-1, 1]."""
    xin = mu_t[t] * x
    return model_x0(model, xin, t, mu_t, sigma_t)


def _delta(lam, mode, cap=0.3, delta_c=0.5, scale=1.0):
    """Per-stage Langevin step size d."""
    if mode == "quad":
        return min(delta_c * lam * lam, cap)
    return min(scale * lam, cap)         # "lambda" (P-ULA, delta = scale*lambda)

def sample_partial(model, op, y, mu_t, sigma_t, lam_t, t_seq, K_seq, sig_n,
                   gamma=1.0, gamma_seq=None, grad_ref=0.1, delta_mode="lambda",
                   delta_c=0.5, delta_cap=0.3, delta_scale=1.0, seed=0, device="cuda",
                   init="backproj", final_prox=False, final_t=None, clamp_x=None):
    """Partial (paper Algorithm 3): annealed proximal Langevin in image space.

    For each stage t (lambda = lam_t[t], step d = delta_t):
        K_seq[i] iterations of
            x <- (d/lambda) Prox_g^lambda(x) + (1 - d/lambda - d*lambda) x
                 - d * gamma_i * grad_f(x) + sqrt(2 d) z
        with grad_f(x) = A^T(A x - y) / grad_ref^2.

    gamma_seq (optional): per-stage likelihood weights; if None, gamma is constant.
    final_t (optional): projection timestep of the final Prox; default t_seq[-1].
    init: "backproj" (single-point Dirac at A^T y), "zero" (Dirac at 0), "gauss".
    """
    g = th.Generator(device=device).manual_seed(seed)
    if init == "gauss":
        x = th.randn(op.A_T(y).shape, generator=g)
    else:
        x = op.A_T(y).clone() if init == "backproj" else th.zeros_like(op.A_T(y))
    total_iters = sum(K_seq)
    with th.no_grad():
        pbar = tqdm(total=total_iters, desc="Sampling progress")
        for i, t in enumerate(t_seq):
            lam = float(lam_t[t])
            d = _delta(lam, delta_mode, cap=delta_cap, delta_c=delta_c, scale=delta_scale)
            gi = gamma_seq[i] if gamma_seq is not None else gamma
            for k in range(K_seq[i]):
                gf = op.A_T(op.A(x) - y) / (grad_ref ** 2)
                p = _prox(model, x, t, mu_t, sigma_t)
                z = th.randn(x.shape, generator=g)
                x = ((d / lam) * p
                     + (1.0 - d / lam - d * lam) * x
                     - d * gi * gf
                     + (2.0 * d) ** 0.5 * z)
                if clamp_x is not None:
                    x = x.clamp(clamp_x[0], clamp_x[1])
                pbar.update(1)
                pbar.set_postfix({"t": t, "k_t": k})
        pbar.close()

    if final_prox:
        ft = final_t if final_t is not None else t_seq[-1]
        x = _prox(model, x, ft, mu_t, sigma_t)
    return x


def sample_joint(model, op, y, mu_t, sigma_t, lam_t, t_seq, K_seq, sig_n,
                 gamma=1.0, gamma_seq=None, grad_ref=0.1, delta_mode="lambda",
                 delta_c=0.5, delta_cap=0.3, delta_scale=1.0, seed=0, device="cuda",
                 init="gauss", final_prox=False, final_t=None, clamp_x=None):
    """Joint (paper Appendix, sec:proxdiff): annealed proximal Langevin with the
    first-order Prox_{f+g} expansion; NO explicit quadratic regularization.

    For each stage t (lambda = lam_t[t], step d = delta_t):
        K_seq[i] iterations of
            x_in = x - gamma_i * lambda * grad_f(x)
            x    <- (d/lambda) Prox_g^lambda(x_in) + (1 - d/lambda) x + sqrt(2 d) z
        with grad_f(x) = A^T(A x - y) / grad_ref^2.

    gamma_seq (optional): per-stage likelihood weights; if None, gamma is constant.
    final_t (optional): projection timestep of the final Prox; default t_seq[-1].
    init: "gauss" (standard Gaussian, the paper's joint initialization),
          "backproj", "zero".
    The coefficient (1 - d/lambda) has NO -d*lambda term: the joint marginal
    e^{-M_{f+g}^lambda} carries no explicit quadratic (that term is partial-only).
    """
    g = th.Generator(device=device).manual_seed(seed)
    if init == "gauss":
        x = th.randn(op.A_T(y).shape, generator=g)
    else:
        x = op.A_T(y).clone() if init == "backproj" else th.zeros_like(op.A_T(y))
    with th.no_grad():
        for i, t in enumerate(t_seq):
            lam = float(lam_t[t])
            d = _delta(lam, delta_mode, cap=delta_cap, delta_c=delta_c, scale=delta_scale)
            gi = gamma_seq[i] if gamma_seq is not None else gamma
            for _ in range(K_seq[i]):
                gf = op.A_T(op.A(x) - y) / (grad_ref ** 2)
                x_in = x - gi * lam * gf
                p = _prox(model, x_in, t, mu_t, sigma_t)
                z = th.randn(x.shape, generator=g)
                x = (d / lam) * p + (1.0 - d / lam) * x + (2.0 * d) ** 0.5 * z
                if clamp_x is not None:
                    x = x.clamp(clamp_x[0], clamp_x[1])
    if final_prox:
        ft = final_t if final_t is not None else t_seq[-1]
        x = _prox(model, x, ft, mu_t, sigma_t)
    return x
