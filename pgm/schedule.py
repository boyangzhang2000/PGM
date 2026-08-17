"""Annealing schedule helpers (Numpy/Th tensors on the chosen device)."""
import numpy as np
import torch as th


def build_schedule(diffusion, device):
    """Read the diffusion's alpha_cumprod and build:
    alphas_cumprod, mu_t = sqrt(ac), sigma2_t = 1-ac, sigma_t, lam_t = sigma2/mu^2
    (all tensors on `device`)."""
    ac = diffusion.alphas_cumprod.astype(np.float64)
    alphas_cumprod = th.from_numpy(ac).to(device)
    mu_t = alphas_cumprod.sqrt()
    sigma2_t = 1.0 - alphas_cumprod
    sigma_t = sigma2_t.sqrt()
    lam_t = sigma2_t / alphas_cumprod
    return alphas_cumprod, mu_t, sigma2_t, sigma_t, lam_t


def lambda_to_t(lam, lam_t, t_max=999):
    """Find the diffusion timestep t whose lambda(t) is closest to `lam`."""
    lam = float(lam)
    idx = int((lam_t[:t_max + 1] - lam).abs().argmin())
    return max(1, idx)


def lam_seq_of(lam_max, lam_min, T):
    """Geometric lambda schedule: lam[0]=lam_max, lam[T-1]=lam_min."""
    if T <= 1:
        return [lam_min]
    rho = (lam_min / lam_max) ** (1.0 / (T - 1))
    return [lam_max * rho ** i for i in range(T)]


def t_of_lams(lam_seq, lam_t):
    """Convert a list of lambdas to a list of diffusion timesteps."""
    return [lambda_to_t(l, lam_t) for l in lam_seq]


def k_us_seq(t_seq, K_peak, K_mid):
    """U-shaped per-stage K_t (paper APL update):
        u=(t-t_min)/(t_max-t_min)
        K(u) = K_mid + (K_peak - K_mid) * (2u - 1)^2
    Endpoints (u=0, u=1) get K_peak, mid-stage (u=0.5) gets K_mid."""
    t_min, t_max = float(min(t_seq)), float(max(t_seq))
    K = []
    for t in t_seq:
        u = (t - t_min) / max(t_max - t_min, 1e-9)
        k = K_mid + (K_peak - K_mid) * (2.0 * u - 1.0) ** 2
        K.append(max(1, int(round(k))))
    return K


def k_const_seq(t_seq, K):
    """Constant per-stage K_t."""
    return [max(1, int(K))] * len(t_seq)


def k_invt_seq(t_seq, K_ref, t_ref=None):
    """K_t proportional to 1/t (capped): K(t) = K_ref * t_ref / max(t, 1).
    More inner iterations at small t (fine detail), fewer at large t."""
    if t_ref is None:
        t_ref = max(t_seq)
    return [max(1, int(round(K_ref * t_ref / max(t, 1)))) for t in t_seq]


def make_k_seq(t_seq, mode, K_peak, K_mid=None):
    """Dispatch: mode in {"us", "const", "invt"}."""
    if mode == "const":
        return k_const_seq(t_seq, K_peak)
    if mode == "invt":
        return k_invt_seq(t_seq, K_peak)
    return k_us_seq(t_seq, K_peak, K_mid if K_mid is not None else max(1, K_peak // 5))
