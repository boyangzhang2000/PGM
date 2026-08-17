"""Image-quality metrics: PSNR (RGB / Y), SSIM, LPIPS (VGG)."""
import torch as th


def rgb2y(x):
    """RGB [-1,1] -> Y channel (BT.601), shape (N,1,H,W) in [0,1]."""
    x = (x + 1) / 2
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def psnr(a, b, max_val=2.0):
    """PSNR with given dynamic range (default 2.0 for [-1,1] images)."""
    mse = ((a - b) ** 2).flatten(1).mean(1)
    return 10 * th.log10(max_val ** 2 / mse.clamp(min=1e-12))


def ssim_torch(a, b, data_range=2.0, win=11):
    """Single-window SSIM (no image downsampling)."""
    from torch.nn.functional import conv2d
    a = (a + 1) / 2
    b = (b + 1) / 2
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    k = th.ones(1, 1, win, win, device=a.device) / (win * win)
    k = k.expand(a.shape[1], 1, win, win)
    pad = win // 2
    mu_a = conv2d(th.nn.functional.pad(a, (pad, pad, pad, pad), mode="replicate"),
                  k, groups=a.shape[1])
    mu_b = conv2d(th.nn.functional.pad(b, (pad, pad, pad, pad), mode="replicate"),
                  k, groups=b.shape[1])
    mu_ab = mu_a * mu_b
    mu_aa, mu_bb = mu_a * mu_a, mu_b * mu_b
    sig_aa = conv2d(th.nn.functional.pad(a * a, (pad, pad, pad, pad), mode="replicate"),
                    k, groups=a.shape[1]) - mu_aa
    sig_bb = conv2d(th.nn.functional.pad(b * b, (pad, pad, pad, pad), mode="replicate"),
                    k, groups=b.shape[1]) - mu_bb
    sig_ab = conv2d(th.nn.functional.pad(a * b, (pad, pad, pad, pad), mode="replicate"),
                    k, groups=a.shape[1]) - mu_ab
    return (((2 * mu_ab + c1) * (2 * sig_ab + c2)) /
            ((mu_aa + mu_bb + c1) * (sig_aa + sig_bb + c2))).flatten(1).mean(1)


def make_lpips(device):
    """LPIPS-VGG network (downloads weights on first use)."""
    import lpips
    return lpips.LPIPS(net="vgg").to(device)


def eval_metrics(x_est, x_gt, lpips_fn=None):
    """Compute per-image PSNR-RGB, PSNR-Y, SSIM, LPIPS (if available).
    x_*: (N, 3, 256, 256) in [-1, 1]. Returns list-of-dicts."""
    ps = psnr(x_est, x_gt, max_val=2.0).tolist()
    ps_y = psnr(rgb2y(x_est), rgb2y(x_gt), max_val=1.0).tolist()
    ss = ssim_torch(x_est, x_gt).tolist()
    lp = []
    if lpips_fn is not None:
        with th.no_grad():
            lp = lpips_fn(x_est, x_gt).flatten().tolist()
    return [
        dict(psnr=round(v_ps, 2), psnr_y=round(v_py, 2),
             ssim=round(v_s, 4), lpips=round(v_l, 4) if lpips_fn else None)
        for v_ps, v_py, v_s, v_l in zip(ps, ps_y, ss, lp or [0.0] * len(ps))
    ]
