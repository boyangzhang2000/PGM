"""Unrounded per-image metrics; Gaussian-window SSIM on the declared range."""
import torch
import torch.nn.functional as F


def psnr(x, gt, data_range=2.0):
    mse = (x - gt).square().flatten(1).mean(1)
    return 10 * torch.log10(data_range**2 / mse.clamp_min(1e-12))


def ssim(x, gt, data_range=2.0, window=11, sigma=1.5):
    if min(x.shape[-2:]) < window:
        raise ValueError("SSIM image is smaller than its window")
    coords = torch.arange(window, device=x.device, dtype=x.dtype) - window // 2
    k = torch.exp(-coords.square() / (2 * sigma**2))
    k = k / k.sum()
    kernel = torch.outer(k, k)[None, None].expand(x.shape[1], 1, -1, -1)
    conv = lambda v: F.conv2d(v, kernel, groups=x.shape[1])
    ux, uy = conv(x), conv(gt)
    vx = conv(x.square()) - ux.square()
    vy = conv(gt.square()) - uy.square()
    cov = conv(x * gt) - ux * uy
    c1, c2 = (0.01 * data_range)**2, (0.03 * data_range)**2
    score = ((2 * ux * uy + c1) * (2 * cov + c2)) / ((ux.square() + uy.square() + c1) * (vx + vy + c2))
    return score.flatten(1).mean(1)


def make_lpips(device, mode="auto"):
    if mode == "off":
        return None, "disabled"
    try:
        import lpips
        from pathlib import Path
        cached = Path(torch.hub.get_dir()) / "checkpoints/vgg16-397923af.pth"
        if mode == "auto" and not cached.is_file():
            return None, "VGG weights absent; use --lpips on to download"
        return lpips.LPIPS(net="vgg", verbose=False).eval().requires_grad_(False).to(device), "available"
    except (ImportError, OSError, RuntimeError) as exc:
        if mode == "on":
            raise
        return None, f"unavailable: {exc}"


@torch.no_grad()
def evaluate(x, gt, lpips_model=None):
    x, gt = x.float(), gt.float()
    x01, gt01 = (x + 1) / 2, (gt + 1) / 2
    weights = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)

    result = {"psnr": float(psnr(x, gt).mean()),
              "psnr_y": float(psnr((x01 * weights).sum(1, keepdim=True),
                                        (gt01 * weights).sum(1, keepdim=True), 1).mean()),
              "ssim": float(ssim(x01, gt01, 1).mean()), "lpips": None}
    if lpips_model is not None:
        result["lpips"] = float(lpips_model(x, gt).mean())
    return result
