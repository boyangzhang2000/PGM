"""Linear forward operators A (degradation) and A^T (adjoint) on 256x256 images.

* GaussianDeblur: circular convolution.
* BicubicSR: 4x bicubic downsampling, with the bicubic upsample as the
  pseudo-adjoint.
* RandomInpaint: uniform random mask.
"""
import numpy as np
import torch as th
import torch.nn.functional as F
from scipy import ndimage


def _np2dev(a, dev):
    return th.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(dev)


def gaussian_kernel(ksize=61, sigma=3.0):
    n = np.zeros((ksize, ksize), dtype=np.float64)
    n[ksize // 2, ksize // 2] = 1.0
    k = ndimage.gaussian_filter(n, sigma=sigma)
    return (k / k.sum()).astype(np.float64)


class GaussianDeblur:
    """Circular convolution with the 61x61 Gaussian kernel (sigma=3.0 default)."""
    def __init__(self, ksize=61, sigma=3.0, device="cuda"):
        k = gaussian_kernel(ksize, sigma)
        self.kernel = k
        H = W = 256
        kp = np.zeros((H, W), dtype=np.float64)
        kp[:ksize, :ksize] = k
        kp = np.roll(np.roll(kp, -(ksize // 2), 0), -(ksize // 2), 1)
        self._fk = np.fft.fft2(kp)
        self.ksize = ksize
        self.device = device

    def A(self, x):
        xn = x.detach().float().cpu().numpy()
        yn = np.fft.ifft2(np.fft.fft2(xn, axes=(-2, -1)) * self._fk,
                          axes=(-2, -1)).real
        return _np2dev(yn, x.device)

    def A_T(self, x):
        xn = x.detach().float().cpu().numpy()
        yn = np.fft.ifft2(np.fft.fft2(xn, axes=(-2, -1)) * np.conj(self._fk),
                          axes=(-2, -1)).real
        return _np2dev(yn, x.device)


class BicubicSR:
    """Bicubic downsampling (with bicubic upsample as the pseudo-adjoint)."""
    def __init__(self, sf=4, device="cuda"):
        self.sf = sf
        self.device = device

    def A(self, x):
        xn = x.detach().float().cpu()
        yn = F.interpolate(xn, size=(256 // self.sf, 256 // self.sf),
                           mode="bicubic", align_corners=False)
        return yn.to(x.device)

    def A_T(self, y):
        yn = y.detach().float().cpu()
        xn = F.interpolate(yn, scale_factor=self.sf, mode="bicubic",
                           align_corners=False)
        return xn.to(y.device)


class RandomInpaint:
    """Random uniform mask (keep `keep` fraction of pixels)."""
    def __init__(self, keep=0.3, seed=7, device="cuda"):
        g = th.Generator(device=device).manual_seed(seed)
        m = (th.rand(1, 1, 256, 256, device=device, generator=g) < keep).float()
        self.mask = m
        self.device = device

    def A(self, x):
        return x * self.mask

    def A_T(self, y):
        return y * self.mask


def verify_deblur_op(device="cuda", seeds=(0, 1, 2)):
    """Sanity check: FFT circular convolution matches scipy's mode='wrap'."""
    op = GaussianDeblur(61, 3.0, device)
    k = op.kernel
    for s in seeds:
        xr = np.random.RandomState(s).rand(256, 256).astype(np.float64) * 2 - 1
        ref = ndimage.convolve(xr, k, mode="wrap")
        ours = op.A(_np2dev(xr[None, None], device))[0, 0].cpu().numpy()
        err = float(np.abs(ref - ours).max())
        print(f"[verify] seed={s}: max|diff|={err:.2e}", flush=True)
        if err > 1e-4:
            return err
    return 0.0
