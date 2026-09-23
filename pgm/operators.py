"""Differentiable linear measurement operators with exact adjoints."""
import numpy as np
import torch
import torch.nn.functional as F
from .resizer import Resizer


class Operator:
    linear = False

    def __init__(self, device="cpu", size=256):
        self.device = torch.device(device)
        self.shape = (1, 3, size, size)

    def forward(self, x):
        raise NotImplementedError

    def gradient(self, x, y):
        """Differentiate the squared observation error with respect to the image."""
        with torch.enable_grad():
            u = x.detach().requires_grad_(True)
            loss = (self.forward(u) - y.detach()).square().sum() / 2
            return torch.autograd.grad(loss, u)[0].detach()

    def vjp(self, x, residual):
        """Jacobian transpose at an explicit point, for initialization probes."""
        with torch.enable_grad():
            u = x.detach().requires_grad_(True)
            return torch.autograd.grad(self.forward(u), u, residual.detach())[0].detach()

    def backproject(self, y, generator=None):
        if self.linear:
            return self.adjoint(y)
        return y.detach().clamp(-1, 1)

    def adjoint(self, y):
        if not self.linear:
            raise TypeError("Nonlinear operators have a Jacobian at x, not a global adjoint")
        with torch.enable_grad():
            u = torch.zeros((y.shape[0], y.shape[1], *self.shape[-2:]),
                            device=y.device, dtype=y.dtype, requires_grad=True)
            return torch.autograd.grad(self.forward(u), u, y)[0].detach()

    # A/A_T retained only for the unambiguous linear API.
    def A(self, x):
        return self.forward(x)

    def A_T(self, y):
        return self.adjoint(y)


class GaussianDeblur(Operator):
    """Separable Gaussian convolution with reflection boundary conditions."""
    linear = True

    def __init__(self, ksize=61, sigma=3.0, device="cpu", size=256):
        super().__init__(device, size)
        if ksize % 2 != 1 or ksize < 1 or ksize // 2 >= size or sigma <= 0:
            raise ValueError("Invalid Gaussian kernel size or sigma")
        self.pad = ksize // 2
        coords = torch.arange(ksize, dtype=torch.float64) - self.pad
        k = torch.exp(-coords.square() / (2 * sigma**2))
        k[coords.abs() > int(4 * sigma + 0.5)] = 0  # scipy gaussian_filter existing support
        k = (k / k.sum()).float().to(device)
        self.kernel = torch.outer(k, k)
        self.horizontal = k.view(1, 1, 1, -1).repeat(3, 1, 1, 1)
        self.vertical = k.view(1, 1, -1, 1).repeat(3, 1, 1, 1)

    def forward(self, x):
        x = F.conv2d(F.pad(x, (self.pad, self.pad, 0, 0), mode="reflect"),
                     self.horizontal.to(dtype=x.dtype), groups=3)
        return F.conv2d(F.pad(x, (0, 0, self.pad, self.pad), mode="reflect"),
                        self.vertical.to(dtype=x.dtype), groups=3)


class BicubicSR(Operator):
    """Original antialiased cubic forward; exact transpose from autograd."""
    linear = True

    def __init__(self, sf=4, device="cpu", size=256):
        super().__init__(device, size)
        if sf < 1 or size % sf:
            raise ValueError("SR factor must divide image size")
        self.sf = sf
        self.down = Resizer(self.shape, scale_factor=1 / sf,
                            kernel="cubic", antialiasing=True).to(device)

    def forward(self, x):
        return self.down(x)

    def backproject(self, y, generator=None):
        # Normalized transpose for initialization, never for the likelihood.
        return self.adjoint(y) / self.adjoint(torch.ones_like(y)).clamp_min(1e-8)


class RandomInpaint(Operator):
    """Seeded exact-count missing mask; shared between RGB channels."""
    linear = True

    def __init__(self, prob=0.7, seed=7, device="cpu", size=256):
        super().__init__(device, size)
        if not 0 <= prob < 1:
            raise ValueError("Missing fraction must be in [0, 1)")
        rng = np.random.RandomState(seed)
        mask = np.ones(size * size, dtype=np.float32)
        mask[rng.choice(size * size, round(size * size * prob), replace=False)] = 0
        self.mask = torch.from_numpy(mask.reshape(1, 1, size, size)).to(device)

    def forward(self, x):
        return self.mask * x

    def adjoint(self, y):
        return self.mask * y

    def gradient(self, x, y):
        return self.mask * (self.mask * x - y)
