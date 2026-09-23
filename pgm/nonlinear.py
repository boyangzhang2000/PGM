"""Stateless nonlinear measurements; gradients always take an explicit x."""
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from .config import ROOT, ConfigLoader, project_path
from .operators import Operator


def centered_fft(x, pad):
    x = F.pad(x, (pad, pad, pad, pad))
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)),
                                          norm="ortho"), dim=(-2, -1))


def centered_ifft(z, pad):
    x = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(z, dim=(-2, -1)),
                                          norm="ortho"), dim=(-2, -1)).real
    return x[..., pad:-pad, pad:-pad] if pad else x


class PhaseRetrievalOperator(Operator):
    """Padded Fourier-amplitude measurements with configurable FFT placement."""

    def __init__(self, oversample=2.0, device="cpu", size=256, fft_device="auto"):
        super().__init__(device, size)
        self.pad = int(oversample * size / 8)
        self.fft_device = fft_device

    def _fft_input(self, x):
        return x.cpu() if self.fft_device == "cpu" else x

    def forward(self, x):
        return centered_fft(self._fft_input(x) * 0.5 + 0.5, self.pad).abs().to(x.device)

    def backproject(self, y, generator=None):
        phase = 2 * torch.pi * torch.rand(y.shape, generator=generator, device=y.device)
        z = y.clamp_min(0) * torch.exp(1j * phase)
        x = centered_ifft(self._fft_input(z), self.pad)
        return (2 * x - 1).to(y.device)

    def project_amplitude(self, x, y, keep=1.0, phase_coupling=0.0):
        u = self._fft_input(x)
        z = centered_fft(0.5 * u + 0.5, self.pad)
        target = self._fft_input(y).clamp_min(0)

        phase = torch.exp(1j * torch.angle(z))
        if phase_coupling:
            # Optional color prior: shrink channel phases toward the luminance phase.
            # This alters the reconstruction prior, not the observed amplitudes.
            common = torch.exp(1j * torch.angle(z.mean(dim=1, keepdim=True)))
            phase = torch.exp(1j * torch.angle((1-phase_coupling)*phase + phase_coupling*common))
        projected = target * phase
        cy, cx = projected.shape[-2] // 2, projected.shape[-1] // 2
        projected[..., cy, cx] = target[..., cy, cx].to(projected.dtype)
        result = 2 * centered_ifft(projected, self.pad) - 1
        return (keep * result + (1 - keep) * u).to(x.device)


def resolve_engine(explicit=None):
    selected = explicit or os.environ.get("PGM_NDB_ENGINE")
    local = ROOT / "configs" / "local.yaml"
    if selected is None and local.is_file():
        selected = (yaml.safe_load(local.read_text(encoding="utf-8")) or {}).get("ndb_engine")
    root = project_path(selected).resolve() if selected else ROOT
    if not root.is_dir():
        raise FileNotFoundError(f"NDB asset directory not found: {root}")
    return root


class NonlinearDeblurOperator(Operator):
    """KernelWizard blur with a reproducible latent kernel."""

    def __init__(self, options, engine_dir=None, device="cpu", kernel_seed=0,
                 kernel_scale=1.2, size=256):
        super().__init__(device, size)
        root = resolve_engine(engine_dir)
        option_path = Path(options)
        option_path = option_path if option_path.is_absolute() else root / option_path
        if not option_path.is_file() and not Path(options).is_absolute():
            option_path = project_path(options)
        opt = yaml.load(option_path.read_text(encoding="utf-8"), Loader=ConfigLoader)["KernelWizard"]
        checkpoint = Path(opt["pretrained"])
        checkpoint = checkpoint if checkpoint.is_absolute() else root / checkpoint
        if not checkpoint.is_file():
            raise FileNotFoundError(f"KernelWizard checkpoint missing: {checkpoint}")
        from .backends.kernel_wizard import KernelWizard
        engine = KernelWizard(opt)
        engine.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        # adaptKernel uses only these three modules. Drop unused kernel extractor
        # after strict loading to save memory on both CPU and small GPUs.
        del engine.kernel_extractor
        self.engine = engine.eval().requires_grad_(False).to(device)
        self.engine_root, self.checkpoint = str(root), str(checkpoint.resolve())
        rng = np.random.RandomState(kernel_seed)
        kernel = rng.randn(1, 512, 2, 2).astype(np.float32) * kernel_scale
        self.kernel = torch.from_numpy(kernel).to(device)

    def forward(self, x):
        y = self.engine.adaptKernel((x + 1) / 2, kernel=self.kernel.expand(x.shape[0], -1, -1, -1))
        return (2 * y - 1).clamp(-1, 1)
