"""Shared diffusion parameterization for proximal training and sampling."""
import hashlib
import math
from pathlib import Path
import torch
import yaml
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from .config import ROOT, ConfigLoader
from .runtime import autocast_context
from .schedule import diffusion_schedule


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_architecture(source=None):
    cfg = model_and_diffusion_defaults()
    if isinstance(source, dict):
        settings = source
    else:
        path = Path(source) if source else ROOT / "configs/models/prox_ffhq.yaml"
        path = path if path.is_absolute() else ROOT / path
        settings = yaml.load(path.read_text(encoding="utf-8"), Loader=ConfigLoader)
    if not isinstance(settings, dict) or set(settings) - set(cfg):
        raise ValueError("Invalid proximal model architecture configuration")
    for name, value in settings.items():
        expected = cfg[name]
        valid = (value is None or type(value) is int) if expected is None else (
            type(value) in (int, float) and math.isfinite(value) if type(expected) is float
            else type(value) is type(expected))
        if not valid:
            raise ValueError(f"Invalid architecture field: {name}")
    cfg.update(settings)
    if cfg['class_cond'] or cfg['predict_xstart'] or cfg['rescale_timesteps']:
        raise ValueError("Proximal models require unconditional noise prediction with direct timesteps")
    if cfg['timestep_respacing'] or cfg['use_fp16']:
        raise ValueError("Use the full diffusion schedule and runtime mixed precision")
    if cfg['in_channels'] != 3 or cfg['out_channels'] not in (None, 3, 6):
        raise ValueError("Proximal image models require RGB prediction")
    if not 0 <= cfg['dropout'] < 1 or any(cfg[k] <= 0 for k in ('image_size', 'num_channels', 'num_res_blocks', 'diffusion_steps')):
        raise ValueError("Invalid network dimensions, diffusion length or dropout")
    return cfg


def checkpoint_weights(payload):
    state = payload.get('state_dict', payload) if isinstance(payload, dict) else None
    if not isinstance(state, dict) or not state or any(not torch.is_tensor(v) for v in state.values()):
        raise ValueError("Checkpoint must contain a network state dictionary")
    return state


def predict_clean(network, noisy, timesteps, mu, sigma, precision="fp32"):
    """Predict the clean image with explicit time conditions and intact gradients."""
    if timesteps.dtype != torch.long or timesteps.shape != (noisy.shape[0],):
        raise ValueError("Timesteps must be a long tensor matching the image batch")
    if bool(((timesteps < 0) | (timesteps >= len(mu))).any()):
        raise ValueError("Time condition is outside the diffusion schedule")
    shape = (-1,) + (1,) * (noisy.ndim - 1)
    with autocast_context(noisy.device, precision):
        epsilon = network(noisy, timesteps)[:, :noisy.shape[1]]
    return ((noisy - sigma[timesteps].view(shape) * epsilon.float()) /
            mu[timesteps].view(shape)).clamp(-1, 1)


class ProximalModel:
    def __init__(self, checkpoint, device, precision="fp32", architecture=None):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        embedded = payload.get('architecture') if isinstance(payload, dict) else None
        cfg = load_architecture(architecture if architecture is not None else embedded)
        if embedded is not None and cfg != load_architecture(embedded):
            raise ValueError("Architecture override disagrees with the checkpoint")
        self.architecture = cfg
        self.device = torch.device(device)
        self.precision = precision if self.device.type == "cuda" else "fp32"
        self.network, diffusion = create_model_and_diffusion(**cfg)
        self.network.load_state_dict(checkpoint_weights(payload), strict=True)
        del payload
        self.network.eval().requires_grad_(False).to(device)
        self.mu, self.sigma, self.lambdas = diffusion_schedule(diffusion.alphas_cumprod, device)
        self.nfe = 0
        self.checkpoint = str(checkpoint)
        self.checkpoint_sha256 = sha256_file(checkpoint)

    @torch.no_grad()
    def prox(self, x, t):
        return self._prox(x, t)

    def _prox(self, x, t):
        self.nfe += 1
        scaled = self.mu[t] * x
        tt = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        return predict_clean(self.network, scaled, tt, self.mu, self.sigma, self.precision)

    def prox_likelihood_gradient(self, x, t, op, y):
        """Evaluate the proximal map and the input gradient of its composite likelihood.
        
        This surrogate drift differs from a likelihood evaluated at the chain state.
        Network parameters remain frozen.
        """
        for module in self.network.modules():
            if hasattr(module, 'use_checkpoint'):
                module.use_checkpoint = True
        with torch.enable_grad():
            u = x.detach().requires_grad_(True)
            p = self._prox(u, t)
            loss = (op.forward(p) - y).square().sum() / 2
            gradient = torch.autograd.grad(loss, u)[0]
        return p.detach(), gradient.detach()
