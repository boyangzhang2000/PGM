"""FFHQ-10M checkpoint loader (epsilon-prediction U-Net).

Both `prox_ffhq_10m` (proximal matching -> Tweedie mode) and
`diffusion_ffhq_10m` (denoising score matching -> Tweedie mean) are
epsilon-prediction networks and share the same architecture config.
"""
import os
import sys
import torch as th

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from guided_diffusion.script_util import (
    model_and_diffusion_defaults, create_model_and_diffusion, args_to_dict,
)
from utils.utils_model import create_argparser


DEFAULT_MODEL_ZOO = os.path.join(ROOT, "model_zoo")


def build_model(name, model_zoo_dir=None, device="cuda"):
    """Load a pretrained FFHQ-10M U-Net (CPU -> device).

    name: 'prox_ffhq_10m' or 'diffusion_ffhq_10m'
    model_zoo_dir: directory containing {name}.pt (defaults to ./model_zoo)
    """
    if model_zoo_dir is None:
        model_zoo_dir = DEFAULT_MODEL_ZOO
    cfg = dict(
        model_path=os.path.join(model_zoo_dir, f"{name}.pt"),
        image_size=256, in_channels=3,
        num_channels=128, num_res_blocks=1, attention_resolutions="16",
        dropout=0.1, learn_sigma=True, diffusion_steps=1000,
        noise_schedule="linear", resblock_updown=True,
        use_scale_shift_norm=True, channel_mult="", use_checkpoint=False,
        use_fp16=False, class_cond=False, use_kl=False,
        predict_xstart=False, timestep_respacing="", num_heads=4,
        num_heads_upsample=-1, num_head_channels=64,
        use_new_attention_order=False, rescale_timesteps=False,
        rescale_learned_sigmas=False, out_channels=None,
    )
    margs = create_argparser(cfg).parse_args([])
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(margs, model_and_diffusion_defaults().keys()))
    model.load_state_dict(th.load(cfg["model_path"], map_location="cpu"))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device), diffusion


def model_x0(model, x, t, mu_t, sigma_t):
    """Tweedie x0 = (x - sigma(t) * eps_theta) / mu(t), clipped to [-1, 1].
    Both prox and score FFHQ-10M checkpoints are epsilon-prediction networks, so
    the same Tweedie path applies; the qualitative difference (mode vs mean) is
    learned by the network, not by this routine."""
    tt = th.full((x.shape[0],), int(t), device=x.device, dtype=th.long)
    out = model(x, tt)
    out = out[:, :3] if out.shape[1] == 6 else out
    return ((x - sigma_t[int(t)] * out) / mu_t[int(t)]).clamp(-1, 1)
