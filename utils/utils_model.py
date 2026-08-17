"""Minimal argparser helper for guided_diffusion U-Net config."""
import argparse
from guided_diffusion.script_util import add_dict_to_argparser


def create_argparser(model_config):
    """Build an argparse.Namespace with the FFHQ-10M checkpoint defaults merged
    with the user-supplied overrides in `model_config`."""
    defaults = dict(
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="",
        diffusion_steps=1000,
        noise_schedule="linear",
        num_head_channels=64,
        resblock_updown=True,
        use_fp16=False,
        use_scale_shift_norm=True,
        num_heads=4,
        num_heads_upsample=-1,
        use_new_attention_order=False,
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        channel_mult="",
        learn_sigma=True,
        class_cond=False,
        use_checkpoint=False,
        image_size=256,
        num_channels=128,
        num_res_blocks=1,
        attention_resolutions="16",
        dropout=0.1,
    )
    defaults.update(model_config)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser
