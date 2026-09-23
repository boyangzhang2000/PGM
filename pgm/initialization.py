"""Observation-only initial-state transforms, without a learned prior or solver."""
import hashlib
import math
import torch
import torch.nn.functional as F


def initialization_generator(generator):
    """Derive a reproducible seed domain without advancing the chain stream."""
    key=f'pgm.initialization:{generator.initial_seed()}'.encode('ascii')
    seed=int.from_bytes(hashlib.sha256(key).digest()[:8],'little')
    return torch.Generator(device=generator.device).manual_seed(seed)


def gaussian_smooth(x,sigma):
    radius=min(math.ceil(3*sigma),min(x.shape[-2:])-1)
    if radius < 1:raise ValueError('Initialization filtering needs spatial dimensions greater than one')
    position=torch.arange(-radius,radius+1,device=x.device,dtype=x.dtype)
    kernel=torch.exp(-position.square()/(2*sigma*sigma));kernel=kernel/kernel.sum()
    channels=x.shape[1]
    horizontal=kernel.view(1,1,1,-1).expand(channels,1,1,-1)
    vertical=kernel.view(1,1,-1,1).expand(channels,1,-1,1)
    return F.conv2d(F.pad(F.conv2d(F.pad(x,(radius,radius,0,0),mode='reflect'),horizontal,groups=channels),
                              (0,0,radius,radius),mode='reflect'),vertical,groups=channels)


def transform_initial_state(x,config,generator,initial_lambda=None):
    """Filter and scale the initial image, then add noise matched to its lambda.
    
    The filter scales the difference between the image and its Gaussian smoothing.
    The observation is unchanged. A separate generator keeps initialization and
    chain noise independent.
    """
    if config.init_filter_gain:
        x=x+config.init_filter_gain*(x-gaussian_smooth(x,config.init_filter_sigma))
    if config.init_gain!=1:
        x=x*config.init_gain
    if config.init_noise:
        if initial_lambda is None or not math.isfinite(initial_lambda) or initial_lambda<=0:
            raise ValueError('Matched initialization noise needs positive initial lambda')
        noise=torch.randn(x.shape,device=x.device,dtype=x.dtype,generator=generator)
        x=x+config.init_noise*math.sqrt(initial_lambda)*noise
    return x
