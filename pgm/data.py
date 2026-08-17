"""Image loading: RGB -> [-1, 1] torch tensors at 256x256."""
from pathlib import Path
import numpy as np
import torch as th
from PIL import Image


def load_images(paths, device="cuda"):
    """Load a list of image file paths and return a (N, 3, 256, 256) tensor on `device`.

    Bicubic-resize to 256x256, scale to [-1, 1]."""
    imgs = []
    for p in paths:
        im = Image.open(p).convert("RGB").resize((256, 256), Image.BICUBIC)
        arr = np.asarray(im).astype(np.float32) / 255.0 * 2 - 1
        imgs.append(th.from_numpy(arr).permute(2, 0, 1))
    return th.stack(imgs).to(device)


def list_images_in_dir(dir_path):
    """Return a sorted list of image paths under `dir_path`."""
    p = Path(dir_path)
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    return sorted([str(f) for f in p.iterdir() if f.suffix.lower() in exts])
