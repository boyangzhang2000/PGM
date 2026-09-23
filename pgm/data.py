"""Deterministic image discovery and normalized RGB tensors."""
from pathlib import Path
import numpy as np
import torch
from PIL import Image


def image_paths(source):
    source = Path(source)
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    paths = (
        sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in extensions)
        if source.is_dir() else [source]
    )
    if not paths or any(not p.is_file() for p in paths):
        raise FileNotFoundError(f"No readable images: {source}")
    if len({p.stem for p in paths}) != len(paths):
        raise ValueError("Image stems must be unique to avoid overwriting results")
    return paths


def load_image(path, device="cpu", size=256):
    with Image.open(path) as source:
        im = source.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
        array = np.array(im, dtype=np.float32) / 127.5 - 1
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def to_pil(tensor):
    array = tensor.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    return Image.fromarray(np.rint((array.clip(-1, 1) + 1) * 127.5).astype(np.uint8))
