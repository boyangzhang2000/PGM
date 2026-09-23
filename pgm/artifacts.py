"""Atomic metadata and lossless numeric outputs with compact image previews."""
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from .data import to_pil


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def write_csv(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "task", "method", "scheme", "image", "psnr", "ssim", "lpips",
        "residual_rms", "elapsed_sec", "nfe", "device", "status",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def save_reconstruction(directory, gt, y, raw, reconstruction, state, mask=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {"observation": y, "raw_estimate": raw, "reconstruction": reconstruction, "state": state}
    if mask is not None:
        arrays["mask"] = mask
    for name, tensor in arrays.items():
        np.save(directory / f"{name}.npy", tensor.detach().cpu().numpy())
    to_pil(reconstruction).save(directory / "reconstruction.png")
    height, width = gt.shape[-2:]
    tiles = [("Ground truth", to_pil(gt))]
    if y.shape == gt.shape or y.shape[-1] < gt.shape[-1]:
        tiles.append(("Observation", to_pil(y).resize((width, height), Image.Resampling.NEAREST)))
    tiles.append(("Reconstruction", to_pil(reconstruction)))
    preview = Image.new("RGB", (width * len(tiles), height + 28), "#f4f6f8")
    draw = ImageDraw.Draw(preview)
    for i, (name, tile) in enumerate(tiles):
        preview.paste(tile, (width * i, 28))
        draw.text((width * i + 10, 8), name, fill="#17212b")
    preview.save(directory / "comparison.png")
