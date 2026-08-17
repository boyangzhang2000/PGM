"""YAML config loader (one config per <method>_<task>.yaml)."""
import yaml


def load_config(path):
    """Read a YAML file and return a dict. Adds a few convenience aliases."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    # convenience aliases
    if "method" not in cfg:
        cfg["method"] = "partial"          # default
    if "model_name" not in cfg:
        cfg["model_name"] = "prox_ffhq_10m"
    if "sig_n" not in cfg and "noise_level_img" in cfg:
        # older yamls used noise_level_img on [0,1] (0.05 == 0.1 on [-1,1])
        cfg["sig_n"] = float(cfg["noise_level_img"]) / 127.5
    return cfg


def merge(a, b):
    """Shallow dict merge (b wins)."""
    out = dict(a)
    out.update(b)
    return out
