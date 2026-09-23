"""Device selection and reproducible, local random streams."""
from contextlib import nullcontext
import platform
import torch


def resolve_device(requested="auto"):
    if requested == "auto":
        # Runtime availability and the visible-device list can disagree under
        # masking (notably an empty CUDA_VISIBLE_DEVICES on Windows).
        usable = torch.cuda.is_available() and torch.cuda.device_count() > 0
        return torch.device("cuda:0" if usable else "cpu")
    device = torch.device(requested)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Supported devices: auto, cpu, cuda, cuda:N")
    if device.type == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            raise RuntimeError("CUDA unavailable. Use --device auto or cpu.")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if device.index >= torch.cuda.device_count():
            raise ValueError(f"Unavailable device: {device}")
    return device


def configure_runtime(device, threads=4):
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.type == "cuda":
        torch.cuda.set_device(device)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def autocast_context(device, precision):
    return (torch.autocast("cuda", dtype=torch.float16)
            if device.type == "cuda" and precision == "fp16" else nullcontext())


def environment(device):
    return {"python": platform.python_version(), "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda, "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor()}
