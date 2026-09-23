"""Reproducible experiment orchestration, independent of the CLI."""
import hashlib
from pathlib import Path
import time
import torch
from .artifacts import save_reconstruction, write_json
from .data import load_image
from .metrics import evaluate
from .nonlinear import NonlinearDeblurOperator, PhaseRetrievalOperator
from .operators import BicubicSR, GaussianDeblur, RandomInpaint
from .runtime import synchronize
from .refinement import refine_image
from .consistency import finalize_reconstruction
from .samplers import sample


def make_operator(cfg, device, image_index=0, engine_dir=None, size=256):
    o = cfg.operator
    if cfg.task == "deblur":
        return GaussianDeblur(o.kernel_size, o.blur_sigma, device, size)
    if cfg.task == "sr":
        return BicubicSR(o.scale_factor, device, size)
    if cfg.task == "inpaint":
        return RandomInpaint(o.missing_fraction, o.mask_seed + image_index * o.mask_seed_step, device, size)
    if cfg.task == "pr":
        return PhaseRetrievalOperator(o.oversample, device, size, o.fft_device)
    if cfg.task == "ndb":
        return NonlinearDeblurOperator(o.options, engine_dir or o.engine_dir, device,
                                       o.kernel_seed, o.kernel_scale, size)
    raise ValueError(f"Unknown task: {cfg.task}")


@torch.no_grad()
def observation(op, gt, noise_std, seed):
    clean = op.forward(gt).detach()
    # Share the noise stream across devices; operator roundoff can still differ.
    noise = torch.randn(clean.shape, generator=torch.Generator().manual_seed(seed))
    return clean + noise_std * noise.to(clean.device)


def tensor_hash(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def run_image(model, cfg, path, device, lpips_model=None, image_index=0,
              engine_dir=None, output=None, op=None, progress=None, refinement_model=None):
    cfg.validate()
    size = getattr(model, 'architecture', {}).get('image_size', 256)
    gt = load_image(path, device, size)
    op = op or make_operator(cfg, device, image_index, engine_dir, size)
    obs_seed = cfg.observation_seed + image_index * cfg.observation_seed_step
    seed = cfg.seed + image_index * cfg.seed_step
    y = observation(op, gt, cfg.operator.noise_std, obs_seed)
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    sampled = sample(model, op, y, cfg, seed, progress)
    synchronize(device)
    sampling_elapsed = time.perf_counter() - start
    initial = sampled.estimate.clamp(-1,1)
    refinement_start = time.perf_counter()
    prior_before_refinement = model.nfe
    perceptual = refinement_model if refinement_model is not None else lpips_model
    if cfg.refinement.steps and cfg.refinement.lpips and perceptual is None:
        from .metrics import make_lpips
        perceptual, _ = make_lpips(device,'on')
    refined = refine_image(op,initial,y,cfg.refinement,perceptual)
    assert model.nfe == prior_before_refinement, 'Refinement must not call the proximal model'
    synchronize(device)
    refinement_elapsed = time.perf_counter()-refinement_start
    elapsed = time.perf_counter() - start
    raw = refined.estimate
    result = finalize_reconstruction(op, raw, y, cfg)
    metrics = evaluate(result, gt, lpips_model)
    raw_metrics = evaluate(raw, gt) if cfg.task == "inpaint" else {k: v for k, v in metrics.items() if k != "lpips"}
    with torch.no_grad():
        residual = float((op.forward(result) - y).square().mean().sqrt())
    record = {"status": "passed", "task": cfg.task, "method": cfg.method,
              "scheme": cfg.sampler.scheme, "image": Path(path).name,
              "image_index": image_index,
              "image_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
              "device": str(device), "precision": model.precision, "seed": seed,
              "observation_seed": obs_seed, "observation_sha256": tensor_hash(y),
              "mask_sha256": tensor_hash(op.mask) if hasattr(op, "mask") else None,
              "observation_shape": list(y.shape), "elapsed_sec": elapsed,
              "nfe": sampled.nfe, "evaluations": sampled.evaluations, "residual_rms": residual,
              "sampling_elapsed_sec": sampling_elapsed, "refinement_elapsed_sec": refinement_elapsed,
              "refinement_evaluations": refined.evaluations, "refinement_trace": refined.trace,
              "sampling_metrics": evaluate(initial,gt),
              "peak_cuda_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
              "raw_metrics": raw_metrics, **metrics, "trace": sampled.trace,
              "config": cfg.to_dict(), "checkpoint_sha256": model.checkpoint_sha256}
    if cfg.task == "ndb":
        record["ndb_engine"] = {"root": op.engine_root, "checkpoint": op.checkpoint}
    if output:
        import numpy as np
        save_reconstruction(output, gt, y, raw, result, sampled.state, getattr(op, "mask", None))
        np.save(Path(output)/'sampling_estimate.npy',initial.cpu().numpy())
        write_json(Path(output) / "metrics.json", record)
    return record


def summarize(records):
    passed = [r for r in records if r["status"] == "passed"]
    means = {}
    for key in ("psnr", "psnr_y", "ssim", "lpips", "elapsed_sec"):
        values = [r[key] for r in passed if r.get(key) is not None]
        means[key] = sum(values) / len(values) if values else None
    return {"total": len(records), "passed": len(passed), "failed": len(records) - len(passed), "means": means}
