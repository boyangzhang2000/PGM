"""PGM entry point -- picks a model, loads images, runs the configured sampler,
saves reconstructions + metrics + a side-by-side figure.

Usage
-----
    python main.py --config configs/partial_deblur.yaml
    python main.py --config configs/partial_inpaint.yaml --imgs testsets/ffhq
    python main.py --config configs/joint_sr.yaml --model-zoo /path/to/zoo

The output goes to results/<config-stem>/ with one .npy per image plus a
metrics.json and a side-by-side figure."""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch as th
import warnings
warnings.filterwarnings("ignore")

# Make sibling packages (guided_diffusion, utils, pgm) importable.
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pgm.config import load_config
from pgm.data import load_images, list_images_in_dir
from pgm.model import build_model
from pgm.schedule import build_schedule, lam_seq_of, t_of_lams, make_k_seq
from pgm.operators import (
    GaussianDeblur, BicubicSR, RandomInpaint, verify_deblur_op,
)
from pgm.samplers import sample_partial, sample_joint
from pgm.metrics import eval_metrics, make_lpips


def make_operator(task, cfg, device):
    """Build the task-specific forward operator A."""
    if task == "deblur":
        return GaussianDeblur(cfg["kernel_size"], cfg["sigma_blur"], device)
    if task == "sr":
        return BicubicSR(cfg["sf"], device)
    if task == "inpaint":
        return RandomInpaint(cfg["keep"], cfg["mask_seed"], device)
    raise ValueError(f"unknown task: {task}")


def make_t_and_k(cfg, lam_t):
    """Resolve the diffusion timestep sequence and per-stage K_t from the config."""
    if "t_seq" in cfg:                       # explicit t list
        t_seq = list(cfg["t_seq"])
    else:                                    # lambda-geometric -> t
        t_seq = t_of_lams(lam_seq_of(cfg["lam_max"], cfg["lam_min"], cfg["T"]),
                           lam_t)
    if "K_seq" in cfg:                       # explicit per-stage inner iterations
        K_seq = [int(k) for k in cfg["K_seq"]]
    else:
        K_mode = cfg.get("k_mode", "us")
        K_seq = make_k_seq(t_seq, K_mode, cfg["K_peak"], cfg.get("K_mid"))
    return t_seq, K_seq


def run_one(model, diffusion, gt, cfg, device, lp, out_dir):
    """Run one image through the configured sampler, save npy + metrics."""
    task = cfg["task"]
    method = cfg["method"]
    op = make_operator(task, cfg, device)
    ac, mu_t, sigma2_t, sigma_t, lam_t = build_schedule(diffusion, device)
    t_seq, K_seq = make_t_and_k(cfg, lam_t)

    # Build the noisy/observed measurement y.
    sig = float(cfg["sig_n"])
    th.manual_seed(42)
    yA = op.A(gt)
    y = yA + (sig * th.randn_like(yA) if sig > 0 else 0.0)

    # APL (paper) sampler parameters.
    gamma = float(cfg.get("gamma", 1.0))
    gamma_seq = cfg.get("gamma_seq", None)      # per-stage weights (gamma annealing)
    if gamma_seq is not None:
        gamma_seq = [float(g) for g in gamma_seq]
    grad_ref = float(cfg.get("grad_ref", 0.1))
    delta_mode = cfg.get("delta_mode", "lambda")
    delta_c = float(cfg.get("delta_c", 0.5))
    delta_cap = float(cfg.get("delta_cap", 0.3))
    delta_scale = float(cfg.get("delta_scale", 1.0))
    final_prox = bool(cfg.get("final_prox", False))
    final_t = cfg.get("final_t", None)          # decoupled final-prox timestep
    if final_t is not None:
        final_t = int(final_t)
    clamp_x = cfg.get("clamp_x", None)
    if clamp_x is not None:
        clamp_x = (float(clamp_x[0]), float(clamp_x[1]))

    # Run the sampler (annealed proximal Langevin in image space).
    t0 = time.time()
    if method == "partial":
        x = sample_partial(
            model, op, y, mu_t, sigma_t, lam_t, t_seq, K_seq, sig,
            gamma=gamma, gamma_seq=gamma_seq, grad_ref=grad_ref,
            delta_mode=delta_mode, delta_c=delta_c, delta_cap=delta_cap,
            delta_scale=delta_scale, seed=42, device=device,
            init=cfg.get("init", "backproj"), final_prox=final_prox,
            final_t=final_t, clamp_x=clamp_x,
        )
    elif method == "joint":
        x = sample_joint(
            model, op, y, mu_t, sigma_t, lam_t, t_seq, K_seq, sig,
            gamma=gamma, gamma_seq=gamma_seq, grad_ref=grad_ref,
            delta_mode=delta_mode, delta_c=delta_c, delta_cap=delta_cap,
            delta_scale=delta_scale, seed=42, device=device,
            init=cfg.get("init", "gauss"), final_prox=final_prox,
            final_t=final_t, clamp_x=clamp_x,
        )
    else:
        raise ValueError(f"unknown method: {method}")
    elapsed = time.time() - t0

    if task == "inpaint":
        x = x * (1 - op.mask) + y * op.mask

    # Compute metrics.
    xc = x.clamp(-1, 1)
    metrics = eval_metrics(xc, gt, lp)

    # Save.
    os.makedirs(out_dir, exist_ok=True)
    return dict(t_seq=t_seq, K_seq=K_seq, elapsed=elapsed,
                y=y.clamp(-1, 1).cpu().numpy(),
                x=xc.cpu().numpy(),
                mask=op.mask.cpu().numpy() if task == "inpaint" else None,
                metrics=metrics)


def save_figure(out_dir, basename, gt_np, y_np, x_np, mask_np=None, psnr=None):
    """Save a simple side-by-side figure (numpy -> matplotlib)."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    def to_img(a):
        a = a.squeeze()
        if a.ndim == 3:
            a = a.transpose(1, 2, 0)
        return ((a + 1) / 2).clip(0, 1)

    cols = ["GT", "y"]
    imgs = [to_img(gt_np), to_img(y_np)]
    if mask_np is not None:
        cols.append("mask")
        imgs.append(mask_np.squeeze())
    cols.append("recon")
    imgs.append(to_img(x_np))

    fig, axes = plt.subplots(1, len(cols), figsize=(3.0 * len(cols), 3.2))
    for c, (im, ax) in enumerate(zip(imgs, axes)):
        ax.imshow(im)
        ax.set_xticks([])
        ax.set_yticks([])
        title = cols[c]
        if c == len(cols) - 1 and psnr is not None:
            title += f"  (PSNR {psnr:.2f})"
        ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{basename}.png"), dpi=110, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/partial_inpaint.yaml", help="path to a YAML config")
    ap.add_argument("--imgs", default="testsets/ffhq",
                    help="image file or directory (default: testsets/ffhq)")
    ap.add_argument("--model-zoo", default="./model_zoo", help="directory containing {model_name}.pt")
    ap.add_argument("--out", default=None, help="output directory (default: results/<config-stem>)")
    args = ap.parse_args()
    args.device = "cuda" if th.cuda.is_available() else "cpu"

    cfg = load_config(args.config)
    out_dir = args.out or os.path.join("results", os.path.splitext(os.path.basename(args.config))[0])
    os.makedirs(out_dir, exist_ok=True)
    json.dump(cfg, open(os.path.join(out_dir, "config.json"), "w"), indent=2)

    # Build model + diffusion.
    model, diffusion = build_model(cfg["model_name"], args.model_zoo, args.device)

    # Collect image paths.
    if os.path.isdir(args.imgs):
        paths = list_images_in_dir(args.imgs)
    else:
        paths = [args.imgs]
    gt = load_images(paths, args.device)
    print(f"[main] {len(paths)} images, task={cfg['task']}, method={cfg['method']}, "
          f"model={cfg['model_name']}")

    lp = make_lpips(args.device)
    all_metrics = []
    for i, path in enumerate(paths):
        name = os.path.splitext(os.path.basename(path))[0]
        result = run_one(model, diffusion, gt[i:i + 1], cfg, args.device, lp, out_dir)
        # Save per-image arrays + figure.
        np.save(os.path.join(out_dir, f"{name}_y.npy"), result["y"])
        np.save(os.path.join(out_dir, f"{name}_x.npy"), result["x"])
        if result["mask"] is not None:
            np.save(os.path.join(out_dir, f"{name}_mask.npy"), result["mask"])
        m = result["metrics"][0]
        all_metrics.append(dict(image=name, **m,
                               t_seq=result["t_seq"], K_seq=result["K_seq"],
                               elapsed_sec=round(result["elapsed"], 2)))
        save_figure(out_dir, name, gt[i].cpu().numpy(), result["y"], result["x"],
                    mask_np=result["mask"], psnr=m["psnr"])
        print(f"[{name}] psnr={m['psnr']:.2f}  psnr_y={m['psnr_y']:.2f}  "
              f"ssim={m['ssim']:.4f}  lpips={m['lpips']:.4f}  "
              f"(t={result['t_seq']}, K={result['K_seq']}, {result['elapsed']:.1f}s)")

    # Aggregate metrics.
    summary = {
        "config": os.path.basename(args.config),
        "model": cfg["model_name"],
        "task": cfg["task"],
        "method": cfg["method"],
        "n_images": len(paths),
        "mean_psnr": round(float(np.mean([m["psnr"] for m in all_metrics])), 2),
        "mean_psnr_y": round(float(np.mean([m["psnr_y"] for m in all_metrics])), 2),
        "mean_ssim": round(float(np.mean([m["ssim"] for m in all_metrics])), 4),
        "mean_lpips": round(float(np.mean([m["lpips"] for m in all_metrics])), 4),
        "per_image": all_metrics,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] saved -> {out_dir}")
    print(f"[summary] mean PSNR={summary['mean_psnr']}  "
          f"SSIM={summary['mean_ssim']}  LPIPS={summary['mean_lpips']}")


if __name__ == "__main__":
    main()
