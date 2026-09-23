"""Small command-line interface; all algorithm parameters come from YAML."""
import argparse
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from .artifacts import write_csv, write_json
from .config import METHODS, ROOT, SCHEMES, TASKS, dump_yaml, load_config, project_path
from .data import image_paths
from .experiment import run_image, summarize
from .metrics import make_lpips
from .model import ProximalModel
from .runtime import configure_runtime, environment, resolve_device


def smoke_config(cfg):
    """Explicitly reduced budget for execution checks; never a quality benchmark."""
    indices = sorted(set([0, len(cfg.sampler.timesteps) // 2, len(cfg.sampler.timesteps) - 1]))
    return replace(cfg, sampler=replace(cfg.sampler,
        timesteps=[cfg.sampler.timesteps[i] for i in indices], steps=[1] * len(indices),
        gamma=[cfg.sampler.gamma[i] for i in indices]),
        refinement=replace(cfg.refinement,steps=min(1,cfg.refinement.steps)),
        consistency=replace(cfg.consistency, polish_steps=min(1, cfg.consistency.polish_steps),
            adam_steps=min(1, cfg.consistency.adam_steps), final_adam_steps=min(1, cfg.consistency.final_adam_steps)))


def main():
    parser = argparse.ArgumentParser(description="Annealed proximal Langevin image restoration")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--scheme", choices=SCHEMES)
    parser.add_argument("--imgs", type=Path, default=ROOT / "testsets/ffhq")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--architecture", type=Path)
    parser.add_argument("--precision", choices=["fp32", "fp16"])
    parser.add_argument("--nl-engine")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--observation-seed", type=int)
    parser.add_argument(
        "--image-offset", type=int, default=0,
        help="Starting image index for matching seeds/masks when replaying a subset",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--lpips", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print validated resolved YAML without loading weights")
    args = parser.parse_args()
    if args.image_offset < 0:
        parser.error("--image-offset must be nonnegative")
    if args.config and (args.task or args.method):
        parser.error("--config cannot be combined with --task/--method")
    cfg = load_config(args.config, task=args.task, method=args.method, scheme=args.scheme)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.observation_seed is not None:
        cfg.observation_seed = args.observation_seed
    if args.checkpoint:
        cfg.model.checkpoint = str(args.checkpoint.resolve())
    if args.architecture:
        cfg.model.architecture = str(args.architecture.resolve())
    if args.precision:
        cfg.model.precision = args.precision
    if args.nl_engine:
        cfg.operator.engine_dir = str(Path(args.nl_engine).resolve())
    cfg = smoke_config(cfg) if args.smoke else cfg
    cfg.validate()
    if args.dry_run:
        print(dump_yaml(cfg.to_dict()))
        return
    device = resolve_device(args.device)
    configure_runtime(device, args.threads)
    paths = image_paths(args.imgs)
    out = args.out or ROOT / "results" / f"{datetime.now():%Y%m%d_%H%M%S}_{cfg.method}_{cfg.task}_{cfg.sampler.scheme}"
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        parser.error(f"Output directory is not empty: {out}; select a new --out")
    from .provenance import snapshot_code
    source_hash = snapshot_code(ROOT, out)
    write_json(out / "config.json", cfg.to_dict())
    (out / "config.yaml").write_text(dump_yaml(cfg.to_dict()), encoding="utf-8")
    model = ProximalModel(project_path(cfg.model.checkpoint), device, cfg.model.precision, cfg.model.architecture)
    lp, lp_status = make_lpips(device, args.lpips)
    regularizer = lp
    if regularizer is None and cfg.refinement.steps and cfg.refinement.lpips:
        regularizer, _ = make_lpips(device,'on')
    metadata = {"environment": environment(device), "profile": "smoke" if args.smoke else "full", "lpips": lp_status,
                "source_sha256": source_hash}
    print(
        f"{cfg.method}/{cfg.task}/{cfg.sampler.scheme}: {len(paths)} images, "
        f"{device}, {sum(cfg.sampler.steps)} chain steps", flush=True,
    )
    records = []
    for i, path in enumerate(paths):
        def progress(stage):
            if stage["stage"] % 10 == 0:
                print(f"  {path.name}: stage={stage['stage']} t={stage['t']}", flush=True)
        record = run_image(
            model, cfg, path, device, lp, i + args.image_offset,
            output=out / path.stem, progress=progress, refinement_model=regularizer,
        )
        records.append(record)
        write_json(out / "metrics.json", {**metadata, "summary": summarize(records), "records": records})
        write_csv(out / "metrics.csv", records)
        print(
            f"  {path.name}: PSNR={record['psnr']:.3f}, "
            f"SSIM={record['ssim']:.4f}, {record['elapsed_sec']:.1f}s", flush=True,
        )
    print(f"Saved: {out.resolve()}")
