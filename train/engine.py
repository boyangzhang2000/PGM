"""Proximal fine-tuning with reproducible batches and sampling-ready exports."""
import json
import math
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from guided_diffusion.script_util import create_model_and_diffusion
from pgm.config import ROOT, dump_yaml
from pgm.model import checkpoint_weights, load_architecture, predict_clean, sha256_file
from pgm.provenance import snapshot_code
from pgm.runtime import configure_runtime, environment, resolve_device
from pgm.schedule import diffusion_schedule
from .checkpoint import (export_model, recover_history, restore_best_checkpoint, restore_rng,
                         resume_settings, rng_state, save_training, validate_progress)
from .config import resource_path
from .data import EpochBatches, ImageDataset
from .objective import bandwidth, gaussian_matching, learning_rate, perturb


def make_scaler(enabled):
    if hasattr(torch.amp, 'GradScaler'):
        return torch.amp.GradScaler('cuda', enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def gradient_norm(parameters, max_norm=None):
    gradients = [p.grad for p in parameters if p.grad is not None]
    if not gradients:
        raise RuntimeError("The training objective produced no network gradients")
    norms = [torch.linalg.vector_norm(g, dtype=torch.float64) for g in gradients]
    norm = torch.linalg.vector_norm(torch.stack(norms))
    if bool(torch.isfinite(norm)) and max_norm is not None and bool(norm > max_norm):
        for gradient in gradients:
            gradient.mul_((max_norm/norm).to(gradient.dtype))
    return norm


def train_batch(model, clean, mu, sigma, optimizer, scaler, config, precision, zeta):
    """Accumulate the batch objective before applying a network update."""
    size = len(clean)
    time = torch.randint(len(mu), (1,), device=clean.device).expand(size)
    noise = torch.randn_like(clean)
    noisy = perturb(clean, time, noise, mu, sigma)
    initial_rng = rng_state(clean.device)
    retries = 0
    while True:
        optimizer.zero_grad(set_to_none=True)
        totals = dict(loss=0., mse=0., kernel_mean=0., zero_kernel_fraction=0.)
        for offset in range(0, size, config.microbatch_size):
            part = slice(offset, offset+config.microbatch_size)
            predicted = predict_clean(model, noisy[part], time[part], mu, sigma, precision)
            losses, errors, kernels = gaussian_matching(predicted, clean[part], zeta)
            loss = losses.sum()/size
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite proximal matching loss")
            scaler.scale(loss).backward()
            totals['loss'] += float(loss.detach())
            totals['mse'] += float(errors.detach().sum())/size
            totals['kernel_mean'] += float(kernels.detach().sum())/size
            totals['zero_kernel_fraction'] += float((kernels.detach() == 0).sum())/size
        scaler.unscale_(optimizer)
        norm = gradient_norm(model.parameters(), config.grad_clip)
        if bool(torch.isfinite(norm)):
            scaler.step(optimizer)
            scaler.update()
            return dict(totals, gradient_norm=float(norm), timestep=int(time[0]), amp_retries=retries)
        if not scaler.is_enabled():
            raise FloatingPointError("Nonfinite network gradients")
        # Unscale has already registered overflow. Retry without an optimizer update.
        scaler.update()
        if scaler.get_scale() < 1:
            raise FloatingPointError("Mixed-precision gradients remain nonfinite; use full precision")
        restore_rng(initial_rng, clean.device)
        retries += 1


@torch.no_grad()
def validate(model, dataset, mu, sigma, config, device, precision):
    """Compare epochs on a fixed corruption stream independent of kernel annealing."""
    generator = torch.Generator(device=device).manual_seed(config.seed)
    loader = DataLoader(dataset, batch_size=config.optimization.microbatch_size,
                        num_workers=config.runtime.workers,
                        generator=torch.Generator().manual_seed(config.seed))
    training = model.training
    model.eval()
    squared_error, count = 0., 0
    try:
        for clean in loader:
            clean = clean.to(device)
            time = torch.randint(len(mu), (len(clean),), device=device, generator=generator)
            noise = torch.randn(clean.shape, device=device, generator=generator)
            predicted = predict_clean(model, perturb(clean, time, noise, mu, sigma), time, mu, sigma, precision)
            errors = (predicted-clean).square().flatten(1).mean(1)
            if not bool(torch.isfinite(errors).all()):
                raise FloatingPointError("Nonfinite validation prediction")
            squared_error += float(errors.sum())
            count += len(clean)
    finally:
        model.train(training)
    return dict(mse=squared_error/count, images=count)


def run_training(config, resume=None, stop_after=None):
    config.validate()
    if stop_after is not None and (type(stop_after) is not int or stop_after <= 0):
        raise ValueError("stop_after must be a positive update count")
    device = resolve_device(config.runtime.device)
    configure_runtime(device, config.runtime.threads)
    precision = config.runtime.precision if device.type == 'cuda' else 'fp32'
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(config.seed)
    architecture = load_architecture(resource_path(config.model.architecture))
    data = config.data
    dataset = ImageDataset(resource_path(data.train_dir), architecture['image_size'], data.recursive,
                           data.crop, data.horizontal_flip, config.seed)
    validation = (ImageDataset(resource_path(data.validation_dir), architecture['image_size'],
                  data.recursive, seed=config.seed) if data.validation_dir else None)
    if validation and set(dataset.paths) & set(validation.paths):
        raise ValueError("Training and validation images must be disjoint")
    signatures = dict(train=dataset.signature(), validation=validation.signature() if validation else None)
    o = config.optimization
    batches = math.ceil(len(dataset)/o.batch_size)
    total_steps = min(o.epochs*batches, o.max_steps or o.epochs*batches)
    if o.warmup_steps >= total_steps:
        raise ValueError("Learning-rate warmup must end before training completes")
    output = resource_path(config.output.directory)
    payload = torch.load(resource_path(resume), map_location='cpu', weights_only=True) if resume else None
    if payload is not None:
        if payload.get('format') != 'pgm.proximal.training':
            raise ValueError("Resume requires a complete training checkpoint")
        if (payload['resume_settings'] != resume_settings(config) or
                payload['architecture'] != architecture or payload['data_signature'] != signatures):
            raise ValueError("Resume configuration, architecture or dataset has changed")
        validate_progress(payload['state'], batches, total_steps, o.epochs)
        if output.exists() and any(output.iterdir()) and resource_path(resume) != output/'latest.pt':
            raise ValueError("Resume into an empty directory or use its own latest checkpoint")
    elif output.exists() and any(output.iterdir()):
        raise ValueError(f"Training output is not empty: {output}")
    model, diffusion = create_model_and_diffusion(**architecture)
    initial_sha = None
    if payload is not None:
        model.load_state_dict(checkpoint_weights(payload), strict=True)
    elif config.model.checkpoint:
        checkpoint = resource_path(config.model.checkpoint)
        initial = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if 'architecture' in initial and load_architecture(initial['architecture']) != architecture:
            raise ValueError("Pretrained checkpoint architecture does not match training configuration")
        model.load_state_dict(checkpoint_weights(initial), strict=True)
        initial_sha = sha256_file(checkpoint)
        del initial
    model.to(device).train().requires_grad_(True)
    for module in model.modules():
        if hasattr(module, 'use_checkpoint'):
            module.use_checkpoint = config.runtime.gradient_checkpointing
    mu, sigma, _ = diffusion_schedule(diffusion.alphas_cumprod, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=o.lr, betas=tuple(o.betas),
                                 eps=o.epsilon, weight_decay=o.weight_decay)
    scaler = make_scaler(device.type == 'cuda' and precision == 'fp16')
    state = dict(step=0, epoch=0, batch=0, best_validation_mse=None)
    if payload is not None:
        optimizer.load_state_dict(payload['optimizer'])
        if scaler.is_enabled() and payload['scaler']:
            scaler.load_state_dict(payload['scaler'])
        state = payload['state']
        restore_rng(payload['rng'], device)
        del payload
    validate_progress(state, batches, total_steps, o.epochs)
    output.mkdir(parents=True, exist_ok=True)
    if resume:
        restore_best_checkpoint(state, resource_path(resume), output)
    source_hash = snapshot_code(ROOT, output)
    (output/'config.yaml').write_text(dump_yaml(config.to_dict()), encoding='utf-8')
    metadata = dict(environment=environment(device), precision=precision, architecture=architecture,
        dataset=signatures, train_images=len(dataset), validation_images=len(validation) if validation else 0,
        total_steps=total_steps, source_sha256=source_hash, initial_checkpoint_sha256=initial_sha,
        resumed_from=str(resource_path(resume)) if resume else None,
        resume_checkpoint_sha256=sha256_file(resource_path(resume)) if resume else None)
    (output/'run.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    history_path = output/'history.jsonl'
    recover_history(history_path, state['step'])
    start_step = state['step']
    for epoch in range(state['epoch'], o.epochs):
        if state['step'] >= total_steps:
            break
        start_batch = state['batch'] if epoch == state['epoch'] else 0
        sampler = EpochBatches(len(dataset), o.batch_size, config.seed, epoch, start_batch)
        loader = DataLoader(dataset, batch_sampler=sampler, num_workers=config.runtime.workers,
            pin_memory=device.type == 'cuda', generator=torch.Generator().manual_seed(config.seed+epoch))
        for batch_index, clean in enumerate(loader, start_batch):
            zeta = bandwidth(config.kernel, state['step'], total_steps)
            lr = learning_rate(o, state['step'], total_steps)
            for group in optimizer.param_groups:
                group['lr'] = lr
            metrics = train_batch(model, clean.to(device), mu, sigma, optimizer, scaler, o, precision, zeta)
            state['step'] += 1
            last_batch = batch_index+1 == batches
            state['epoch'], state['batch'] = (epoch+1, 0) if last_batch else (epoch, batch_index+1)
            finished = state['step'] == total_steps
            stopping = stop_after is not None and state['step']-start_step >= stop_after
            record = dict(step=state['step'], epoch=epoch, bandwidth=zeta, lr=lr, **metrics)
            if validation and (last_batch or finished):
                checked = validate(model, validation, mu, sigma, config, device, precision)
                record['validation'] = checked
                if state['best_validation_mse'] is None or checked['mse'] < state['best_validation_mse']:
                    state['best_validation_mse'] = checked['mse']
                    export_model(model, architecture, output/'best.pt', state['step'])
                    state['best_checkpoint_sha256'] = sha256_file(output/'best.pt')
            with history_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record)+'\n')
            if state['step'] % config.output.log_interval == 0 or finished or stopping:
                print(json.dumps(record), flush=True)
            if state['step'] % config.output.checkpoint_interval == 0 or last_batch or finished or stopping:
                save_training(model, architecture, optimizer, scaler, config, dict(state), signatures,
                              device, output/'latest.pt')
                export_model(model, architecture, output/'proximal.pt', state['step'])
            if finished or stopping:
                result = dict(step=state['step'], complete=finished, checkpoint=str(output/'proximal.pt'),
                              device=str(device), precision=precision)
                (output/'status.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
                return result
    save_training(model, architecture, optimizer, scaler, config, dict(state), signatures,
                  device, output/'latest.pt')
    export_model(model, architecture, output/'proximal.pt', state['step'])
    result = dict(step=state['step'], complete=True, checkpoint=str(output/'proximal.pt'),
                  device=str(device), precision=precision)
    (output/'status.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result
