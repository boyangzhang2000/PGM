"""Portable network exports and complete resumable optimizer snapshots."""
import random
import json
import math
import shutil
import numpy as np
import torch


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def save_atomic(payload, path):
    temporary = path.with_name(path.name+'.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def export_model(model, architecture, path, step):
    save_atomic(dict(format='pgm.proximal', architecture=architecture,
                     state_dict=cpu_tree(model.state_dict()), step=step), path)


def rng_state(device):
    state = np.random.get_state()
    return dict(python=random.getstate(), numpy=(state[0], torch.tensor(state[1].astype(np.int64)),
                *state[2:]), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)


def restore_rng(state, device):
    random.setstate(state['python'])
    numpy_state = state['numpy']
    np.random.set_state((numpy_state[0], numpy_state[1].numpy().astype(np.uint32), *numpy_state[2:]))
    torch.set_rng_state(state['torch'])
    if device.type == 'cuda' and state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'], device)


def resume_settings(config):
    settings = config.to_dict()
    settings.pop('output')
    for key in ('device', 'workers', 'threads'):
        settings['runtime'].pop(key)
    return settings


def validate_progress(state, batches, total_steps, epochs):
    if not isinstance(state, dict):
        raise ValueError("Training progress must be a mapping")
    if any(type(state.get(key)) is not int for key in ('step', 'epoch', 'batch')):
        raise ValueError("Training progress must contain integer counters")
    if not (0 <= state['step'] <= total_steps and 0 <= state['epoch'] <= epochs
            and 0 <= state['batch'] < batches
            and state['step'] == state['epoch']*batches+state['batch']):
        raise ValueError("Inconsistent training progress or data cursor")
    score = state.get('best_validation_mse')
    if score is not None and (type(score) not in (int, float) or not math.isfinite(score) or score < 0):
        raise ValueError("Invalid saved validation score")


def recover_history(path, step):
    if not path.exists():
        return
    lines = path.read_text(encoding='utf-8').splitlines(keepends=True)
    records = []
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines)-1 and not line.endswith('\n'):
                break
            raise ValueError("Training history contains a corrupt complete record") from None
        if row['step'] <= step:
            records.append(row)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(''.join(json.dumps(row)+'\n' for row in records), encoding='utf-8')
    temporary.replace(path)


def restore_best_checkpoint(state, source, output):
    if source.parent == output or state.get('best_validation_mse') is None:
        return
    from pgm.model import sha256_file
    previous = source.parent/'best.pt'
    digest = state.get('best_checkpoint_sha256')
    if not digest or not previous.is_file() or sha256_file(previous) != digest:
        raise ValueError("Matching best checkpoint is unavailable; resume in the original output directory")
    temporary = output/'best.pt.tmp'
    shutil.copyfile(previous, temporary)
    temporary.replace(output/'best.pt')


def save_training(model, architecture, optimizer, scaler, config, state, data_signature, device, path):
    save_atomic(dict(format='pgm.proximal.training', architecture=architecture,
        state_dict=cpu_tree(model.state_dict()), optimizer=cpu_tree(optimizer.state_dict()),
        scaler=scaler.state_dict(), config=config.to_dict(), resume_settings=resume_settings(config),
        state=state, data_signature=data_signature, rng=rng_state(device)), path)
