"""Strict training configuration with project-relative resource paths."""
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
import math
from pathlib import Path
import yaml
from pgm.config import ROOT, ConfigLoader, _matches_type


@dataclass
class ModelConfig:
    architecture: str
    checkpoint: str | None


@dataclass
class DataConfig:
    train_dir: str
    validation_dir: str | None
    recursive: bool
    crop: str
    horizontal_flip: bool


@dataclass
class OptimizationConfig:
    epochs: int
    max_steps: int | None
    batch_size: int
    microbatch_size: int
    lr: float
    weight_decay: float
    betas: list[float]
    epsilon: float
    grad_clip: float | None
    lr_schedule: str
    warmup_steps: int


@dataclass
class KernelConfig:
    start: float
    end: float
    schedule: str


@dataclass
class RuntimeConfig:
    device: str
    precision: str
    workers: int
    threads: int
    gradient_checkpointing: bool


@dataclass
class OutputConfig:
    directory: str
    checkpoint_interval: int
    log_interval: int


@dataclass
class TrainConfig:
    seed: int
    model: ModelConfig
    data: DataConfig
    optimization: OptimizationConfig
    kernel: KernelConfig
    runtime: RuntimeConfig
    output: OutputConfig

    def to_dict(self):
        return asdict(self)

    def validate(self):
        for section in (self, self.model, self.data, self.optimization, self.kernel, self.runtime, self.output):
            for entry in fields(section):
                value = getattr(section, entry.name)
                if not _matches_type(value, entry.type):
                    raise ValueError(f"Invalid type for {type(section).__name__}.{entry.name}")
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError(f"Nonfinite configuration: {entry.name}")
        o, k, r = self.optimization, self.kernel, self.runtime
        if self.seed < 0 or self.seed >= 2**32:
            raise ValueError("Seed is outside the supported range")
        if min(o.epochs, o.batch_size, o.microbatch_size, r.threads,
               self.output.checkpoint_interval, self.output.log_interval) <= 0:
            raise ValueError("Training counts must be positive")
        if o.microbatch_size > o.batch_size or r.workers < 0 or o.warmup_steps < 0:
            raise ValueError("Invalid microbatch, worker or warmup setting")
        if o.max_steps is not None and o.max_steps <= 0:
            raise ValueError("max_steps must be positive when specified")
        if o.lr <= 0 or o.epsilon <= 0 or o.weight_decay < 0:
            raise ValueError("Invalid optimizer settings")
        if len(o.betas) != 2 or any(not 0 <= b < 1 for b in o.betas):
            raise ValueError("Invalid optimizer decay factors")
        if o.grad_clip is not None and o.grad_clip <= 0:
            raise ValueError("Gradient clipping threshold must be positive")
        if not 0 < k.end <= k.start or k.schedule not in {'linear', 'geometric'}:
            raise ValueError("Kernel bandwidth must follow a positive decreasing schedule")
        if o.lr_schedule not in {'constant', 'cosine'} or r.precision not in {'fp32', 'fp16'}:
            raise ValueError("Unsupported learning-rate schedule or precision")
        if self.data.crop not in {'center', 'random'}:
            raise ValueError("Image crop must be center or random")
        for path in (self.model.architecture, self.data.train_dir, self.output.directory):
            if not path.strip():
                raise ValueError("Required paths cannot be empty")
        return self


def merge(base, updates):
    if not isinstance(updates, dict):
        raise ValueError("Training YAML must be a mapping")
    result = deepcopy(base)
    for key, value in updates.items():
        if key not in result:
            raise ValueError(f"Unknown training field: {key}")
        result[key] = merge(result[key], value) if isinstance(result[key], dict) else value
    return result


def from_dict(data):
    classes = dict(model=ModelConfig, data=DataConfig, optimization=OptimizationConfig,
                   kernel=KernelConfig, runtime=RuntimeConfig, output=OutputConfig)
    try:
        sections = {name: cls(**data[name]) for name, cls in classes.items()}
        return TrainConfig(**{**data, **sections}).validate()
    except (TypeError, KeyError) as error:
        raise ValueError(f"Invalid training configuration: {error}") from error


def load_config(path):
    defaults = yaml.load((ROOT/'configs/train/defaults.yaml').read_text(encoding='utf-8'), Loader=ConfigLoader)
    updates = yaml.load(Path(path).read_text(encoding='utf-8'), Loader=ConfigLoader)
    return from_dict(merge(defaults, updates))


def resource_path(path):
    value = Path(path).expanduser()
    return (value if value.is_absolute() else ROOT/value).resolve()
