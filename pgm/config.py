"""Strict YAML configuration resolved relative to the project."""
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
import math
from pathlib import Path
from types import UnionType
from typing import get_args, get_origin
import yaml

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("deblur", "inpaint", "sr", "ndb", "pr")
METHODS = ("joint", "partial")
SCHEMES = ("em", "exp", "imp")


class ConfigLoader(yaml.SafeLoader):
    """Reject duplicate YAML keys instead of silently taking the last value."""


def _unique_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"Duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


ConfigLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


class ConfigDumper(yaml.SafeDumper):
    """Readable compact arrays while retaining block mappings."""


ConfigDumper.add_representer(list, lambda dumper, value:
    dumper.represent_sequence('tag:yaml.org,2002:seq', value, flow_style=True))


def dump_yaml(data):
    return yaml.dump(data, Dumper=ConfigDumper, sort_keys=False, allow_unicode=True, width=100)


def _matches_type(value, hint):
    origin = get_origin(hint)
    if origin is UnionType:
        return any(_matches_type(value, item) for item in get_args(hint))
    if origin is list:
        return isinstance(value, list) and all(
            _matches_type(item, get_args(hint)[0]) for item in value
        )
    if hint is float:
        return type(value) in (float, int)
    if hint in (int, bool):
        return type(value) is hint
    return isinstance(value, hint)


@dataclass
class ModelConfig:
    checkpoint: str = "model_zoo/prox_ffhq_10m.pt"
    precision: str = "fp32"
    architecture: str | None = None


@dataclass
class OperatorConfig:
    noise_std: float = 0.05
    kernel_size: int = 61
    blur_sigma: float = 3.0
    scale_factor: int = 4
    missing_fraction: float = 0.7
    mask_seed: int = 7
    mask_seed_step: int = 100
    oversample: float = 2.0
    fft_device: str = "auto"
    engine_dir: str | None = None
    options: str = "configs/operators/ndb.yaml"
    kernel_seed: int = 0
    kernel_scale: float = 1.2


@dataclass
class SamplerConfig:
    scheme: str = "em"
    timesteps: list[int] = field(default_factory=list)
    steps: list[int] = field(default_factory=list)
    gamma: list[float] = field(default_factory=list)
    likelihood_mode: str = "direct"
    temperature: float = 1.0
    temperature_mode: str = "constant"
    temperature_power: float = 0.0
    guidance_rms_cap: float | None = None
    guidance_pixel_cap: float | None = None
    confinement: float = 1.0
    preconditioner: str = "none"
    curvature_probes: int = 4
    curvature_smoothing: int = 5
    data_direction: str = "gradient"
    data_location: str = "state"
    adam_betas: list[float] = field(default_factory=lambda: [0.9, 0.999])
    adam_epsilon: float = 1e-8
    history_size: int = 10
    initial_inverse_curvature: float = 0.001
    grad_scale: float = 0.1
    delta_mode: str = "lambda"
    delta_scale: float = 0.55
    delta_cap: float = 0.3
    delta_c: float = 0.5
    init: str = "backproj"
    init_gain: float = 1.0
    init_noise: float = 0.0
    init_filter_sigma: float = 0.0
    init_filter_gain: float = 0.0
    gauss_scale: float = 1.0
    clamp: list[float] | None = field(default_factory=lambda: [-2.0, 2.0])
    final_prox: bool = True
    final_t: int = 13
    readout: str = "prox"
    average_window: int = 10


@dataclass
class ConsistencyConfig:
    amplitude: bool = False
    amplitude_keep: float = 0.98
    amplitude_from: int = 0
    phase_coupling: float = 0.0
    polish_steps: int = 0
    polish_lr: float = 0.001
    adam_steps: int = 0
    adam_lr: float = 0.001
    adam_from: int = 0
    final_amplitude: bool = False
    final_adam_steps: int = 0
    final_adam_lr: float = 0.01
    paste_observed: bool = True


@dataclass
class RefinementConfig:
    steps: int = 0
    lr: float = 0.01
    data: float = 1.0
    lpips: float = 0.0
    tv: float = 0.0
    optimizer: str = 'adam'
    tv_epsilon: float = 0.0
    history_size: int = 10
    max_evaluations: int = 1000


@dataclass
class ExperimentConfig:
    task: str
    method: str
    model: ModelConfig = field(default_factory=ModelConfig)
    operator: OperatorConfig = field(default_factory=OperatorConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    consistency: ConsistencyConfig = field(default_factory=ConsistencyConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    seed: int = 42
    observation_seed: int = 1234
    seed_step: int = 1000
    observation_seed_step: int = 1000
    provenance: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    def validate(self):
        for section in (self, self.model, self.operator, self.sampler, self.consistency, self.refinement):
            for entry in fields(section):
                if not _matches_type(getattr(section, entry.name), entry.type):
                    raise ValueError(f"Invalid type for {type(section).__name__}.{entry.name}")
        s, o, c = self.sampler, self.operator, self.consistency
        r = self.refinement
        if not 0 <= r.steps <= 100:
            raise ValueError('Post-sampling refinement supports 0 to 100 optimizer steps')
        if not math.isfinite(r.lr) or r.lr <= 0:
            raise ValueError('Refinement learning rate must be finite and positive')
        if r.optimizer not in ('adam','lbfgs'):
            raise ValueError('Refinement optimizer must be adam or lbfgs')
        if not math.isfinite(r.tv_epsilon) or r.tv_epsilon < 0 or r.history_size < 1:
            raise ValueError('Invalid refinement TV smoothing or optimizer history')
        minimum_evaluations=(r.steps*(2 if r.optimizer=='lbfgs' else 1)+1) if r.steps else 0
        if r.max_evaluations < max(1,minimum_evaluations):
            raise ValueError('Refinement objective budget cannot cover the requested steps')
        if any(not math.isfinite(v) or v < 0 for v in (r.data,r.lpips,r.tv)) or r.data == 0:
            raise ValueError('Refinement requires positive data weight and nonnegative regularization')
        if r.steps and c.final_adam_steps:
            raise ValueError('Use refinement or final consistency optimization, not both')
        if self.task not in TASKS or self.method not in METHODS or s.scheme not in SCHEMES:
            raise ValueError("Unknown task, method, or scheme")
        n = len(s.timesteps)
        if not n or len(s.steps) != n or len(s.gamma) != n:
            raise ValueError("timesteps, steps and gamma must have equal nonzero lengths")
        if any(type(t) is not int or not 0 <= t < 1000 for t in s.timesteps + [s.final_t]):
            raise ValueError("Timesteps must be integers in [0, 999]")
        if any(a < b for a, b in zip(s.timesteps, s.timesteps[1:])):
            raise ValueError("Annealing timesteps must be nonincreasing")
        if any(type(k) is not int or k < 1 for k in s.steps):
            raise ValueError("Every stage needs positive integer steps")
        positive = [s.grad_scale, s.delta_scale, s.delta_cap, s.delta_c, o.blur_sigma, o.oversample]
        if any(not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in positive):
            raise ValueError("Scales and step sizes must be finite and positive")
        if any(not math.isfinite(x) or x < 0 for x in s.gamma + [o.noise_std, s.gauss_scale, o.kernel_scale, s.temperature, s.temperature_power, s.confinement]):
            raise ValueError("gamma, noise and Gaussian scales must be finite and nonnegative")
        if s.guidance_rms_cap is not None and (not math.isfinite(s.guidance_rms_cap) or s.guidance_rms_cap <= 0):
            raise ValueError('guidance_rms_cap must be null or finite and positive')
        if s.guidance_pixel_cap is not None and (not math.isfinite(s.guidance_pixel_cap) or s.guidance_pixel_cap <= 0):
            raise ValueError('guidance_pixel_cap must be null or finite and positive')
        if (s.guidance_rms_cap is not None or s.guidance_pixel_cap is not None) and s.preconditioner != 'none':
            raise ValueError('Scalar guidance clipping does not support the spectral metric')
        if s.delta_mode not in {"lambda", "quad", "abs", "ad"}:
            raise ValueError("delta_mode must be lambda, quad, abs or ad")
        if s.likelihood_mode not in {"direct", "inverse_lambda", "step"}:
            raise ValueError("likelihood_mode must be direct, inverse_lambda or step")
        if s.temperature_mode not in {"constant", "matched"}:
            raise ValueError("temperature_mode must be constant or matched")
        if s.preconditioner not in {'none', 'spectral'}:
            raise ValueError('preconditioner must be none or spectral')
        if s.curvature_probes < 1 or s.curvature_smoothing < 1 or s.curvature_smoothing % 2 != 1:
            raise ValueError('Invalid curvature probe count or smoothing width')
        if s.preconditioner == 'spectral' and (s.likelihood_mode == 'step' or s.data_direction != 'gradient'):
            raise ValueError('The fixed spectral metric requires direct/inverse-lambda likelihood and the raw gradient')
        if s.data_direction not in {"gradient", "adam", "lbfgs"}:
            raise ValueError("data_direction must be gradient, adam or lbfgs")
        if s.history_size < 1 or not math.isfinite(s.initial_inverse_curvature) or s.initial_inverse_curvature <= 0:
            raise ValueError("Invalid L-BFGS history or initial inverse curvature")
        if s.data_location not in {"state", "prox", "denoised"} or (self.method == "joint" and s.data_location != "state"):
            raise ValueError("data_location=prox is an optional Partial splitting only")
        if len(s.adam_betas) != 2 or any(not 0 <= b < 1 for b in s.adam_betas):
            raise ValueError("adam_betas must contain two values in [0,1)")
        if not math.isfinite(s.adam_epsilon) or s.adam_epsilon <= 0:
            raise ValueError("adam_epsilon must be finite and positive")
        if s.init not in {"backproj", "zero", "gauss", "y_plus"} or s.readout not in {"prox", "average", "state"}:
            raise ValueError("Unknown initialization or readout")
        if any(not math.isfinite(v) or v < 0 for v in (s.init_gain,s.init_noise,s.init_filter_sigma)):
            raise ValueError('Initialization scales must be finite and nonnegative')
        if not math.isfinite(s.init_filter_gain) or (s.init_filter_gain and not s.init_filter_sigma):
            raise ValueError('Initialization filtering requires finite gain and positive sigma')
        if s.clamp is not None and (
            len(s.clamp) != 2
            or not all(math.isfinite(x) for x in s.clamp)
            or s.clamp[0] >= s.clamp[1]
        ):
            raise ValueError("clamp must be null or [lower, upper]")
        if type(s.average_window) is not int or s.average_window < 1:
            raise ValueError("average_window must be a positive integer")
        if not 0 <= o.missing_fraction < 1 or not 0 <= c.amplitude_keep <= 1:
            raise ValueError("Invalid mask or amplitude fraction")
        if not math.isfinite(c.phase_coupling) or not 0 <= c.phase_coupling <= 1:
            raise ValueError('phase_coupling must be in [0,1]')
        if c.phase_coupling and (self.task != 'pr' or not (c.amplitude or c.final_amplitude)):
            raise ValueError('Phase coupling requires PR amplitude consistency')
        if type(o.kernel_size) is not int or not 1 <= o.kernel_size < 256 or o.kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd and in [1, 255]")
        if type(o.scale_factor) is not int or o.scale_factor < 1 or 256 % o.scale_factor:
            raise ValueError("scale_factor must divide 256")
        if self.model.precision not in {"fp32", "fp16"} or o.fft_device not in {"auto", "cpu"}:
            raise ValueError("Unknown precision or FFT device")
        if (c.amplitude or c.final_amplitude) and self.task != "pr":
            raise ValueError("Amplitude projection is only available for PR")
        if self.task == "inpaint" and c.paste_observed and o.noise_std != 0:
            raise ValueError("Pasting observations requires noiseless inpainting")
        for k in ("polish_steps", "adam_steps", "final_adam_steps", "amplitude_from", "adam_from"):
            if type(getattr(c, k)) is not int or getattr(c, k) < 0:
                raise ValueError(f"{k} must be a nonnegative integer")
        for k in ("polish_lr", "adam_lr", "final_adam_lr"):
            if not math.isfinite(getattr(c, k)) or getattr(c, k) <= 0:
                raise ValueError(f"{k} must be finite and positive")
        for k in ("seed", "observation_seed", "seed_step", "observation_seed_step"):
            if type(getattr(self, k)) is not int or getattr(self, k) < 0:
                raise ValueError(f"{k} must be a nonnegative integer")
        return self


def _construct(cls, data):
    if not isinstance(data, dict):
        raise ValueError(f"{cls.__name__} must be a mapping")
    unknown = set(data) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**data)


def from_dict(data):
    data = deepcopy(data)
    # Resolved existing records used one stride for both independent streams.
    data.setdefault('observation_seed_step', data.get('seed_step', ExperimentConfig.seed_step))
    for key, cls in [("model", ModelConfig), ("operator", OperatorConfig),
                     ("sampler", SamplerConfig), ("consistency", ConsistencyConfig),
                     ("refinement", RefinementConfig)]:
        data[key] = _construct(cls, data.get(key, {}))
    return _construct(ExperimentConfig, data).validate()


def load_config(path=None, *, task=None, method=None, scheme=None):
    if path is None:
        if task not in TASKS or method not in METHODS:
            raise ValueError("Supply --config or both --task and --method")
        path = ROOT / "configs" / method / f"{task}.yaml"
    with Path(path).open(encoding="utf-8") as stream:
        data = yaml.load(stream, Loader=ConfigLoader)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    defaults_path = ROOT / "configs/defaults.yaml"
    if defaults_path.is_file() and "timesteps" not in data.get("sampler", {}):
        with defaults_path.open(encoding="utf-8") as stream:
            data = deep_merge(yaml.load(stream, Loader=ConfigLoader), data)
    variants = data.pop("schemes", {})
    if not isinstance(variants, dict) or any(not isinstance(v, dict) for v in variants.values()):
        raise ValueError("schemes and each scheme override must be mappings")
    selected = scheme or data.get("sampler", {}).get("scheme", "em")
    if selected not in SCHEMES or set(variants) - set(SCHEMES):
        raise ValueError("Unknown scheme override")
    variant = deepcopy(variants.get(selected, {}))
    for key in ('seed', 'seed_step'):
        if key in variant:
            data[key] = variant.pop(key)
    if 'consistency' in variant:
        data['consistency'] = deep_merge(data.get('consistency', {}), variant.pop('consistency'))
    if 'refinement' in variant:
        data['refinement'] = deep_merge(data.get('refinement', {}), variant.pop('refinement'))
    data["sampler"] = deep_merge(data.get("sampler", {}), variant)
    curve = variant.get('likelihood')
    if isinstance(curve, dict) and {'start', 'end'} <= set(curve):
        # A complete curve replaces optional peak/ramp fields from the base.
        data['sampler']['likelihood'] = deepcopy(curve)
    data["sampler"]["scheme"] = selected
    if "schedule" in data["sampler"]:
        from .schedule import geometric_values, regular_schedule
        sampler = data["sampler"]
        final = int(sampler.get("final_prox", True) and sampler.get("readout", "prox") == "prox")
        sampler["timesteps"], sampler["steps"] = regular_schedule(sampler.pop("schedule"), final)
        sampler["gamma"] = geometric_values(sampler.pop("likelihood"), len(sampler["steps"]))
    return from_dict(data)


def deep_merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        result[key] = deep_merge(result[key], value) if (
            isinstance(value, dict) and isinstance(result.get(key), dict)
        ) else deepcopy(value)
    return result


def project_path(path):
    p = Path(path).expanduser()
    return p if p.is_absolute() else ROOT / p
