"""Diffusion time conversion and configurable annealing schedules."""
import numpy as np
import torch


def regular_schedule(rule, final_evaluations=1):
    """Generate a monotone schedule and exactly allocate the evaluation budget."""
    allowed = {"budget", "stages", "start", "end", "spacing", "allocation", "power", "warmup_end", "warmup_fraction", "center", "width"}
    if set(rule) - allowed:
        raise ValueError(f"Unknown schedule fields: {sorted(set(rule) - allowed)}")
    if type(rule.get("budget")) is not int or type(final_evaluations) is not int or final_evaluations < 0:
        raise ValueError("budget and final_evaluations must be integer counts")
    budget = rule["budget"] - final_evaluations
    stages = rule.get("stages", budget)
    if type(budget) is not int or type(stages) is not int or not 1 <= stages <= budget:
        raise ValueError("The schedule must have a positive budget and no more stages than updates")
    start, end = rule["start"], rule["end"]
    if any(type(v) not in (int, float) or not np.isfinite(v) for v in (start, end)) or not 1 <= end <= start < 1000:
        raise ValueError("Require 1 <= schedule end <= start < 1000")
    spacing = rule.get("spacing", "geometric")
    if spacing == "geometric":
        timesteps = np.geomspace(start, end, stages)
    elif spacing == "linear":
        timesteps = np.linspace(start, end, stages)
    else:
        raise ValueError("spacing must be geometric or linear")
    if 'warmup_end' in rule or 'warmup_fraction' in rule:
        knee, fraction = rule.get('warmup_end'), rule.get('warmup_fraction')
        if (type(knee) not in (int, float) or not end <= knee <= start or
                type(fraction) not in (int, float) or not 0 < fraction < 1):
            raise ValueError('Warmup needs an intermediate time and a fraction in (0,1)')
        values = np.array([start, knee, end], dtype=float)
        if spacing == 'geometric':
            values = np.log(values)
        timesteps = np.interp(np.linspace(0, 1, stages), [0, fraction, 1], values)
        if spacing == 'geometric':
            timesteps = np.exp(timesteps)
    allocation = rule.get("allocation", "uniform")
    if allocation == "uniform":
        weights = np.ones(stages)
    elif allocation == 'focus':
        center, width = rule.get('center'), rule.get('width')
        if (type(center) not in (int, float) or not 0 <= center <= 1 or
                type(width) not in (int, float) or not np.finfo(float).eps <= width <= 1):
            raise ValueError('Focused allocation needs center in [0,1] and a positive resolvable width <=1')
        position = np.linspace(0, 1, stages)
        log_weights = -((position-center)/width)**2 / 2
        weights = np.exp(log_weights-log_weights.max())
    elif allocation in {"head", "tail", "middle", "ends"}:
        power = rule.get("power", 1)
        if type(power) not in (int, float) or not np.isfinite(power) or power <= 0:
            raise ValueError("allocation power must be finite and positive")
        position = np.linspace(1 / stages, 1, stages)
        shapes = {"tail": position, "head": position[::-1],
                  "middle": np.minimum(position, position[::-1]),
                  "ends": np.maximum(position, position[::-1])}
        shape = shapes[allocation]
        weights = (shape / shape.max()) ** power
    else:
        raise ValueError("allocation must be uniform, head, tail, middle, ends or focus")
    quotas = (budget - stages) * weights / weights.sum()
    steps = 1 + np.floor(quotas).astype(int)
    remainder = budget - steps.sum()
    order = np.argsort(-(quotas - np.floor(quotas)), kind="stable")
    steps[order[:remainder]] += 1
    return np.rint(timesteps).astype(int).tolist(), steps.tolist()


def geometric_values(rule, count):
    """Constant or compact likelihood schedule, geometric by default.
    
    An optional peak creates a rise-and-fall schedule. Linear spacing also
    permits zero endpoints. The default path retains stored rounding exactly.
    """
    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive integer")
    if type(rule) in (int, float):
        if not np.isfinite(rule) or rule < 0:
            raise ValueError("Likelihood must be finite and nonnegative")
        return [float(rule)] * count
    if not isinstance(rule, dict):
        raise ValueError("Likelihood must be a number or mapping")
    allowed = {"start", "end", "ramp", "spacing", "power", "peak", "peak_fraction"}
    if set(rule) - allowed or not {'start', 'end'} <= set(rule):
        raise ValueError("Invalid likelihood schedule fields")
    spacing = rule.get('spacing', 'geometric')
    if spacing not in {'geometric', 'linear'}:
        raise ValueError('Likelihood spacing must be geometric or linear')
    names = ['start', 'end'] + (['peak'] if 'peak' in rule else [])
    if any(type(rule[k]) not in (int, float) or not np.isfinite(rule[k]) or rule[k] < 0 for k in names):
        raise ValueError('Likelihood values must be finite and nonnegative')
    if spacing == 'geometric' and any(rule[k] == 0 for k in names):
        raise ValueError('Geometric likelihood values must be positive')
    ramp = rule.get("ramp", 1.0)
    if type(ramp) not in (int, float) or not 0 < ramp <= 1:
        raise ValueError("ramp must be in (0,1]")
    power = rule.get('power', 1)
    if type(power) not in (int, float) or not np.isfinite(power) or power <= 0:
        raise ValueError('Likelihood power must be finite and positive')
    progress = np.minimum(np.linspace(0, 1, count) / ramp, 1) ** power
    if 'peak' in rule or 'peak_fraction' in rule:
        fraction = rule.get('peak_fraction')
        if 'peak' not in rule or type(fraction) not in (int, float) or not 0 < fraction < 1:
            raise ValueError('Likelihood peak needs a peak_fraction in (0,1)')
        values = np.array([rule['start'], rule['peak'], rule['end']], dtype=float)
        if spacing == 'geometric':
            values = np.log(values)
        values = np.interp(progress, [0, fraction, 1], values)
        return (np.exp(values) if spacing == 'geometric' else values).tolist()
    if spacing == 'linear':
        return (rule['start'] + (rule['end'] - rule['start']) * progress).tolist()
    return (rule["start"] * (rule["end"] / rule["start"]) ** progress).tolist()


def diffusion_schedule(alphas_cumprod, device):
    """Build device-local signal scalings and host-side image-space lambdas."""
    ac = np.asarray(alphas_cumprod, dtype=np.float64)
    if ac.ndim != 1 or not np.isfinite(ac).all() or not ((ac > 0) & (ac < 1)).all():
        raise ValueError("alphas_cumprod must be a finite vector strictly between 0 and 1")
    mu = torch.tensor(np.sqrt(ac), dtype=torch.float32, device=device)
    sigma = torch.tensor(np.sqrt(1 - ac), dtype=torch.float32, device=device)
    return mu, sigma, (1 - ac) / ac
