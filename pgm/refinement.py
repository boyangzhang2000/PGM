"""Image-only optimization after sampling; no proximal network or truth input."""
from dataclasses import dataclass,field
import torch


def total_variation(x,epsilon=0):
    """Anisotropic TV, averaged separately over horizontal and vertical edges."""
    if epsilon==0:
        return ((x[...,1:,:]-x[...,:-1,:]).abs().mean()
                + (x[...,:,1:]-x[...,:,:-1]).abs().mean())
    horizontal=x[...,1:,:]-x[...,:-1,:]
    vertical=x[...,:,1:]-x[...,:,:-1]
    return ((horizontal.square()+epsilon**2).sqrt().mean()
            +(vertical.square()+epsilon**2).sqrt().mean()-2*epsilon)


@dataclass
class RefinementResult:
    estimate: torch.Tensor
    trace: list[dict]
    snapshots: dict
    evaluations: dict
    checkpoint_evaluations: dict = field(default_factory=dict)


def refine_image(op, initial, y, config, perceptual=None, checkpoints=()):
    """Optimize the final image with fixed-anchor regularization and saved prefixes."""
    if config.steps and config.lpips and perceptual is None:
        raise ValueError('LPIPS regularization requires its frozen perceptual model')
    if any(type(k) is not int or not 0 <= k <= config.steps for k in checkpoints):
        raise ValueError('Invalid refinement checkpoint')
    if perceptual is not None:
        perceptual.eval().requires_grad_(False)
    # NPY reloads are contiguous; match their reduction order for live samples.
    reference = initial.detach().contiguous().clamp(-1,1)
    target = y.detach()
    snapshots = {0:reference.clone()} if 0 in checkpoints else {}
    counts = dict(adam_steps=0,data_evaluations=0,lpips_evaluations=0,prior=0)
    if config.optimizer=='lbfgs':
        counts['lbfgs_steps']=0
    checkpoint_counts={0:dict(counts)} if 0 in checkpoints else {}
    if config.steps == 0:
        return RefinementResult(reference,[],snapshots,counts,checkpoint_counts)
    with torch.enable_grad():
        x = reference.clone().requires_grad_(True)
        optimizer = (torch.optim.Adam([x],lr=config.lr) if config.optimizer=='adam' else
                     torch.optim.LBFGS([x],lr=config.lr,max_iter=1,max_eval=config.max_evaluations,
                         history_size=config.history_size,tolerance_grad=0,tolerance_change=0,
                         line_search_fn='strong_wolfe'))
        def objective():
            if counts['data_evaluations']>=config.max_evaluations:
                raise RuntimeError('Refinement objective evaluation budget exhausted')
            data = (op.forward(x)-target).square().mean()/2
            counts['data_evaluations'] += 1
            lp = perceptual(x,reference).mean() if config.lpips else x.new_zeros(())
            counts['lpips_evaluations'] += int(bool(config.lpips))
            tv = total_variation(x,config.tv_epsilon) if config.tv else x.new_zeros(())
            loss = config.data*data + config.lpips*lp + config.tv*tv
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite refinement objective')
            terms = dict(loss=float(loss.detach()),data=float(data.detach()),
                         lpips=float(lp.detach()),tv=float(tv.detach()))
            return loss,terms
        trace = []
        def closure():
            optimizer.zero_grad(set_to_none=True)
            loss,_=objective()
            loss.backward()
            if x.grad is None or not torch.isfinite(x.grad).all():
                raise FloatingPointError('Nonfinite refinement gradient')
            return loss
        for iteration in range(config.steps):
            optimizer.zero_grad(set_to_none=True)
            if config.optimizer=='adam':
                loss, terms = objective()
            else:
                with torch.no_grad():
                    _, terms = objective()
            trace.append(dict(step=iteration,**terms))
            if config.optimizer=='adam':
                loss.backward()
                if x.grad is None or not torch.isfinite(x.grad).all():
                    raise FloatingPointError('Nonfinite refinement gradient')
                optimizer.step()
                counts['adam_steps'] += 1
            else:
                optimizer.step(closure)
                counts['lbfgs_steps'] += 1
            with torch.no_grad():
                x.clamp_(-1,1)
            if iteration+1 in checkpoints:
                snapshots[iteration+1] = x.detach().clone()
                # A standalone prefix evaluates its terminal objective once.
                # The full trajectory reuses the next preview/final evaluation.
                checkpoint_counts[iteration+1]=dict(counts,
                    data_evaluations=counts['data_evaluations']+1,
                    lpips_evaluations=counts['lpips_evaluations']+int(bool(config.lpips)))
        with torch.no_grad():
            _, terms = objective()
        trace.append(dict(step=config.steps,**terms))
    return RefinementResult(x.detach(),trace,snapshots,counts,checkpoint_counts)
