"""
Sampling Algorithms for Inverse Problems
Implements specialized sampling procedures for inverse problems with gradient guidance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Optional, Tuple, List, Dict, Callable, Any
import math
from tqdm import tqdm
from .training import VPSDE, VESDE, MYSDE
import lpips
import warnings
warnings.filterwarnings('ignore')

class InverseProblemSampler:
    """Sampler for inverse problems using trained proximal networks"""
    
    def __init__(self, model: nn.Module, 
                 inverse_problem, 
                 sde_type: str = 've', 
                 device: str = 'cuda', 
                 dc_optimizer_kwargs: Optional[Dict] = None):

        self.model = model
        self.inverse_problem = inverse_problem
        self.sde_type = sde_type.lower()
        self.device = device
        
        if self.sde_type == 've': self.sde = VESDE()
        elif self.sde_type == 'vp': self.sde = VPSDE()
        elif self.sde_type == 'my': self.sde = MYSDE()
        else: raise ValueError(f"Unknown SDE type: {sde_type}")
        
        self.model.to(self.device)
        self.model.eval()

        dc_optimizer_kwargs = dc_optimizer_kwargs or {}
        self.dc_optimizer = DataConsistencyOptimizer(inverse_problem=inverse_problem, device=device, **dc_optimizer_kwargs)
    
    def sample_with_data_guidance(self, measurements: torch.Tensor, 
                                  num_samples: int = 1, 
                                  num_steps: int = 100, 
                                  data_weight: float = 10.0, 
                                  use_ode: bool = False, 
                                  method: str = 'euler', 
                                  return_trajectory: bool = False, 
                                  guidance_schedule: Optional[Callable] = None) -> Tuple[torch.Tensor, Optional[List]]:

        img_shape = measurements.shape[1:]
        if self.inverse_problem.problem_type == 'super_resolution':
            scale_factor = self.inverse_problem.scale_factor
            img_shape = (img_shape[0], img_shape[1]*scale_factor, img_shape[2]*scale_factor)
            
        if self.sde_type == 've': x = torch.randn(num_samples, *img_shape, device=self.device) * self.sde.sigma_max
        elif self.sde_type == 'vp': x = torch.randn(num_samples, *img_shape, device=self.device)
        elif self.sde_type == 'my':
            t_start = torch.tensor(1.0, device=self.device)
            x = torch.randn(num_samples, *img_shape, device=self.device) * self.sde.sigma(t_start)
        
        if measurements.shape[0] == 1 and num_samples > 1:
            measurements = measurements.repeat(num_samples, *([1] * (measurements.dim() - 1)))
        
        dt = 1.0 / num_steps
        trajectory = [x.detach().cpu()] if return_trajectory else None
        
        with torch.no_grad():
            for step in tqdm(range(num_steps), desc="Sampling"):
                t_scale = torch.tensor(1.0 - step * dt, device=self.device)
                t_current = torch.full((num_samples, 1), t_scale, device=self.device)
                lambda_t = self.sde.lambda_val(t_current)
                
                data_grad = self.inverse_problem.gradient_data_consistency(x, measurements)
                current_data_weight = data_weight if guidance_schedule is None else guidance_schedule(step / num_steps) * data_weight
                x_guided = x - lambda_t.view(-1, 1, 1, 1) * current_data_weight * data_grad
                
                pred_prox = self.model(x_guided, lambda_t)
                score = (pred_prox - x) / lambda_t.view(-1, 1, 1, 1)
                
                if self.sde_type == 've':
                    sigma_t = self.sde.sigma(t_current)
                    sigma_deriv = self.sde.sigma_derivative(t_current)
                    if use_ode: drift = -sigma_t * sigma_deriv * score
                    else:
                        drift = -2 * sigma_t * sigma_deriv * score
                        diffusion = torch.sqrt(2 * sigma_deriv * sigma_t)
                elif self.sde_type == 'vp':
                    beta_t = self.sde.beta(t_current)
                    if use_ode: drift = -0.5 * beta_t * x - beta_t * score
                    else:
                        drift = -beta_t * x - 2 * beta_t * score
                        diffusion = torch.sqrt(2 * beta_t)
                elif self.sde_type == 'my':
                    if use_ode:
                        c1, c2 = self.sde.reverse_coeff_ode(x, pred_prox, t_current, dt)
                        c3 = torch.zeros_like(c1)
                    else:
                        c1, c2, c3 = self.sde.reverse_coeff_sde(x, pred_prox, t_current, dt)
                    x = c1.view(-1, 1, 1, 1) * x + c2.view(-1, 1, 1, 1) * pred_prox + c3.view(-1, 1, 1, 1) * torch.randn_like(x)
                    if return_trajectory and step % max(1, num_steps // 10) == 0: trajectory.append(x.detach().cpu())
                    continue
                
                if self.sde_type in ['ve', 'vp']:
                    if use_ode: x = x + drift * dt
                    else: x = x + drift * dt + diffusion * torch.randn_like(x) * math.sqrt(dt)
                
                if return_trajectory and step % max(1, num_steps // 10) == 0: trajectory.append(x.detach().cpu())
            
        return x.detach().cpu(), (trajectory if return_trajectory else None)

    def sample_with_data_guidance_and_optimization(self, measurements: torch.Tensor, 
                                                   num_samples: int = 1, 
                                                   num_steps: int = 100, 
                                                   data_weight: float = 10.0, 
                                                   use_ode: bool = True, 
                                                   method: str = 'euler', 
                                                   return_trajectory: bool = False, 
                                                   guidance_schedule: Optional[Callable] = None, 
                                                   apply_optimization: bool = True, 
                                                   optimization_method: str = 'gradient_descent', 
                                                   optimization_kwargs: Optional[Dict] = None) -> Tuple[torch.Tensor, Optional[List], Dict]:

        initial_samples, trajectory = self.sample_with_data_guidance(measurements, 
                                                                     num_samples, 
                                                                     num_steps, 
                                                                     data_weight, 
                                                                     use_ode, 
                                                                     method, 
                                                                     return_trajectory, 
                                                                     guidance_schedule)

        metrics = {'initial_psnr': None, 
                   'optimized_psnr': None, 
                   'initial_ssim': None, 
                   'optimized_ssim': None, 
                   'initial_lpips': None, 
                   'optimized_lpips': None, 
                   'data_consistency_before': None, 
                   'data_consistency_after': None, 
                   'optimization_metrics': None}
        
        if not apply_optimization: return initial_samples, trajectory, metrics
        
        optimization_kwargs = optimization_kwargs or {}
        optimized_samples, optimization_metrics_list = [], []
        
        for i in range(len(initial_samples)):
            sample = initial_samples[i:i+1]
            measurement = measurements[i:i+1] if measurements.shape[0] > 1 else measurements
            optimized_sample, opt_metrics = self.dc_optimizer.optimize_sample(sample, 
                                                                              measurement, 
                                                                              method=optimization_method, 
                                                                              **optimization_kwargs)
            optimized_sample = optimized_sample + self.sde.sigma(torch.tensor(0)) * torch.randn(1, *optimized_sample.shape[1:], device=self.device)
            optimized_samples.append(optimized_sample)
            optimization_metrics_list.append(opt_metrics)
        
        optimized_samples = torch.cat(optimized_samples, dim=0)
        metrics['optimization_metrics'] = optimization_metrics_list
        return optimized_samples, trajectory, metrics

class DataConsistencyOptimizer:
    def __init__(self, inverse_problem, device: str = 'cuda', verbose: bool = False):
        self.inverse_problem = inverse_problem
        self.device = torch.device(device)
        self.verbose = verbose
    
    def tv_regularizer(self, x: torch.Tensor, weight: float = 0.01) -> torch.Tensor:
        if x.dim() == 3: x = x.unsqueeze(0)
        dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        return weight * (torch.sum(dx) + torch.sum(dy)) / x.numel()
    
    def lpips_regularizer(self, x: torch.Tensor, reference: torch.Tensor, lpips_model: nn.Module, weight: float = 0.1) -> torch.Tensor:
        if x.shape != reference.shape:
            reference = F.interpolate(reference, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return weight * lpips_model(x, reference).mean()
    
    def gradient_descent_optimization(self, initial_sample: torch.Tensor, 
                                      measurement: torch.Tensor, 
                                      num_iterations: int = 100, 
                                      learning_rate: float = 0.1, 
                                      data_weight: float = 1.0, 
                                      tv_weight: float = 0.001, 
                                      lpips_weight: float = 0.0, 
                                      lpips_model: Optional[nn.Module] = None, 
                                      clip_range: Optional[Tuple[float, float]] = (-1.0, 1.0)) -> Tuple[torch.Tensor, Dict]:

        x = initial_sample.clone().detach().to(self.device).requires_grad_(True)
        measurement = measurement.clone().detach().to(self.device)
        optimizer = optim.Adam([x], lr=learning_rate)
        metrics = {'data_loss': [], 'tv_loss': [], 'lpips_loss': [], 'total_loss': [], 'psnr': []}
        reference = initial_sample.clone().detach().to(self.device)
        
        progress_bar = tqdm(range(num_iterations), desc='Data Consistency Optimization') if self.verbose else range(num_iterations)
        for iteration in progress_bar:
            optimizer.zero_grad()
            predicted_measurement = self.inverse_problem.forward(x)
            data_loss = F.mse_loss(predicted_measurement, measurement)
            total_loss = data_loss
            
            if tv_weight > 0: total_loss = total_loss + self.tv_regularizer(x, tv_weight)
            if lpips_weight > 0:
                if lpips_model is None: lpips_model = lpips.LPIPS(net='alex').to(self.device)
                lpips_loss = self.lpips_regularizer(x, reference, lpips_model, lpips_weight)
                total_loss = total_loss + lpips_loss
                
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_([x], max_norm=1.0)
            optimizer.step()
            if clip_range is not None: x.data = torch.clamp(x.data, clip_range[0], clip_range[1])
            
            with torch.no_grad():
                metrics['data_loss'].append(data_loss.item())
                metrics['total_loss'].append(total_loss.item())
                psnr = 20 * torch.log10(1.0 / torch.sqrt(F.mse_loss(x, reference)))
                metrics['psnr'].append(psnr.item())
                if self.verbose: progress_bar.set_postfix({'loss': total_loss.item(), 'psnr': metrics['psnr'][-1]})
        
        return x.detach(), metrics

    def optimize_sample(self, sample: torch.Tensor, 
                        measurement: torch.Tensor, 
                        method: str = 'gradient_descent', **kwargs) -> Tuple[torch.Tensor, Dict]:
        if method == 'gradient_descent': return self.gradient_descent_optimization(sample, measurement, **kwargs)
        else: raise ValueError(f"Unknown optimization method: {method}")