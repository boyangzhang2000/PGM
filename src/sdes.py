"""
Shared SDE formulations (VE-SDE, VP-SDE, Custom MY-SDE)
Consolidates SDE definitions used in both Training and Sampling phases.
"""

import torch
import math
from typing import Tuple

class VESDE:
    """Enhanced Variance Exploding SDE"""
    def __init__(self, sigma_min: float = 0.01, sigma_max: float = 50.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.log_ratio = math.log(sigma_max / sigma_min)
    
    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    
    def sigma_derivative(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma(t) * self.log_ratio
    
    def mean_weight(self, t: torch.Tensor) -> torch.Tensor:
        return torch.tensor(1.0, device=t.device)
    
    def variance(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma(t)**2

    def lambda_val(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma(t) ** 2
    
    def forward_process(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sigma_t = self.sigma(t)
        z = torch.randn_like(x0)
        return x0 + sigma_t * z, z
    
    def reverse_drift_ode(self, x: torch.Tensor, score: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return -self.sigma(t) * self.sigma_derivative(t) * score
    
    def reverse_drift_sde(self, x: torch.Tensor, score: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return -2 * self.sigma(t) * self.sigma_derivative(t) * score
    
    def reverse_diffusion(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(2 * self.sigma_derivative(t) * self.sigma(t))

class VPSDE:
    """Enhanced Variance Preserving SDE"""
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0):
        self.beta_min = beta_min
        self.beta_max = beta_max
    
    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_min + t * (self.beta_max - self.beta_min)
    
    def beta_integral(self, t: torch.Tensor) -> torch.Tensor:
        return t * self.beta_min + 0.5 * t ** 2 * (self.beta_max - self.beta_min)
    
    def mean_weight(self, t: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.beta_integral(t))
    
    def variance(self, t: torch.Tensor) -> torch.Tensor:
        return 1 - torch.exp(-2 * self.beta_integral(t))

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.variance(t))

    def lambda_val(self, t: torch.Tensor) -> torch.Tensor:
        return self.variance(t) / self.mean_weight(t)**2
    
    def forward_process(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu_t = self.mean_weight(t)
        sigma_t = self.sigma(t)
        z = torch.randn_like(x0)
        return mu_t * x0 + sigma_t * z, z
    
    def reverse_drift_ode(self, x: torch.Tensor, score: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        beta_t = self.beta(t)
        return -beta_t * x - beta_t * score
    
    def reverse_drift_sde(self, x: torch.Tensor, score: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        beta_t = self.beta(t)
        return -beta_t * x - 2 * beta_t * score
    
    def reverse_diffusion(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(2 * self.beta(t))

class MYSDE:
    """Custom MY-SDE Formulation"""
    def __init__(self, alpha: float = 10.0, beta: float = 8.0):
        self.alpha = alpha
        self.beta = beta
    
    def mean_weight(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t)

    def lambda_val(self, t: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.alpha * t - self.beta)

    def variance(self, t: torch.Tensor) -> torch.Tensor:
        return self.lambda_val(t) * self.mean_weight(t)**2

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.variance(t))

    def forward_process(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu_t = self.mean_weight(t)
        sigma_t = self.sigma(t)
        z = torch.randn_like(x0)
        return mu_t * x0 + sigma_t * z, z
    
    def reverse_coeff_ode(self, x: torch.Tensor, score: torch.Tensor, t: torch.Tensor, dt: float) -> Tuple[torch.Tensor, torch.Tensor]:
        t_prev = t - dt
        mu_t = self.mean_weight(t)
        mu_t_prev = self.mean_weight(t_prev)
        lambd_t = self.lambda_val(t)
        lambd_t_prev = self.lambda_val(t_prev)
        coeff1 = torch.sqrt(lambd_t_prev * mu_t_prev / lambd_t / mu_t)
        coeff2 = 1 - coeff1
        return coeff1, coeff2
    
    def reverse_coeff_sde(self, x: torch.Tensor, prox: torch.Tensor, t: torch.Tensor, dt: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t_prev = t - dt
        mu_t = self.mean_weight(t)
        mu_t_prev = self.mean_weight(t_prev)
        lambd_t = self.lambda_val(t)
        lambd_t_prev = self.lambda_val(t_prev)
        coeff1 = lambd_t_prev * mu_t_prev / (lambd_t * mu_t)
        coeff2 = mu_t_prev * (1 - lambd_t_prev / lambd_t)
        coeff3 = mu_t_prev * torch.sqrt(lambd_t_prev * (1 - lambd_t_prev / lambd_t))
        return coeff1, coeff2, coeff3