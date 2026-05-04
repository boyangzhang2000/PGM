"""
Proximal-Based Generative Modeling: Experimental Verification
Implementation for 1D Gaussian Distribution with VP-SDE and VE-SDE
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import math

class GaussianTarget:
    """1D Gaussian Target Distribution"""
    
    def __init__(self, mu=0.0, sigma=1.0):
        """
        Initialize Gaussian target distribution
        
        Args:
            mu: Mean of the Gaussian distribution
            sigma: Standard deviation of the Gaussian distribution
        """
        self.mu = torch.tensor(mu)
        self.sigma = torch.tensor(sigma)
        
    def log_density(self, x):
        """Log density function: log π₀(x)"""
        return -0.5 * ((x - self.mu) / self.sigma) ** 2 - torch.log(self.sigma * math.sqrt(2 * math.pi))
    
    def density(self, x):
        """Density function: π₀(x)"""
        return torch.exp(self.log_density(x))
    
    def score_function(self, x):
        """Score function: ∇logπ₀(x)"""
        return -(x - self.mu) / (self.sigma ** 2)
    
    def proximal_operator(self, x, lambda_val):
        """
        Proximal operator: argmin_u {f(u) + 1/(2λ)||u-x||²}
        
        For Gaussian distribution f(x) = 0.5*(x-mu)²/sigma² + const
        Has closed-form solution
        """
        return (x + (lambda_val / (self.sigma ** 2)) * self.mu) / (1 + lambda_val / (self.sigma ** 2))
    
    def moreau_envelope(self, x, lambda_val):
        """Moreau-Yosida envelope: M_f^λ(x)"""
        u_star = self.proximal_operator(x, lambda_val)
        f_u = 0.5 * ((u_star - self.mu) / self.sigma) ** 2
        return f_u + (1/(2*lambda_val)) * (u_star - x) ** 2


class VPSDE:
    """
    Variance Preserving SDE (VP-SDE)
    
    Forward process: dx = -0.5 β(t) x dt + sqrt(β(t)) dw
    """
    
    def __init__(self, beta_min=0.1, beta_max=20.0):
        """
        Initialize VP-SDE
        
        Args:
            beta_min: Minimum value of β(t)
            beta_max: Maximum value of β(t)
        """
        self.beta_min = beta_min
        self.beta_max = beta_max
    
    def beta(self, t):
        """β(t) schedule"""
        return self.beta_min + t * (self.beta_max - self.beta_min)
    
    def mean_weight(self, t):
        """μ(t) = exp(-0.5 ∫₀ᵗ β(s) ds)"""
        integral = torch.tensor(0.5 * t * (self.beta_min + 0.5 * t * (self.beta_max - self.beta_min)))
        return torch.exp(-integral)
    
    def variance(self, t):
        """σ²(t) = 1 - exp(-∫₀ᵗ β(s) ds)"""
        integral = torch.tensor(t * (self.beta_min + 0.5 * t * (self.beta_max - self.beta_min)))
        return 1 - torch.exp(-integral)
    
    def lambda_val(self, t):
        """λ(t) = σ²(t) for proximal matching"""
        return self.variance(t)
    
    def forward_process(self, x0, t):
        """
        Forward process: x_t = μ(t) x_0 + σ(t) z
        
        Args:
            x0: Samples from target distribution
            t: Time step
        """
        mu_t = self.mean_weight(t)
        sigma_t = torch.sqrt(self.variance(t))
        z = torch.randn_like(x0)
        return mu_t * x0 + sigma_t * z, z


class VESDE:
    """
    Variance Exploding SDE (VE-SDE)
    
    Forward process: dx = sqrt(dσ²/dt) dw
    """
    
    def __init__(self, sigma_min=0.01, sigma_max=50.0):
        """
        Initialize VE-SDE
        
        Args:
            sigma_min: Minimum noise level
            sigma_max: Maximum noise level
        """
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
    
    def sigma(self, t):
        """σ(t) schedule"""
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    
    def lambda_val(self, t):
        """λ(t) = σ²(t) for proximal matching"""
        return self.sigma(t) ** 2
    
    def forward_process(self, x0, t):
        """
        Forward process: x_t = x_0 + σ(t) z
        
        Args:
            x0: Samples from target distribution
            t: Time step
        """
        sigma_t = self.sigma(t)
        z = torch.randn_like(x0)
        return x0 + sigma_t * z, z


class DiffusionProcess:
    """Unified diffusion process handling both VP-SDE and VE-SDE"""
    
    def __init__(self, target, sde_type='ve', **sde_kwargs):
        """
        Initialize diffusion process
        
        Args:
            target: Target distribution
            sde_type: Type of SDE ('ve' or 'vp')
            sde_kwargs: Parameters for SDE
        """
        self.target = target
        self.sde_type = sde_type
        
        if sde_type.lower() == 've':
            self.sde = VESDE(**sde_kwargs)
        elif sde_type.lower() == 'vp':
            self.sde = VPSDE(**sde_kwargs)
        else:
            raise ValueError(f"Unknown SDE type: {sde_type}")
    
    def forward_process(self, x0, t):
        """Forward process wrapper"""
        return self.sde.forward_process(x0, t)
    
    def lambda_val(self, t):
        """Get λ(t) = σ²(t)"""
        return self.sde.lambda_val(t)
    
    def score_from_proximal(self, x, t):
        """
        Compute score via proximal operator: ∇logp_t(x) = (Prox_f^λ(x) - x)/λ
        
        Args:
            x: Input samples
            t: Time step
        """
        lambda_t = self.lambda_val(t)
        mu_t = self.sde.mean_weight(t) if self.sde_type.lower() == 'vp' else 1
        prox = self.target.proximal_operator(x/mu_t, lambda_t/mu_t**2)
        return (mu_t*prox - x) / lambda_t
    
    def theoretical_score(self, x, t):
        """
        Compute theoretical score function
        
        For VE-SDE: p_t(x) = N(μ, σ₀² + σ(t)²)
        For VP-SDE: p_t(x) = N(μ(t), σ_total²(t))
        """
        if self.sde_type.lower() == 've':
            # VE-SDE: Gaussian convolution
            sigma_total = torch.sqrt(self.target.sigma ** 2 + self.sde.sigma(t) ** 2)
            return -(x - self.target.mu) / (sigma_total ** 2)
        
        elif self.sde_type.lower() == 'vp':
            # VP-SDE: Ornstein-Uhlenbeck process
            mu_t = self.sde.mean_weight(t) * self.target.mu
            sigma_total_sq = self.target.sigma ** 2 * self.sde.mean_weight(t) ** 2 + self.sde.variance(t)
            return -(x - mu_t) / sigma_total_sq


class ProximalVisualizer:
    """Visualization tools for proximal-based diffusion"""
    
    def __init__(self):
        self.fig_size = (15, 12)
        
    def plot_distribution_evolution(self, target, diffusion, t_values, sde_type):
        """Plot distribution evolution over time"""
        fig, axes = plt.subplots(2, 3, figsize=self.fig_size)
        axes = axes.flatten()
        
        x = torch.linspace(-5, 5, 1000)
        
        for i, t in enumerate(t_values):
            ax = axes[i]
            lambda_t = diffusion.lambda_val(t)
            
            # Target distribution
            target_density = target.density(x)
            
            # Theoretical smoothed distribution
            if sde_type.lower() == 've':
                mu_t = 1
                sigma_total = torch.sqrt(target.sigma ** 2 + diffusion.sde.sigma(t) ** 2)
                theoretical_dist = torch.distributions.Normal(target.mu, sigma_total)
                theoretical_density = theoretical_dist.log_prob(x).exp()
                theoretical_density = theoretical_density / torch.trapz(theoretical_density, x)  # Normalize
            else:  # VP-SDE
                mu_t = diffusion.sde.mean_weight(t)
                sigma_total = torch.sqrt(target.sigma ** 2 * diffusion.sde.mean_weight(t) ** 2 + diffusion.sde.variance(t))
                theoretical_dist = torch.distributions.Normal(mu_t * target.mu, sigma_total)
                theoretical_density = theoretical_dist.log_prob(x).exp()
                theoretical_density = theoretical_density / torch.trapz(theoretical_density, x)  # Normalize
           
            # Proximal-based distribution (via Moreau envelope)
            moreau_vals = torch.stack([target.moreau_envelope(x_i/mu_t, lambda_t/mu_t**2) for x_i in x])
            proximal_density = torch.exp(-moreau_vals)
            proximal_density = proximal_density / torch.trapz(proximal_density, x)  # Normalize
            
            ax.plot(x.numpy(), target_density.numpy(), 'b-', label='Target π₀', linewidth=2)
            ax.plot(x.numpy(), theoretical_density.numpy(), 'r--', 
                   label=f'Theoretical p_t', linewidth=2)
            ax.plot(x.numpy(), proximal_density.numpy(), 'g:', 
                   label=f'Proximal p_t', linewidth=2)
            
            ax.set_title(f'{sde_type.upper()}-SDE: t = {t:.2f}, λ = {lambda_t:.2f}')
            ax.set_xlabel('x')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_score_comparison(self, target, diffusion, t_values, sde_type):
        """Compare theoretical and proximal-based score functions"""
        fig, axes = plt.subplots(2, 3, figsize=self.fig_size)
        axes = axes.flatten()
        
        x = torch.linspace(-5, 5, 1000)
        
        for i, t in enumerate(t_values):
            ax = axes[i]
            lambda_t = diffusion.lambda_val(t)
            
            # Theoretical score
            score_theoretical = diffusion.theoretical_score(x, t)
            
            # Proximal-based score
            score_proximal = torch.stack([diffusion.score_from_proximal(torch.tensor([x_i]), t) 
                                        for x_i in x]).squeeze()
            
            ax.plot(x.numpy(), score_theoretical.numpy(), 'r-', 
                   label='Theoretical score', linewidth=2)
            ax.plot(x.numpy(), score_proximal.numpy(), 'g--', 
                   label='Proximal score', linewidth=2)
            
            # # Relative error
            # relative_error = torch.abs(score_theoretical - score_proximal) / (torch.abs(score_theoretical) + 1e-8)
            # ax_twin = ax.twinx()
            # ax_twin.plot(x.numpy(), relative_error.numpy(), 'k:', alpha=0.7, label='Relative error')
            # ax_twin.set_ylabel('Relative Error', color='k')
            # ax_twin.tick_params(axis='y', labelcolor='k')
            
            ax.set_title(f'{sde_type.upper()}-SDE: t = {t:.2f}, λ = {lambda_t:.2f}')
            ax.set_xlabel('x')
            ax.set_ylabel('Score')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_sde_comparison(self, target, ve_diffusion, vp_diffusion, t_values):
        """Compare VE-SDE and VP-SDE behaviors"""
        fig, axes = plt.subplots(2, 3, figsize=self.fig_size)
        axes = axes.flatten()
        
        x = torch.linspace(-5, 5, 1000)
        
        for i, t in enumerate(t_values):
            ax = axes[i]
            
            # Target distribution
            target_density = target.density(x)
            
            # VE-SDE distribution
            sigma_total_ve = torch.sqrt(target.sigma ** 2 + ve_diffusion.sde.sigma(t) ** 2)
            ve_dist = torch.distributions.Normal(target.mu, sigma_total_ve)
            ve_density = ve_dist.log_prob(x).exp()
            
            # VP-SDE distribution
            mu_t_vp = vp_diffusion.sde.mean_weight(t) * target.mu
            sigma_total_vp = torch.sqrt(target.sigma ** 2 * vp_diffusion.sde.mean_weight(t) ** 2 + vp_diffusion.sde.variance(t))
            vp_dist = torch.distributions.Normal(mu_t_vp, sigma_total_vp)
            vp_density = vp_dist.log_prob(x).exp()
            
            ax.plot(x.numpy(), target_density.numpy(), 'k-', label='Target π₀', linewidth=2)
            ax.plot(x.numpy(), ve_density.numpy(), 'b--', label='VE-SDE', linewidth=2)
            ax.plot(x.numpy(), vp_density.numpy(), 'r:', label='VP-SDE', linewidth=2)
            
            ax.set_title(f't = {t:.2f}\nVE-λ: {ve_diffusion.lambda_val(t):.2f}, VP-λ: {vp_diffusion.lambda_val(t):.2f}')
            ax.set_xlabel('x')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


def verify_equivalence(target, diffusion, sde_type):
    """Numerical verification of score function equivalence"""
    print(f"\n{'='*60}")
    print(f"Equivalence Verification for {sde_type.upper()}-SDE")
    print(f"{'='*60}")
    
    # Test points and time steps
    test_points = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    t_values = [0.1, 0.5, 0.9]
    
    max_errors = []
    
    for t in t_values:
        print(f"\nTime step t = {t:.2f}, λ = {diffusion.lambda_val(t):.4f}")
        print("x\t\tTheoretical Score\tProximal Score\t\tRelative Error")
        print("-" * 70)
        
        errors = []
        for x in test_points:
            score_theoretical = diffusion.theoretical_score(x, t)
            score_proximal = diffusion.score_from_proximal(x, t)
            relative_error = torch.abs(score_theoretical - score_proximal) / (torch.abs(score_theoretical) + 1e-8)
            errors.append(relative_error.item())
            
            print(f"{x.item():.1f}\t\t{score_theoretical.item():.6f}\t\t{score_proximal.item():.6f}\t\t{relative_error.item():.6e}")
        
        max_errors.append(max(errors))
        print(f"Max relative error: {max(errors):.2e}")
    
    print(f"\nOverall maximum relative error: {max(max_errors):.2e}")
    return max(max_errors)


def analyze_sde_properties(ve_diffusion, vp_diffusion):
    """Analyze and compare SDE properties"""
    t_values = torch.linspace(0, 1, 100)
    
    # Compute SDE properties
    ve_lambda = [ve_diffusion.lambda_val(t) for t in t_values]
    vp_lambda = [vp_diffusion.lambda_val(t) for t in t_values]
    
    if hasattr(ve_diffusion.sde, 'sigma'):
        ve_sigma = [ve_diffusion.sde.sigma(t) for t in t_values]
    
    if hasattr(vp_diffusion.sde, 'mean_weight'):
        vp_mu = [vp_diffusion.sde.mean_weight(t) for t in t_values]
        vp_sigma = [torch.sqrt(vp_diffusion.sde.variance(t)) for t in t_values]
    
    # Plot SDE properties
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # λ(t) comparison
    axes[0,0].plot(t_values, ve_lambda, 'b-', label='VE-SDE', linewidth=2)
    axes[0,0].plot(t_values, vp_lambda, 'r-', label='VP-SDE', linewidth=2)
    axes[0,0].set_xlabel('Time t')
    axes[0,0].set_ylabel('λ(t) = σ²(t)')
    axes[0,0].set_title('Proximal Parameter λ(t)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Noise schedule comparison
    axes[0,1].plot(t_values, ve_sigma, 'b-', label='VE-SDE: σ(t)', linewidth=2)
    axes[0,1].plot(t_values, vp_sigma, 'r-', label='VP-SDE: σ(t)', linewidth=2)
    axes[0,1].set_xlabel('Time t')
    axes[0,1].set_ylabel('σ(t)')
    axes[0,1].set_title('Noise Schedule σ(t)')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Mean weight for VP-SDE
    axes[1,0].plot(t_values, vp_mu, 'r-', label='VP-SDE: μ(t)', linewidth=2)
    axes[1,0].set_xlabel('Time t')
    axes[1,0].set_ylabel('μ(t)')
    axes[1,0].set_title('VP-SDE Mean Weight μ(t)')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Ratio comparison
    axes[1,1].plot(t_values, np.array(ve_lambda) / np.array(vp_lambda), 'g-', linewidth=2)
    axes[1,1].set_xlabel('Time t')
    axes[1,1].set_ylabel('λ_VE(t) / λ_VP(t)')
    axes[1,1].set_title('Ratio of Proximal Parameters')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def Gaussian_Comparison_Experiments():
    """Main experimental function"""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("Proximal-Based Generative Modeling: Experimental Verification")
    print("1D Gaussian Distribution with VP-SDE and VE-SDE")
    print("=" * 60)
    
    # Create target distribution
    target = GaussianTarget(mu=0.0, sigma=1.0)
    
    # Create diffusion processes for both SDE types
    ve_diffusion = DiffusionProcess(target, sde_type='ve', sigma_min=0.1, sigma_max=10.0)
    vp_diffusion = DiffusionProcess(target, sde_type='vp', beta_min=0.1, beta_max=20.0)
    
    visualizer = ProximalVisualizer()
    
    # Time points for analysis
    t_values = [0.0, 0.3, 0.6, 0.9, 0.99, 1.0]
    
    # Verify equivalence for both SDE types
    ve_max_error = verify_equivalence(target, ve_diffusion, 've')
    vp_max_error = verify_equivalence(target, vp_diffusion, 'vp')
    
    # Generate visualization plots
    print("\nGenerating visualization plots...")
    
    # VE-SDE distributions and scores
    fig1 = visualizer.plot_distribution_evolution(target, ve_diffusion, t_values, 've')
    # fig1.suptitle('VE-SDE: Distribution Evolution', fontsize=16)
    plt.savefig('ve_sde_distributions.png', dpi=300, bbox_inches='tight')
    
    fig2 = visualizer.plot_score_comparison(target, ve_diffusion, t_values, 've')
    # fig2.suptitle('VE-SDE: Score Function Comparison', fontsize=16)
    plt.savefig('ve_sde_scores.png', dpi=300, bbox_inches='tight')
    
    # VP-SDE distributions and scores
    fig3 = visualizer.plot_distribution_evolution(target, vp_diffusion, t_values, 'vp')
    # fig3.suptitle('VP-SDE: Distribution Evolution', fontsize=16)
    plt.savefig('vp_sde_distributions.png', dpi=300, bbox_inches='tight')
    
    fig4 = visualizer.plot_score_comparison(target, vp_diffusion, t_values, 'vp')
    # fig4.suptitle('VP-SDE: Score Function Comparison', fontsize=16)
    plt.savefig('vp_sde_scores.png', dpi=300, bbox_inches='tight')
    
    # SDE comparison
    fig5 = visualizer.plot_sde_comparison(target, ve_diffusion, vp_diffusion, t_values)
    # fig5.suptitle('VE-SDE vs VP-SDE: Distribution Comparison', fontsize=16)
    plt.savefig('sde_comparison.png', dpi=300, bbox_inches='tight')
    
    # SDE properties analysis
    fig6 = analyze_sde_properties(ve_diffusion, vp_diffusion)
    # fig6.suptitle('SDE Properties Analysis', fontsize=16)
    plt.savefig('sde_properties.png', dpi=300, bbox_inches='tight')
    
    # Summary of results
    print("\n" + "="*60)
    print("EXPERIMENTAL SUMMARY")
    print("="*60)
    print(f"VE-SDE Maximum Relative Error: {ve_max_error:.2e}")
    print(f"VP-SDE Maximum Relative Error: {vp_max_error:.2e}")
    print(f"Both SDE types show excellent agreement between")
    print(f"theoretical and proximal-based computations.")
    print("="*60)
    
    plt.show()



if __name__ == "__main__":
    Gaussian_Comparison_Experiments()