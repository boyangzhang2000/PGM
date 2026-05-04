"""
Toy Experiment: Boundary Accumulation in VP-SDE Diffusion
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, gaussian_kde
from scipy.special import erf
import os

# Create directory for saving results
os.makedirs('vp_diffusion_results', exist_ok=True)

# Set parameters
np.random.seed(1)
n_samples = 50000  # Number of samples
T = 200  # Diffusion steps
dt = 1.0 / T  # Time step

# VP SDE parameters (from DDPM paper)
# beta_min = 0.1
# beta_max = 10.0
beta_min = 0.1
beta_max = 10

# Beta schedule (linear)
def beta(t):
    """Linear beta schedule for VP SDE"""
    return beta_min + (beta_max - beta_min) * t

# Alpha functions for VP SDE
def alpha_bar(t):
    """Cumulative product of alphas: ᾱ(t) = exp(-∫₀ᵗ β(s) ds)"""
    # For linear β(t), we have closed form
    integral = beta_min * t + 0.5 * (beta_max - beta_min) * t**2
    return np.exp(-integral)

def sigma(t):
    """Standard deviation at time t: sqrt(1 - ᾱ(t))"""
    return np.sqrt(1 - alpha_bar(t))

# Target distribution: standard normal
mu0 = 0.5
sigma0 = 1
p0 = lambda x: norm.pdf(x, mu0, sigma0)

lb = -1
ub = 1
# Truncated normal distribution (conditional distribution on [-1,1])
def truncated_normal_pdf(x):
    """Conditional distribution of p0 on [-1,1]"""
    # Z = erf(ub/sigma0) - erf(lb/sigma0)  # Normalization constant
    Z = norm.cdf(ub, loc=mu0, scale=sigma0) - norm.cdf(lb, loc=mu0, scale=sigma0)
    pdf = p0(x) / (Z)  # Divide by normalization constant
    pdf = np.where((x >= lb) & (x <= ub), pdf, 0)
    return pdf

# Projection function: restrict samples to [-1,1] interval
def project_to_bounds(x):
    return np.clip(x, lb, ub)

# Method 1: Standard VP reverse diffusion process (no projection)
def vp_backward_diffusion_no_projection():
    """Standard VP reverse diffusion process, no boundary constraints"""
    # Start from forward process end: x_T ~ N(0, 1) since ᾱ(1) ≈ 0
    x = np.random.randn(n_samples)
    
    # Reverse process
    for i in range(T, 0, -1):
        t = i * dt  # Current time
        t_prev = (i-1) * dt  # Previous time
        
        # Compute ᾱ values
        alpha_bar_t = alpha_bar(t)
        alpha_bar_t_prev = alpha_bar(t_prev)
        
        # Compute β_t and other parameters
        beta_t = beta(t)
        
        # Compute α_t = ᾱ(t)/ᾱ(t-1)
        alpha_t = alpha_bar_t / alpha_bar_t_prev if alpha_bar_t_prev > 0 else 0
        
        # Compute σ_t^2 = 1 - ᾱ(t-1)
        sigma_t_sq = 1 - alpha_bar_t_prev
        
        # Score function: ∇log p_t(x) = -(x - sqrt(ᾱ(t))*mean)/σ_t^2
        # Since forward process: x_t = sqrt(ᾱ(t)) * x_0 + sqrt(1-ᾱ(t)) * ε
        # And p_t(x) = N(x; 0, 1) for VP SDE
        # So score = -x / (1 - ᾱ(t)) = -x / σ_t^2
        sigma_t = np.sqrt(1 - alpha_bar_t)
        # score = -x / (sigma_t**2 + 1e-8)  # Add small epsilon for stability
        score = -(x-np.sqrt(alpha_bar_t)*mu0) / (alpha_bar_t**2*sigma0**2+sigma_t**2 + 1e-8)
        
        # Reverse process update (DDPM formulation)
        coeff1 = 1.0 / np.sqrt(alpha_t + 1e-8)
        coeff2 = beta_t / np.sqrt(1 - alpha_bar_t + 1e-8)
        noise = np.sqrt(beta_t)*np.sqrt(dt) * np.random.randn(n_samples)
        
        # x = coeff1 * (x - coeff2 * score) + noise
        x = x + dt*(0.5*beta_t*x+beta_t*score)+noise

    
    return x

# Method 2: VP reverse diffusion process with projection at each step
def vp_backward_diffusion_with_projection_each_step():
    """VP reverse diffusion process with projection to [-1,1] at each step"""
    # Start from forward process end
    x = np.random.randn(n_samples)
    
    # Reverse process with projection at each step
    for i in range(T, 0, -1):
        t = i * dt
        t_prev = (i-1) * dt
        
        alpha_bar_t = alpha_bar(t)
        alpha_bar_t_prev = alpha_bar(t_prev)
        
        beta_t = beta(t)
        alpha_t = alpha_bar_t / alpha_bar_t_prev if alpha_bar_t_prev > 0 else 0
        
        sigma_t = np.sqrt(1 - alpha_bar_t)
        score = -(x-np.sqrt(alpha_bar_t)*mu0) / (alpha_bar_t**2*sigma0**2+sigma_t**2 + 1e-8)
        
        # Reverse process update (DDPM formulation)
        coeff1 = 1.0 / np.sqrt(alpha_t + 1e-8)
        coeff2 = beta_t / np.sqrt(1 - alpha_bar_t + 1e-8)
        noise = np.sqrt(beta_t)*np.sqrt(dt) * np.random.randn(n_samples)
        
        # x = coeff1 * (x - coeff2 * score) + noise
        x = x + dt*(0.5*beta_t*x+beta_t*score)+noise
        
        # Project after each step
        x = project_to_bounds(x)
    
    return x

# Method 3: VP reverse diffusion process with projection only at the last step
def vp_backward_diffusion_with_projection_last_step():
    """VP reverse diffusion process with projection only at the last step"""
    # Start from forward process end
    x = np.random.randn(n_samples)
    
    # Reverse process without intermediate projection
    for i in range(T, 0, -1):
        t = i * dt
        t_prev = (i-1) * dt
        
        alpha_bar_t = alpha_bar(t)
        alpha_bar_t_prev = alpha_bar(t_prev)
        
        beta_t = beta(t)
        alpha_t = alpha_bar_t / alpha_bar_t_prev if alpha_bar_t_prev > 0 else 0
        
        sigma_t = np.sqrt(1 - alpha_bar_t)
        score = -(x-np.sqrt(alpha_bar_t)*mu0) / (alpha_bar_t**2*sigma0**2+sigma_t**2 + 1e-8)
        
        # Reverse process update (DDPM formulation)
        coeff1 = 1.0 / np.sqrt(alpha_t + 1e-8)
        coeff2 = beta_t / np.sqrt(1 - alpha_bar_t + 1e-8)
        noise = np.sqrt(beta_t)*np.sqrt(dt) * np.random.randn(n_samples)
        
        # x = coeff1 * (x - coeff2 * score) + noise
        x = x + dt*(0.5*beta_t*x+beta_t*score)+noise
        
    # Project only at the end
    x = project_to_bounds(x)
    return x

def cal_prox(x, lambda_t, N):
    alpha = 1
    x0 = x
    for i in range(N):
        # y = project_to_bounds(x)
        y = project_to_bounds((x+x0)/2)
        w = ((2*y-x)*sigma0**2+lambda_t*mu0)/(lambda_t+sigma0**2)
        x = x + alpha*(w-y)
    
    return project_to_bounds((x+x0)/2)


def mu_t(t):
    # return np.sqrt(alpha_bar(t))
    # integral = 0.01 + 0.1 * t + 5 * t**2
    # return np.exp(-integral)
    # return np.cos(0.02+t*1.4)
    return 1

def lambd_t(t):
    # return (1 - alpha_bar(t))/alpha_bar(t)
    # return 1/mu_t(t)**2-1
    return np.exp(10*t-8)
    # return 10*t+0.01

# Method 5: VP reverse diffusion process with real prox 
def vp_backward_diffusion_with_real_prox():
    """VP reverse diffusion process with prox"""
    # Start from forward process end
    x = np.random.randn(n_samples)
    # x = np.random.randn(n_samples) * np.sqrt(lambd_t(1)*mu_t(1)+sigma0**2) + mu0
    
    # Reverse process without intermediate projection
    for i in range(T, 0, -1):
        t = i * dt
        t_prev = (i-1) * dt

        lambd = lambd_t(t)
        # print(lambd)
        prox = project_to_bounds((lambd*mu0+sigma0**2*x/mu_t(t))/(lambd+sigma0**2))

        coeff1 = lambd_t(t_prev)*mu_t(t_prev)/lambd_t(t)/mu_t(t)
        coeff2 = mu_t(t_prev)*(1-lambd_t(t_prev)/lambd_t(t))
        coeff3 = mu_t(t_prev)*np.sqrt(lambd_t(t_prev))*np.sqrt(1-lambd_t(t_prev)/lambd_t(t))
        # print(coeff1,coeff2,coeff3)

        # x = coeff1*x + coeff2*prox + coeff3*np.random.randn(n_samples)
        x = prox + np.sqrt(2*lambd)*np.random.randn(n_samples)
    # Project only at the end
    # x = project_to_bounds(x)
    return x


# Method 4: VP reverse diffusion process with prox
def vp_backward_diffusion_with_prox():
    """VP reverse diffusion process with prox"""
    # Start from forward process end
    x = np.random.randn(n_samples)
    
    # Reverse process without intermediate projection
    for i in range(T, 0, -1):
        t = i * dt
        t_prev = (i-1) * dt

        lambd = lambd_t(t)
        prox = cal_prox(x/np.sqrt(mu_t(t)+1e-8), lambd, 2)
        # prox = project_to_bounds((lambd*mu0+sigma0**2*x/mu_t(t))/(lambd+sigma0**2))
        # prox = (lambd*mu0+sigma0**2*x/mu_t(t))/(lambd+sigma0**2)

        coeff1 = lambd_t(t_prev)*mu_t(t_prev)/lambd_t(t)/mu_t(t)
        coeff2 = mu_t(t_prev)*(1-lambd_t(t_prev)/lambd_t(t))
        coeff3 = mu_t(t_prev)*np.sqrt(lambd_t(t_prev))*np.sqrt(1-lambd_t(t_prev)/lambd_t(t))

        # x = coeff1*x + coeff2*prox + coeff3*np.random.randn(n_samples)
        x = prox + np.sqrt(2*lambd)*np.random.randn(n_samples)
    # Project only at the end
    # x = project_to_bounds(x)
    return x

# Method 6: VP reverse diffusion process with pla
def vp_backward_diffusion_with_pla():
    """VP reverse diffusion process with prox"""
    # Start from forward process end
    x = np.random.randn(n_samples)
    
    lambda0 = dt/2
    # Reverse process without intermediate projection
    for i in range(T, 0, -1):
        noise = np.sqrt(dt) * np.random.randn(n_samples)
        
        prox = project_to_bounds((lambda0*mu0+sigma0**2*x)/(lambda0+sigma0**2))
        # prox = project_to_bounds(((1 - alpha_bar_t)/alpha_bar_t*mu0+sigma0**2*x/np.sqrt(alpha_bar_t+1e-8))/((1 - alpha_bar_t)/alpha_bar_t+sigma0**2))
        x = prox+noise
        
    # Project only at the end
    # x = project_to_bounds(x)
    return x

# Generate samples
print("Generating VP SDE samples...")
vp_samples_no_proj = vp_backward_diffusion_no_projection()
vp_samples_proj_each = vp_backward_diffusion_with_projection_each_step()
vp_samples_proj_last = vp_backward_diffusion_with_projection_last_step()
vp_samples_prox = vp_backward_diffusion_with_prox()
vp_samples_real_prox = vp_backward_diffusion_with_real_prox()
vp_samples_pla = vp_backward_diffusion_with_pla()

# Generate samples directly from truncated normal distribution for comparison
def sample_truncated_normal(n, mu=0, sigma=1, a=-1, b=1):
    """Sample from truncated normal distribution"""
    alpha = (a - mu) / sigma
    beta_val = (b - mu) / sigma
    
    # Use inverse transform sampling
    u = np.random.uniform(0, 1, n)
    phi_alpha = norm.cdf(alpha)
    phi_beta = norm.cdf(beta_val)
    
    # Inverse CDF transform
    samples = mu + sigma * norm.ppf(phi_alpha + u * (phi_beta - phi_alpha))
    return samples

samples_truncated = sample_truncated_normal(n_samples)

# Calculate KDE for plotting
print("Calculating KDE for VP SDE...")
x_grid = np.linspace(-1.5, 1.5, 1000)

# Theoretical distributions
p0_vals = p0(x_grid)
truncated_vals = truncated_normal_pdf(x_grid)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
methods = ['reverse_diffusion','proj_each','proj_last','prox_lang','prox_one_step','real_prox']
results = np.array([vp_samples_no_proj,vp_samples_proj_each,vp_samples_proj_last,vp_samples_pla,vp_samples_prox,vp_samples_real_prox])

# methods = ['no_projection','prox','real_prox']
# results = np.array([vp_samples_no_proj,vp_samples_prox,vp_samples_real_prox])
for idx in range(6):
    ax = axes[idx]
    data = results[idx]
    samples = data
        
    ax.hist(samples, bins=100, density=True, alpha=0.6, 
            color='skyblue', edgecolor='black', label=methods[idx])
        
    ax.plot(x_grid, p0_vals, 'k-', linewidth=2, label='Target distribution N(0,1)')
    ax.plot(x_grid, truncated_vals, 'r--', linewidth=2, label='Truncated normal')
    
    ax.set_xlabel('x')
    ax.set_ylabel('density')
    ax.set_title(methods[idx])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([0, 2])

plt.savefig('vp_diffusion_results/hist_comparison.png', dpi=300, bbox_inches='tight')
plt.show()


# Sample KDE
# vp_kde_no_proj = gaussian_kde(vp_samples_no_proj)
# vp_kde_proj_each = gaussian_kde(vp_samples_proj_each)
vp_kde_proj_last = gaussian_kde(vp_samples_proj_last)
vp_kde_prox = gaussian_kde(vp_samples_prox)
vp_kde_real_prox = gaussian_kde(vp_samples_real_prox)
# vp_kde_pla = gaussian_kde(vp_samples_pla)
kde_truncated = gaussian_kde(samples_truncated)

# Plotting VP SDE results
print("Plotting VP SDE results...")
plt.figure(figsize=(14, 10))

plt.plot(x_grid, p0_vals, 'k-', linewidth=2, label='N(0,1)')
plt.plot(x_grid, truncated_vals, 'r--', linewidth=2, label='Truncated normal')
# plt.plot(x_grid, vp_kde_no_proj(x_grid), 'b-', alpha=0.7, label='VP SDE (no projection)')
# plt.plot(x_grid, vp_kde_proj_each(x_grid), 'g-', alpha=0.7, label='VP SDE (projection each step)')
plt.plot(x_grid, vp_kde_proj_last(x_grid), 'm--', alpha=0.7, label='VP SDE (projection)')
# plt.plot(x_grid, vp_kde_pla(x_grid), 'y-', alpha=0.7, label='VP SDE (prox langevin)')
plt.plot(x_grid, vp_kde_prox(x_grid), 'b-', alpha=0.7, label='VP SDE (prox one step)')
plt.plot(x_grid, vp_kde_real_prox(x_grid), 'c-', alpha=0.7, label='VP SDE (real prox)')
plt.xlabel('x')
plt.ylabel('Probability density')
plt.title('VP SDE: Distribution Comparison of Different Sampling Methods')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('vp_diffusion_results/vp_distribution_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Subplot 1: All distributions comparison
plt.subplot(2, 2, 1)
plt.plot(x_grid, p0_vals, 'k-', linewidth=2, label='Target distribution N(0,1)')
plt.plot(x_grid, truncated_vals, 'r--', linewidth=2, label='Truncated normal')
# plt.plot(x_grid, vp_kde_no_proj(x_grid), 'b-', alpha=0.7, label='VP SDE (no projection)')
# plt.plot(x_grid, vp_kde_proj_each(x_grid), 'g-', alpha=0.7, label='VP SDE (projection each step)')
plt.plot(x_grid, vp_kde_proj_last(x_grid), 'm-', alpha=0.7, label='VP SDE (projection last step)')
plt.xlabel('x')
plt.ylabel('Probability density')
plt.title('VP SDE: Distribution Comparison of Different Sampling Methods')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 2: Boundary region zoom
plt.subplot(2, 2, 2)
boundary_region = np.linspace(0.8, 1.2, 400)
plt.plot(boundary_region, truncated_normal_pdf(boundary_region), 'r--', linewidth=2, label='Truncated normal')
# plt.plot(boundary_region, vp_kde_proj_each(boundary_region), 'g-', alpha=0.7, label='Projection each step')
plt.plot(boundary_region, vp_kde_proj_last(boundary_region), 'm-', alpha=0.7, label='Projection last step')
plt.xlabel('x')
plt.ylabel('Probability density')
plt.title('VP SDE: Boundary Region Zoom (0.8-1.2)')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 3: Histogram comparison
plt.subplot(2, 2, 3)
plt.hist(vp_samples_proj_each, bins=50, density=True, alpha=0.5, label='Projection each step', color='green')
plt.hist(vp_samples_proj_last, bins=50, density=True, alpha=0.5, label='Projection last step', color='magenta')
plt.plot(x_grid, truncated_vals, 'r--', linewidth=2, label='Theoretical truncated')
plt.xlabel('x')
plt.ylabel('Density')
plt.title('VP SDE: Histogram Comparison of Projection Methods')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 4: Alpha bar schedule and variance plot
plt.subplot(2, 2, 4)
times = np.linspace(0, 1, 100)
alpha_bars = [alpha_bar(t) for t in times]
sigmas = [np.sqrt(1 - alpha_bar(t)) for t in times]

plt.plot(times, alpha_bars, 'b-', label='ᾱ(t)')
plt.plot(times, sigmas, 'r--', label='σ(t) = sqrt(1-ᾱ(t))')
plt.xlabel('Time t')
plt.ylabel('Value')
plt.title('VP SDE: ᾱ(t) and σ(t) Schedule')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('vp_diffusion_results/vp_distribution_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Calculate statistics for VP SDE
print("\nVP SDE Statistical Analysis:")
print("="*50)

# Calculate boundary accumulation ratio
def boundary_mass(samples, threshold=0.01):
    """Calculate proportion of samples near boundaries (distance < threshold from boundary)"""
    near_boundary = np.sum((np.abs(samples) > 1 - threshold) & (np.abs(samples) <= 1))
    return near_boundary / len(samples)

print(f"Probability of N(0,1) in [-1,1]: {norm.cdf(1) - norm.cdf(-1):.4f}")
print(f"VP SDE projection each step - boundary accumulation (distance<0.01): {boundary_mass(vp_samples_proj_each, 0.01):.4f}")
print(f"VP SDE projection last step - boundary accumulation (distance<0.01): {boundary_mass(vp_samples_proj_last, 0.01):.4f}")
print(f"Truncated normal - theoretical boundary accumulation (distance<0.01): {2*(norm.cdf(1) - norm.cdf(0.99)):.4f}")

# Calculate KL divergence (approximate)
def estimate_kl(p_samples, q_samples, bins=100):
    """Approximate KL divergence between two sample distributions"""
    # Use histograms to estimate probabilities
    min_val, max_val = -1, 1
    p_hist, _ = np.histogram(p_samples, bins=bins, range=(min_val, max_val), density=True)
    q_hist, _ = np.histogram(q_samples, bins=bins, range=(min_val, max_val), density=True)
    
    # Avoid zeros
    p_hist = np.clip(p_hist, 1e-10, None)
    q_hist = np.clip(q_hist, 1e-10, None)
    
    # Calculate KL divergence
    kl = np.sum(p_hist * np.log(p_hist / q_hist)) * (max_val - min_val) / bins
    return kl

print(f"\nVP SDE KL divergence estimate (relative to truncated normal):")
print(f"Projection each step: {estimate_kl(vp_samples_proj_each, samples_truncated):.4f}")
print(f"Projection last step: {estimate_kl(vp_samples_proj_last, samples_truncated):.4f}")

# Track a sample path during VP reverse diffusion
print("\nAnalyzing VP SDE reverse diffusion dynamics...")
def vp_track_sample_path(use_projection=False):
    """Track a sample path during VP reverse diffusion"""
    x = np.random.randn()
    path = [x]
    times = [1.0]
    
    for i in range(T, 0, -1):
        t = i * dt
        t_prev = (i-1) * dt
        
        alpha_bar_t = alpha_bar(t)
        alpha_bar_t_prev = alpha_bar(t_prev)
        
        beta_t = beta(t)
        alpha_t = alpha_bar_t / alpha_bar_t_prev if alpha_bar_t_prev > 0 else 0
        
        sigma_t = np.sqrt(1 - alpha_bar_t)
        score = -x / (sigma_t**2 + 1e-8)
        
        coeff1 = 1.0 / np.sqrt(alpha_t + 1e-8)
        coeff2 = beta_t / np.sqrt(1 - alpha_bar_t + 1e-8)
        noise = np.sqrt(beta_t) * np.random.randn()
        
        x = coeff1 * (x - coeff2 * score) + noise
        
        if use_projection:
            x = project_to_bounds(x)
        
        path.append(x)
        times.append(t - dt)
    
    return np.array(times), np.array(path)

# Track multiple sample paths for VP SDE
plt.figure(figsize=(12, 5))
for _ in range(10):
    times, path = vp_track_sample_path(use_projection=True)
    plt.plot(times, path, alpha=0.5, linewidth=0.5)

plt.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Upper bound')
plt.axhline(y=-1, color='r', linestyle='--', alpha=0.5, label='Lower bound')
plt.xlabel('Time (reverse process)')
plt.ylabel('Sample value')
plt.title('VP SDE: Sample Paths in Reverse Diffusion with Projection (10 samples)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('vp_diffusion_results/vp_sample_paths.png', dpi=300, bbox_inches='tight')
plt.show()

# Compare VE SDE and VP SDE results side by side
# First, let's run VE SDE again with the same target distribution
print("\nRunning VE SDE for comparison...")
# VE SDE parameters
sigma_min = 0.1
sigma_max = 2.0

def ve_sigma(t):
    return sigma_min + (sigma_max - sigma_min) * t

def ve_backward_diffusion_no_projection():
    x = np.random.randn(n_samples) * ve_sigma(1.0)
    for i in range(T, 0, -1):
        t = i * dt
        sigma_t = ve_sigma(t)
        score = -x / (sigma_t**2)
        sigma_prime = (sigma_max - sigma_min)
        b_t = sigma_prime * sigma_t
        x = x + b_t**2 * score * dt + b_t * np.sqrt(dt) * np.random.randn(n_samples)
    return x

def ve_backward_diffusion_with_projection_each_step():
    x = np.random.randn(n_samples) * ve_sigma(1.0)
    for i in range(T, 0, -1):
        t = i * dt
        sigma_t = ve_sigma(t)
        score = -x / (sigma_t**2)
        sigma_prime = (sigma_max - sigma_min)
        b_t = sigma_prime * sigma_t
        x = x + b_t**2 * score * dt + b_t * np.sqrt(dt) * np.random.randn(n_samples)
        x = project_to_bounds(x)
    return x

ve_samples_no_proj = ve_backward_diffusion_no_projection()
ve_samples_proj_each = ve_backward_diffusion_with_projection_each_step()

# Create comparison plot
plt.figure(figsize=(14, 8))

plt.subplot(2, 2, 1)
ve_kde_proj_each = gaussian_kde(ve_samples_proj_each)
vp_kde_proj_each = gaussian_kde(vp_samples_proj_each)
plt.plot(x_grid, truncated_vals, 'r--', linewidth=2, label='Target truncated normal')
plt.plot(x_grid, ve_kde_proj_each(x_grid), 'b-', alpha=0.7, label='VE SDE (projection each step)')
plt.plot(x_grid, vp_kde_proj_each(x_grid), 'g-', alpha=0.7, label='VP SDE (projection each step)')
plt.xlabel('x')
plt.ylabel('Probability density')
plt.title('VE SDE vs VP SDE: Projection Each Step')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
# Boundary accumulation comparison
thresholds = np.linspace(0.001, 0.1, 20)
ve_boundary = [boundary_mass(ve_samples_proj_each, t) for t in thresholds]
vp_boundary = [boundary_mass(vp_samples_proj_each, t) for t in thresholds]
theoretical_boundary = [2*(norm.cdf(1) - norm.cdf(1-t)) for t in thresholds]

plt.plot(thresholds, ve_boundary, 'bo-', label='VE SDE')
plt.plot(thresholds, vp_boundary, 'go-', label='VP SDE')
plt.plot(thresholds, theoretical_boundary, 'r--', label='Theoretical')
plt.xlabel('Distance from boundary')
plt.ylabel('Proportion of samples')
plt.title('Boundary Accumulation Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
# Compare variance schedules
ve_times = np.linspace(0, 1, 100)
ve_sigmas = [ve_sigma(t) for t in ve_times]
vp_sigmas = [np.sqrt(1 - alpha_bar(t)) for t in ve_times]

plt.plot(ve_times, ve_sigmas, 'b-', label='VE SDE: σ(t)')
plt.plot(ve_times, vp_sigmas, 'g--', label='VP SDE: sqrt(1-ᾱ(t))')
plt.xlabel('Time t')
plt.ylabel('Standard deviation')
plt.title('Variance Schedule Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
# KL divergence comparison
methods = ['VE SDE each step', 'VP SDE each step', 'VE SDE last step', 'VP SDE last step']
# Need to generate VE last step samples
def ve_backward_diffusion_with_projection_last_step():
    x = np.random.randn(n_samples) * ve_sigma(1.0)
    for i in range(T, 0, -1):
        t = i * dt
        sigma_t = ve_sigma(t)
        score = -x / (sigma_t**2)
        sigma_prime = (sigma_max - sigma_min)
        b_t = sigma_prime * sigma_t
        x = x + b_t**2 * score * dt + b_t * np.sqrt(dt) * np.random.randn(n_samples)
    x = project_to_bounds(x)
    return x

ve_samples_proj_last = ve_backward_diffusion_with_projection_last_step()

kl_values = [
    estimate_kl(ve_samples_proj_each, samples_truncated),
    estimate_kl(vp_samples_proj_each, samples_truncated),
    estimate_kl(ve_samples_proj_last, samples_truncated),
    estimate_kl(vp_samples_proj_last, samples_truncated)
]

bars = plt.bar(methods, kl_values, color=['blue', 'green', 'lightblue', 'lightgreen'])
plt.ylabel('KL Divergence (vs truncated normal)')
plt.title('KL Divergence Comparison')
plt.xticks(rotation=45, ha='right')
# Add value labels on bars
for bar, value in zip(bars, kl_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
             f'{value:.4f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('vp_diffusion_results/sde_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Save summary statistics to file
with open('vp_diffusion_results/summary_statistics.txt', 'w') as f:
    f.write("VP SDE Diffusion Results Summary\n")
    f.write("="*50 + "\n")
    f.write(f"Number of samples: {n_samples}\n")
    f.write(f"Diffusion steps: {T}\n")
    f.write(f"Beta_min: {beta_min}\n")
    f.write(f"Beta_max: {beta_max}\n")
    f.write(f"Probability of N(0,1) in [-1,1]: {norm.cdf(1) - norm.cdf(-1):.4f}\n\n")
    
    f.write("Boundary Accumulation (distance < 0.01 from boundary):\n")
    f.write(f"  VP SDE projection each step: {boundary_mass(vp_samples_proj_each, 0.01):.4f}\n")
    f.write(f"  VP SDE projection last step: {boundary_mass(vp_samples_proj_last, 0.01):.4f}\n")
    f.write(f"  VE SDE projection each step: {boundary_mass(ve_samples_proj_each, 0.01):.4f}\n")
    f.write(f"  VE SDE projection last step: {boundary_mass(ve_samples_proj_last, 0.01):.4f}\n")
    f.write(f"  Truncated normal (theoretical): {2*(norm.cdf(1) - norm.cdf(0.99)):.4f}\n\n")
    
    f.write("KL Divergence (relative to truncated normal):\n")
    f.write(f"  VP SDE projection each step: {estimate_kl(vp_samples_proj_each, samples_truncated):.4f}\n")
    f.write(f"  VP SDE projection last step: {estimate_kl(vp_samples_proj_last, samples_truncated):.4f}\n")
    f.write(f"  VE SDE projection each step: {estimate_kl(ve_samples_proj_each, samples_truncated):.4f}\n")
    f.write(f"  VE SDE projection last step: {estimate_kl(ve_samples_proj_last, samples_truncated):.4f}\n\n")
    
    f.write("Key Observations:\n")
    f.write("1. Both VE and VP SDE show boundary accumulation when projection is applied each step.\n")
    f.write("2. Projection at the last step only is closer to the target distribution.\n")
    f.write("3. VP SDE typically has less boundary accumulation than VE SDE.\n")
    f.write("4. The KL divergence is significantly higher for projection each step vs last step only.\n")

print("\nVP SDE results saved in 'vp_diffusion_results' directory:")
print("  - vp_distribution_comparison.png: Main distribution comparison plot")
print("  - vp_sample_paths.png: Sample paths visualization")
print("  - sde_comparison.png: VE vs VP SDE comparison")
print("  - summary_statistics.txt: Statistical summary")

# Create a main result plot showing the key finding
plt.figure(figsize=(10, 6))
plt.plot(x_grid, truncated_vals, 'r--', linewidth=2, label='Target truncated distribution')
plt.plot(x_grid, vp_kde_proj_each(x_grid), 'g-', alpha=0.7, label='VP SDE projection each step')
plt.fill_between(x_grid, 0, vp_kde_proj_each(x_grid), where=(x_grid>=0.99), color='green', alpha=0.3, label='Boundary accumulation')
plt.fill_between(x_grid, 0, vp_kde_proj_each(x_grid), where=(x_grid<=-0.99), color='green', alpha=0.3)
plt.xlabel('x')
plt.ylabel('Probability density')
plt.title('VP SDE: Projection Causes Boundary Accumulation (Same Issue as VE SDE)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('vp_diffusion_results/vp_main_result.png', dpi=300, bbox_inches='tight')
plt.show()

print("  - vp_main_result.png: Main result visualization")