"""
Toy Experiment: Visualize equivalence between score and proximal gradient for Gaussian
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import stats
import seaborn as sns
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection

# Set professional plotting style
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 150,
    'figure.constrained_layout.use': True,
    'font.family': 'DejaVu Sans',
})

class ProximalVisualization:
    def __init__(self):
        self.fig_width = 12
        self.fig_height = 9
        
    def figure1_gaussian_equivalence(self):
        """
        Figure 1: Visualize equivalence between score and proximal gradient for Gaussian
        """
        print("Generating Figure 1: Gaussian Distribution Equivalence")
        
        # Parameters for 2D Gaussian
        mu = torch.tensor([1.0, 0.5])  # Mean
        sigma1, sigma2 = 1.0, 0.6  # Standard deviations
        Sigma = torch.diag(torch.tensor([sigma1**2, sigma2**2]))  # Covariance matrix
        Sigma_inv = torch.inverse(Sigma)
        
        # Smoothing parameter
        lambda_val = 0.5
        
        # Define grid
        n_points = 150
        x1 = torch.linspace(-2, 3.5, n_points)
        x2 = torch.linspace(-1.5, 2.5, n_points)
        X1, X2 = torch.meshgrid(x1, x2, indexing='ij')
        X1_np = X1.numpy()  # Convert to numpy
        X2_np = X2.numpy()  # Convert to numpy
        
        # Define g(x) = 0.5 * (x - mu)^T Sigma^{-1} (x - mu)
        Sigma_t = Sigma + lambda_val * torch.eye(2)
        Sigma_t_inv = torch.inverse(Sigma_t)
        
        # Log density of p_t(x)
        log_pt_grid = torch.zeros(n_points, n_points)
        for i in range(n_points):
            for j in range(n_points):
                point = torch.tensor([X1[i, j], X2[i, j]])
                diff = point - mu
                log_pt_grid[i, j] = -0.5 * diff @ Sigma_t_inv @ diff
        
        # Normalize for visualization
        log_pt_grid = log_pt_grid - log_pt_grid.max()
        log_pt_np = log_pt_grid.numpy()  # Convert to numpy
        
        # Gradient of log p_t(x) = -Sigma_t^{-1} (x - mu)
        def score_function(x):
            return -Sigma_t_inv @ (x - mu)
        
        # Proximal operator
        I = torch.eye(2)
        prox_matrix = torch.inverse(I + lambda_val * Sigma_inv)
        
        def proximal_operator(x):
            return prox_matrix @ (x + lambda_val * Sigma_inv @ mu)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(self.fig_width, self.fig_height), layout='tight')
        
        # Create better contour levels
        levels = np.linspace(np.sqrt(-log_pt_np.min()), np.sqrt(log_pt_np.max()), 20)
        levels = -np.square(levels)
        # Plot contour
        contour = ax.contour(X1_np, X2_np, log_pt_np, 
                            levels=levels, colors='gray', linewidths=0.7, alpha=0.6)
        ax.clabel(contour, inline=True, fontsize=9, fmt='%.1f')
        
        # Fill contour with color gradient
        contourf = ax.contourf(X1_np, X2_np, log_pt_np,
                              levels=levels, cmap='Blues', alpha=0.3)
        
        # Select 3 representative points for clarity
        test_points = torch.tensor([
            [-0.5, -0.5],
            [2.0, 1.5],
            [0.6, -0.4],
            [2.5, 0.0],
            [-0.0, 1.4]
        ])
        
        # Bright colors for better visibility
        colors = ['#FF4444', '#44AA44', '#4444FF', '#AA44AA', '#FFFF44']
        # labels = ['Point A', 'Point B', 'Point C', 'Point D']
        
        # Create legend handles
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        
        legend_handles = []
        
        # Plot mean point
        ax.scatter(mu[0].item(), mu[1].item(), color='black', s=200,
                  marker='*', edgecolor='white', linewidth=2, zorder=10)
        legend_handles.append(Line2D([0], [0], marker='*', color='w', markerfacecolor='black',
                                   markersize=15, label='Optimal'))
        
        # Function to create curly brace
        def add_curly_brace(ax, x1, y1, x2, y2, text, color='black'):
            """Add a curly brace between two points with text"""
            # Calculate midpoint
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            
            # Calculate perpendicular direction
            dx, dy = x2 - x1, y2 - y1
            length = np.sqrt(dx**2 + dy**2)
            perp_dx, perp_dy = -dy/length * 0.3, dx/length * 0.3
            
            # Control points for cubic bezier curve (curly brace shape)
            cp1_x, cp1_y = x1 + dx/3 + perp_dx, y1 + dy/3 + perp_dy
            cp2_x, cp2_y = x2 - dx/3 + perp_dx, y2 - dy/3 + perp_dy
            
            # Create path for curly brace
            vertices = [(x1, y1),
                       (cp1_x, cp1_y),
                       (cp2_x, cp2_y),
                       (x2, y2)]
            
            codes = [Path.MOVETO,
                    Path.CURVE4,
                    Path.CURVE4,
                    Path.CURVE4]
            
            path = Path(vertices, codes)
            patch = PathPatch(path, facecolor='none', edgecolor=color, 
                            linewidth=2, alpha=0.8)
            ax.add_patch(patch)
            
            # Add text near the brace
            text_x, text_y = mx + perp_dx * 1.5, my + perp_dy * 1.5
            ax.text(text_x, text_y, text, fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9),
                   ha='center', va='center')
        
        for idx, (point, color) in enumerate(zip(test_points, colors)):
            # Compute score and proximal gradient
            score = score_function(point)
            P = proximal_operator(point)
            prox_gradient = -(P - point) / lambda_val
            
            # Plot the point
            ax.scatter(point[0].item(), point[1].item(), 
                      color=color, s=100, edgecolor='black', linewidth=1.5,
                      zorder=10)
            # legend_handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
            #                            markersize=10, label=label))
            
            # Plot score vector (thick solid line)
            score_scaled = score * 0.5
            ax.arrow(point[0].item(), point[1].item(),
                    score_scaled[0].item(), score_scaled[1].item(),
                    head_width=0.05, head_length=0.1,
                    fc='#' + color[1:], ec='black', linewidth=2,
                    alpha=0.9, zorder=9, linestyle='solid')
            
            # Plot proximal gradient vector (thick dashed line)
            prox_grad_scaled = prox_gradient * 0.5
            ax.arrow(point[0].item(), point[1].item(),
                    prox_grad_scaled[0].item(), prox_grad_scaled[1].item(),
                    head_width=0.05, head_length=0.1,
                    fc='#' + color[1:], ec='black', linewidth=2,
                    alpha=0.9, zorder=8, linestyle='dashed')
            
            # Add curly braces for vector lengths
            score_norm = torch.norm(score).item()
            prox_norm = torch.norm(prox_gradient).item()
            
            # Calculate points for braces (perpendicular to vectors)
            score_end = point + score_scaled
            prox_end = point + prox_grad_scaled
            
            # Offset for braces (perpendicular direction)
            if idx == 0:
                # For score vector (right side)
                offset = torch.tensor([-score_scaled[1], score_scaled[0]]) * 0.2
                offset = offset / torch.norm(offset)
                brace_start1 = point + offset * 0.1
                brace_end1 = score_end + offset * 0.1
                add_curly_brace(ax, 
                               brace_start1[0].item(), brace_start1[1].item(),
                               brace_end1[0].item(), brace_end1[1].item(),
                               f'$\\|\\nabla\\log p(x)\\|={score_norm:.2f}$', color)
                
                # For proximal vector (left side)
                offset2 = -offset
                brace_start2 = point + offset2 * 0.1
                brace_end2 = prox_end + offset2 * 0.1
                add_curly_brace(ax,
                               brace_start2[0].item(), brace_start2[1].item(),
                               brace_end2[0].item(), brace_end2[1].item(),
                               f'$\\|(x-Prox)/\\lambda\\|={prox_norm:.2f}$', color)
        
        ax.set_xlabel('$x_1$', fontsize=14)
        ax.set_ylabel('$x_2$', fontsize=14)
        # ax.set_title('Equivalence: $\\nabla \\log p_t(x) = (P - x)/\\lambda$', 
        #             fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.set_aspect('equal')
        
        # Add colorbar
        cbar = plt.colorbar(contourf, ax=ax, pad=0.01)
        cbar.set_label('$\\log p(x)$', fontsize=12)
        
        # Create comprehensive legend
        score_handle = Line2D([0], [0], color='black', linewidth=2, label='Gradient $\\nabla \\log p(x)$')
        prox_handle = Line2D([0], [0], color='black', linewidth=2, linestyle='dashed', 
                           label='Proximal $(x-Prox)/\\lambda$')
        
        legend_handles.extend([score_handle, prox_handle])
        
        # Create legend
        ax.legend(handles=legend_handles, loc='upper left', framealpha=0.9)
        
        # plt.tight_layout()
        plt.savefig('figure1_gaussian_equivalence.png', dpi=600)
        plt.show()
        
        # Print verification
        print("\nVerification of equivalence:")
        print("-" * 40)
        for idx, point in enumerate(test_points):
            score = score_function(point)
            P = proximal_operator(point)
            prox_grad = (P - point) / lambda_val
            diff = torch.norm(score - prox_grad).item()
            print(f"Point {idx+1}: ||∇log p_t - (P-x)/λ|| = {diff:.6e}")
        print("-" * 40)
        
    def figure2_truncated_gaussian_evolution(self):
        """
        Figure 2: 3D visualization of truncated Gaussian evolution over time
        Stack distributions at different t values along the t-axis
        """
        print("\nGenerating Figure 2: 3D Evolution of Truncated Gaussian")
        
        # Parameters for truncated Gaussian
        mu = 1.0
        sigma = 1.0
        bounds = [0.0, 2.0]
        
        # Time points
        t_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        a, b = 10.0, 8.0
        lambda_values = [np.exp(a * t - b) for t in t_values]
        
        # Spatial grid
        x = torch.linspace(-0.5, 2.5, 200)
        x_np = x.numpy()
        
        print("Computing distributions...")
        
        # Define original truncated Gaussian density
        def p0_density(x_vals):
            normal_pdf = torch.exp(-0.5 * ((x_vals - mu) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))
            in_bounds = (x_vals >= bounds[0]) & (x_vals <= bounds[1])
            Z = torch.erf(torch.tensor((bounds[1] - mu) / (sigma * np.sqrt(2)))) - torch.erf(torch.tensor((bounds[0] - mu) / (sigma * np.sqrt(2))))
            Z = Z / 2
            return normal_pdf * in_bounds.float() / Z
        
        # Precompute distributions
        p_t_3d = []
        p_prime_t_3d = []
        
        for t_val, lambda_val in zip(t_values, lambda_values):
            print(f"  Processing t={t_val:.1f}, λ={lambda_val:.4f}")
            
            # Gaussian convolution
            u_grid = torch.linspace(bounds[0] - 2, bounds[1] + 2, 300)
            du = u_grid[1] - u_grid[0]
            
            p0_vals = p0_density(u_grid)
            
            x_expanded = x.unsqueeze(1)
            u_expanded = u_grid.unsqueeze(0)
            kernel = torch.exp(-(x_expanded - u_expanded)**2 / (2 * lambda_val)) / torch.sqrt(torch.tensor(2 * np.pi * lambda_val))
            conv = kernel @ (p0_vals * du)
            
            dx = x[1] - x[0]
            conv = conv / (torch.sum(conv) * dx + 1e-10)
            p_t_3d.append(conv.numpy())
            
            # Moreau approximation
            u_opt_unconstrained = (sigma**2 * x + lambda_val * mu) / (sigma**2 + lambda_val)
            u_opt = torch.clamp(u_opt_unconstrained, bounds[0], bounds[1])
            
            p0_at_opt = p0_density(u_opt)
            exp_term = torch.exp(-(u_opt - x)**2 / (2 * lambda_val))
            moreau = p0_at_opt * exp_term
            
            moreau = moreau / (torch.sum(moreau) * dx + 1e-10)
            p_prime_t_3d.append(moreau.numpy())
        
        # Create 3D figure
        fig = plt.figure(figsize=(self.fig_width, self.fig_height),layout='tight')
        ax = fig.add_subplot(111, projection='3d')
        
        # Colors
        conv_color = '#1f77b4'  # Blue
        moreau_color = '#ff7f0e'  # Orange
        overlap_color = '#d7d7d7'

        # Plot each time point
        for i, (t_val, lambda_val) in enumerate(zip(t_values, lambda_values)):
            # Vertical offset for stacking
            # vertical_offset = i * 0.1
            vertical_offset = i * 0.0
            
            # Get distributions
            y_conv = p_t_3d[i]
            y_moreau = p_prime_t_3d[i]
            
            # Plot Gaussian convolution (solid line)
            ax.plot(x_np, [t_val] * len(x_np), y_conv + vertical_offset,
                   color=conv_color, linewidth=1.5, alpha=0.9,
                   label='Gaussian Conv' if i == 0 else '')
            
            # Plot Moreau approximation (dashed line)
            ax.plot(x_np, [t_val] * len(x_np), y_moreau + vertical_offset,
                   color=moreau_color, linewidth=1.5, alpha=0.9, linestyle='dashed',
                   label='Moreau Approx' if i == 0 else '')
            
            # Create fill between curves with three different cases
            # We'll create polygons for different regions
            
            # Case 1: y_conv > y_moreau (Blue region)
            mask_conv_greater = y_conv > y_moreau
            # mask_conv_greater = (y_conv != y_moreau)
            if np.any(mask_conv_greater):
                # Create polygon vertices: along x where conv > moreau
                x_region = x_np[mask_conv_greater]
                y_conv_region = y_conv[mask_conv_greater]
                y_moreau_region = y_moreau[mask_conv_greater]
                
                # Create polygon vertices
                verts = []
                # Bottom line (along moreau curve)
                for j in range(len(x_np)):
                    # verts.append((x_region[j], t_val, y_moreau_region[j]))
                    verts.append((x_np[j], t_val, 0))
                # Top line (along conv curve, in reverse)
                for j in range(len(x_np)-1, -1, -1):
                    verts.append((x_np[j], t_val, y_conv[j]))
                verts.append(verts[0])  # Close the polygon
                
                # Create polygon collection
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                poly = Poly3DCollection([verts], alpha=0.3, facecolor=conv_color, edgecolor='none')
                ax.add_collection3d(poly)
                
                # Add vertical lines in this region (hatch effect)
                for j in range(0, len(x_region), 3):
                    ax.plot([x_region[j], x_region[j]],  
                           [t_val, t_val],
                           [y_moreau_region[j], y_conv_region[j]],
                           color=conv_color, alpha=0.5, linewidth=0.8)
            
            # Case 2: y_moreau > y_conv (Orange region)
            mask_moreau_greater = y_moreau > y_conv
            # mask_moreau_greater = (y_conv != y_moreau)
            if np.any(mask_moreau_greater):
                # Create polygon vertices
                x_region = x_np[mask_moreau_greater]
                y_conv_region = y_conv[mask_moreau_greater]
                y_moreau_region = y_moreau[mask_moreau_greater]
                
                verts = []
                # Bottom line (along conv curve)
                for j in range(len(x_np)):
                    # verts.append((x_region[j], t_val, y_conv_region[j]))
                    verts.append((x_np[j], t_val, 0))
                # Top line (along moreau curve, in reverse)
                for j in range(len(x_np)-1, -1, -1):
                    verts.append((x_np[j], t_val, y_moreau[j]))
                verts.append(verts[0])
                
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                poly = Poly3DCollection([verts], alpha=0.1, facecolor=moreau_color, edgecolor='none')
                ax.add_collection3d(poly)
                
                # Add vertical lines in this region (hatch effect)
                for j in range(0, len(x_region), 3):
                    ax.plot([x_region[j], x_region[j]], 
                           [t_val, t_val],
                           [y_conv_region[j], y_moreau_region[j]], 
                           color=moreau_color, alpha=0.5, linewidth=0.8)
            
            # # Case 3: Overlap region (where curves are approximately equal)
            # # We'll define overlap as where the absolute difference is less than 1% of max
            # max_val = max(np.max(y_conv), np.max(y_moreau))
            # mask_overlap = np.abs(y_conv - y_moreau) < 0.01 * max_val
            # if np.any(mask_overlap):
            #     # Create polygon vertices (fill from 0 to curve)
            #     x_region = x_np[mask_overlap]
            #     y_conv_region = y_conv[mask_overlap]
                
            #     verts = []
            #     # Bottom line (y=0)
            #     for j in range(len(x_region)):
            #         verts.append((x_region[j], 0, t_val))
            #     # Top line (along curve, in reverse)
            #     for j in range(len(x_region)-1, -1, -1):
            #         verts.append((x_region[j], y_conv_region[j], t_val))
            #     verts.append(verts[0])
                
            #     from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            #     poly = Poly3DCollection([verts], alpha=0.3, facecolor=overlap_color, edgecolor='none')
            #     ax.add_collection3d(poly)
                
            #     # Add vertical lines in overlap region (denser)
            #     for j in range(0, len(x_region), 2):
            #         ax.plot([x_region[j], x_region[j]], 
            #                [0, y_conv_region[j]], 
            #                [t_val, t_val],
            #                color=overlap_color, alpha=0.5, linewidth=0.8)
            
        
        # Set labels and title
        # ax.set_xlabel('Position $x$', fontsize=14, labelpad=15)
        ax.set_xticks([])
        ax.set_ylabel('Time $t$', fontsize=14, labelpad=15)
        ax.set_zlabel('Density', fontsize=14, labelpad=15)
        # ax.set_title('3D Evolution of Truncated Gaussian Distribution\n' +
        #             'Stacked by Time Parameter $t$', 
        #             fontsize=16, fontweight='bold', pad=20)
        
        # Set viewing angle
        ax.view_init(elev=20, azim=25)
        ax.set_box_aspect([1, 2, 1])
        ax.set_xlim(-0.01,2.01)
        ax.set_ylim(0,1.0)
        ax.set_zlim(0,0.7)
        ax.grid(False)
        # Add grid plane at y=0 (density = 0) for better visualization
        # Create a grid on x-t plane at y=0
        x_grid = np.linspace(-0.5, 2.5, 10)
        t_grid = np.linspace(0, 1, 10)
        X_grid, T_grid = np.meshgrid(x_grid, t_grid)
        Y_grid = np.zeros_like(X_grid)
        
        # Plot wireframe grid on x-t plane
        ax.plot_wireframe(X_grid, T_grid, Y_grid, 
                         color='gray', alpha=0.2, linewidth=0.5)
        
        # # Mark truncation boundaries on each time plane
        # for t_val in t_values:
        #     # Left boundary
        #     ax.plot([bounds[0], bounds[0]], [t_val, t_val], [0, 0.2], 
        #            color='gray', linestyle=':', alpha=0.5, linewidth=1)
        #     # Right boundary
        #     ax.plot([bounds[1], bounds[1]], [t_val, t_val], [0, 0.2], 
        #            color='gray', linestyle=':', alpha=0.5, linewidth=1)
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=conv_color, linewidth=3, label='Gaussian Convolution'),
            Line2D([0], [0], color=moreau_color, linewidth=3, linestyle='--', 
                  label='Moreau Approximation'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)
        
        plt.tight_layout()
        plt.savefig('figure2_3d_evolution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\nDistribution Statistics:")
        print("-" * 50)
        print(f"At t=0: λ={lambda_values[0]:.6f} (minimal smoothing)")
        print(f"At t=1: λ={lambda_values[-1]:.6f} (maximal smoothing)")
        print("-" * 50)

class ErrorAnalysisFigure:
    def __init__(self):
        self.fig_width = 12
        self.fig_height = 8
        
    def compute_distributions(self, t, a=10, b=8):
        """
        Compute Gaussian convolution and Moreau approximation distributions for a given time t
        
        Parameters:
        -----------
        t : float
            Time parameter (0 to 1)
        a, b : float
            Parameters for λ(t) = exp(a*t - b)
            
        Returns:
        --------
        x_grid : np.array
            Spatial grid points
        p_t : np.array
            Gaussian convolution distribution
        p_prime_t : np.array
            Moreau approximation distribution
        lambda_val : float
            Smoothing parameter λ
        """
        # Parameters for truncated Gaussian
        mu = 1.0
        sigma = 1.0
        bounds = [0.0, 2.0]
        
        # Calculate λ(t)
        lambda_val = np.exp(a * t - b)
        
        # Spatial grid
        x = torch.linspace(-0.5, 2.5, 300)
        x_grid = x.numpy()
        
        # Define original truncated Gaussian density
        def p0_density(x_vals):
            normal_pdf = torch.exp(-0.5 * ((x_vals - mu) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))
            in_bounds = (x_vals >= bounds[0]) & (x_vals <= bounds[1])
            Z = torch.erf(torch.tensor((bounds[1] - mu) / (sigma * np.sqrt(2)))) - torch.erf(torch.tensor((bounds[0] - mu) / (sigma * np.sqrt(2))))
            Z = Z / 2
            return normal_pdf * in_bounds.float() / Z
        
        # Gaussian convolution
        u_grid = torch.linspace(bounds[0] - 2, bounds[1] + 2, 400)
        du = u_grid[1] - u_grid[0]
        p0_vals = p0_density(u_grid)
        
        # Vectorized convolution
        x_expanded = x.unsqueeze(1)
        u_expanded = u_grid.unsqueeze(0)
        kernel = torch.exp(-(x_expanded - u_expanded)**2 / (2 * lambda_val)) / torch.sqrt(torch.tensor(2 * np.pi * lambda_val))
        conv = kernel @ (p0_vals * du)
        
        # Normalize
        dx = x[1] - x[0]
        conv = conv / (torch.sum(conv) * dx + 1e-10)
        
        # Moreau approximation
        u_opt_unconstrained = (sigma**2 * x + lambda_val * mu) / (sigma**2 + lambda_val)
        u_opt = torch.clamp(u_opt_unconstrained, bounds[0], bounds[1])
        
        p0_at_opt = p0_density(u_opt)
        exp_term = torch.exp(-(u_opt - x)**2 / (2 * lambda_val))
        moreau = p0_at_opt * exp_term
        
        # Normalize
        moreau = moreau / (torch.sum(moreau) * dx + 1e-10)
        
        return x_grid, conv.numpy(), moreau.numpy(), lambda_val
    
    def compute_error_metrics(self, p_t, p_prime_t, x_grid):
        """
        Compute various error metrics between two distributions
        
        Parameters:
        -----------
        p_t, p_prime_t : np.array
            Two probability distributions
        x_grid : np.array
            Spatial grid points
            
        Returns:
        --------
        error_metrics : dict
            Dictionary containing various error metrics
        """
        dx = x_grid[1] - x_grid[0]
        
        # L1 error (Total Variation Distance)
        l1_error = np.sum(np.abs(p_t - p_prime_t)) * dx
        
        # L2 error
        l2_error = np.sqrt(np.sum((p_t - p_prime_t)**2) * dx)
        
        # KL divergence (symmetric approximation)
        eps = 1e-10
        kl_forward = np.sum(p_t * np.log((p_t + eps) / (p_prime_t + eps))) * dx
        kl_backward = np.sum(p_prime_t * np.log((p_prime_t + eps) / (p_t + eps))) * dx
        kl_symmetric = 0.5 * (kl_forward + kl_backward)
        
        # Hellinger distance
        hellinger = np.sqrt(0.5 * np.sum((np.sqrt(p_t) - np.sqrt(p_prime_t))**2) * dx)
        
        return {
            'l1': l1_error,
            'l2': l2_error,
            'kl_symmetric': kl_symmetric,
            'hellinger': hellinger
        }
    
    def plot_figure3(self):
        """
        Figure 3: Error evolution between Gaussian convolution and Moreau approximation
        """
        print("Generating Figure 3: Error Evolution Analysis")
        
        # Time points for analysis
        t_values = np.linspace(0, 1, 11)  # [0, 0.1, 0.2, ..., 1.0]
        print(f"Time points: {t_values}")
        
        # Store results
        results = []
        
        print("\nComputing distributions and errors...")
        for i, t in enumerate(t_values):
            print(f"  t={t:.1f}: ", end="")
            
            # Compute distributions
            x_grid, p_t, p_prime_t, lambda_val = self.compute_distributions(t)
            
            # Compute error metrics
            errors = self.compute_error_metrics(p_t, p_prime_t, x_grid)
            
            # Store results
            results.append({
                't': t,
                'lambda': lambda_val,
                'p_t': p_t,
                'p_prime_t': p_prime_t,
                'errors': errors
            })
            
            # print(f"λ={lambda_val:.4f}, L1 error={errors['l1']:.6f}")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(self.fig_width, self.fig_height), layout='tight')
        ax_main = fig.add_subplot(111)
        
        # Extract errors
        t_array = np.array([r['t'] for r in results])
        l1_errors = np.array([r['errors']['kl_symmetric'] for r in results])
        lambda_array = np.array([r['lambda'] for r in results])
        
        # Create a smooth curve through the points
        from scipy.interpolate import make_interp_spline
        t_smooth = np.linspace(0, 1, 200)
        if len(t_array) > 3:  # Need enough points for spline
            spline = make_interp_spline(t_array, l1_errors, k=1)
            l1_smooth = spline(t_smooth)
        else:
            t_smooth = t_array
            l1_smooth = l1_errors
        
        # Plot smooth error curve
        ax_main.plot(t_smooth, l1_smooth, color='#9467bd', linewidth=3, 
                     alpha=0.8, label='KL Divergence')
        
        # Plot actual data points
        scatter = ax_main.scatter(t_array[1:-1], l1_errors[1:-1], s=100, 
                                 color='#ff7f0e', edgecolor='black', linewidth=2,
                                 zorder=5)
        
        # Highlight t=0 and t=1 points
        zero_idx = np.where(t_array == 0)[0][0]
        one_idx = np.where(t_array == 1)[0][0]
        
        ax_main.scatter([0], [l1_errors[zero_idx]], s=150, 
                       color='#2ca02c', edgecolor='black', linewidth=2,
                       zorder=6, marker='s', label='t=0 (theoretical zero)')
        ax_main.scatter([1], [l1_errors[one_idx]], s=150,
                       color='#d62728', edgecolor='black', linewidth=2,
                       zorder=6, marker='^', label='t=1 (theoretical zero)')
        
        # Fill under curve
        ax_main.fill_between(t_smooth, 0, l1_smooth, alpha=0.2, color='#9467bd')
        
        # Set labels and title
        ax_main.set_xlabel('Time $t$', fontsize=14)
        ax_main.set_ylabel('KL Divergence', fontsize=14)
        
        # Set grid and limits
        ax_main.grid(True, alpha=0.3)
        ax_main.set_xlim(-0.05, 1.05)
        
        # Add legend
        ax_main.legend(loc='upper left', framealpha=0.9, ncol=1,fontsize=16)
        
        plt.tight_layout()
        plt.savefig('figure3_error_evolution.png', dpi=600, bbox_inches='tight')
        plt.show()
        
        return results

# Main execution
if __name__ == "__main__":
    print("=" * 60)
    print("Generating Proximal Method Visualizations")
    print("=" * 60)
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    viz = ProximalVisualization()
    
    # Generate Figure 1
    # viz.figure1_gaussian_equivalence()
    
    # Generate Figure 2
    # viz.figure2_truncated_gaussian_evolution()
    
    # Create and plot Figure 3
    error_figure = ErrorAnalysisFigure()
    results = error_figure.plot_figure3()
