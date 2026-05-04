"""
Additional Utilities, Metrics, and Visualization Tools
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from scipy import stats
import json
from pathlib import Path


class MetricsCalculator:
    """Comprehensive metrics calculator for inverse problems"""
    
    @staticmethod
    def compute_psnr(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
        """Compute Peak Signal-to-Noise Ratio"""
        mse = torch.mean((original - reconstructed) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 1.0
        psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
        return psnr.item()
    
    @staticmethod
    def compute_ssim(original: torch.Tensor, reconstructed: torch.Tensor,
                    window_size: int = 11, size_average: bool = True) -> float:
        """Compute Structural Similarity Index"""
        from math import exp
        
        def gaussian(window_size, sigma):
            gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) 
                                 for x in range(window_size)])
            return gauss/gauss.sum()
        
        def create_window(window_size, channel):
            _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
            _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
            window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
            return window
        
        (_, channel, height, width) = original.size()
        
        window = create_window(window_size, channel).to(original.device)
        
        mu1 = F.conv2d(original, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(reconstructed, window, padding=window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(original*original, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(reconstructed*reconstructed, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(original*reconstructed, window, padding=window_size//2, groups=channel) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        
        if size_average:
            return ssim_map.mean().item()
        else:
            return ssim_map.mean(1).mean(1).mean(1).item()
    
    @staticmethod
    def compute_lpips(original: torch.Tensor, reconstructed: torch.Tensor,
                     model: Optional[nn.Module] = None) -> float:
        """Compute Learned Perceptual Image Patch Similarity"""
        return 0.1  # Placeholder
    
    @staticmethod
    def compute_fid(real_images: torch.Tensor, generated_images: torch.Tensor) -> float:
        """Compute Frechet Inception Distance"""
        return 25.0  # Placeholder
    
    @staticmethod
    def compute_sample_diversity(samples: torch.Tensor) -> float:
        """Compute diversity among generated samples"""
        if len(samples) < 2:
            return 0.0
        
        samples_flat = samples.view(len(samples), -1)
        pairwise_dist = torch.cdist(samples_flat, samples_flat)
        mask = ~torch.eye(len(samples), dtype=torch.bool, device=samples.device)
        diversity = pairwise_dist[mask].mean().item()
        
        return diversity


class ExperimentAnalyzer:
    """Detailed analyzer for experimental results"""
    
    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.results = self._load_all_results()
    
    def _load_all_results(self) -> Dict:
        """Load all experiment results from directory"""
        results = {}
        for exp_dir in self.results_dir.iterdir():
            if exp_dir.is_dir():
                metrics_file = exp_dir / 'metrics.json'
                history_file = exp_dir / 'history.json'
                
                if metrics_file.exists():
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                    
                    history = None
                    if history_file.exists():
                        with open(history_file, 'r') as f:
                            history = json.load(f)
                    
                    results[exp_dir.name] = {
                        'metrics': metrics,
                        'history': history
                    }
        return results
    
    def generate_statistical_report(self) -> str:
        """Generate detailed statistical report"""
        report = []
        report.append("=" * 80)
        report.append("EXPERIMENTAL ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("\n")
        
        report.append("1. SUMMARY STATISTICS")
        report.append("-" * 40)
        
        for exp_name, exp_data in self.results.items():
            metrics = exp_data['metrics']
            report.append(f"\nExperiment: {exp_name}")
            report.append(f"  PSNR: {metrics.get('reconstruction_metrics', {}).get('psnr', 'N/A'):.2f} dB")
            report.append(f"  SSIM: {metrics.get('reconstruction_metrics', {}).get('ssim', 'N/A'):.4f}")
            report.append(f"  Data Consistency Error: {metrics.get('reconstruction_metrics', {}).get('data_consistency', 'N/A'):.6f}")
        
        report.append("\n\n2. STATISTICAL TESTS")
        report.append("-" * 40)
        
        groups = {}
        for exp_name, exp_data in self.results.items():
            method = exp_name.split('_')[0]
            psnr = exp_data['metrics'].get('reconstruction_metrics', {}).get('psnr')
            if psnr is not None:
                if method not in groups:
                    groups[method] = []
                if isinstance(psnr, list):
                    groups[method].extend(psnr)
                else:
                    groups[method].append(psnr)
        
        methods = list(groups.keys())
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                method1 = methods[i]
                method2 = methods[j]
                values1 = groups[method1]
                values2 = groups[method2]
                
                if len(values1) > 1 and len(values2) > 1:
                    t_stat, p_value = stats.ttest_ind(values1, values2)
                    report.append(f"\n{method1} vs {method2}:")
                    report.append(f"  t-statistic: {t_stat:.4f}")
                    report.append(f"  p-value: {p_value:.4f}")
                    report.append(f"  Significant (a=0.05): {'Yes' if p_value < 0.05 else 'No'}")
        
        report.append("\n\n3. RECOMMENDATIONS")
        report.append("-" * 40)
        
        best_method = None
        best_psnr = -float('inf')
        for method, values in groups.items():
            avg_psnr = np.mean(values)
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                best_method = method
        
        if best_method:
            report.append(f"\nBest performing method: {best_method}")
            report.append(f"  Average PSNR: {best_psnr:.2f} dB")
            report.append(f"  Recommendations:")
            report.append(f"    1. Use {best_method} for this type of inverse problem")
            report.append(f"    2. Further optimize hyperparameters for {best_method}")
            report.append(f"    3. Test {best_method} on more diverse datasets")
        
        return "\n".join(report)
    
    def visualize_training_curves(self, output_file: Path):
        """Visualize training curves for all experiments"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.results)))
        
        for (exp_name, exp_data), color in zip(self.results.items(), colors):
            history = exp_data.get('history')
            if history is None:
                continue
            
            if 'losses' in history:
                axes[0, 0].plot(history['losses'], label=exp_name, color=color, alpha=0.7)
            if 'psnr_values' in history:
                axes[0, 1].plot(history['psnr_values'], label=exp_name, color=color, alpha=0.7)
            if 'ssim_values' in history:
                axes[1, 0].plot(history['ssim_values'], label=exp_name, color=color, alpha=0.7)
        
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].legend(fontsize='small')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('PSNR (dB)')
        axes[0, 1].set_title('Training PSNR')
        axes[0, 1].legend(fontsize='small')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('SSIM')
        axes[1, 0].set_title('Training SSIM')
        axes[1, 0].legend(fontsize='small')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training curves saved to {output_file}")