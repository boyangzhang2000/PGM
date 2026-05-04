"""
Experiment Runner for Inverse Problems
Runs comprehensive experiments comparing with baseline methods
"""

import torch
from torch.cuda import memory_allocated
import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path
import json
import time
import warnings
warnings.filterwarnings('ignore')

from .inverse_problems import InpaintingProblem, SuperResolutionProblem, DeblurringProblem, CompressedSensingProblem, NonlinearProblem
from .training import InverseProblemDataset, HighResImageDataset, InverseProblemTrainer
from .sampling import InverseProblemSampler
from .models import EnhancedProximalNetwork

class InverseProblemExperiment:
    """Comprehensive experiment runner for inverse problems"""
    def __init__(self, dataset_name: str = 'celeba', 
                 problem_type: str = 'inpainting', 
                 image_size: Tuple[int, int] = (64, 64), 
                 batch_size: int = 32, 
                 device: str = 'cuda'):

        self.dataset_name = dataset_name
        self.problem_type = problem_type
        self.image_size = image_size
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.base_dir = Path('./experiments/inverse_problems')
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.inverse_problem = self._setup_inverse_problem()
        self.train_data, self.val_data, self.test_data = self._load_dataset()
        self.results = {}
    
    def _load_dataset(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.dataset_name in ['FFHQ', 'CelebA_HQ', 'LSUN', 'ImageNet']:
            dataset_path = Path(f'./data/{self.dataset_name}')
            all_paths = list(dataset_path.glob('*.jpg')) + list(dataset_path.glob('*.png'))
            all_paths = all_paths[:10000]
            
            train_size = int(0.8 * len(all_paths))
            val_size = int(0.1 * len(all_paths))
            
            train_data = HighResImageDataset(data_dir=dataset_path, 
                                             inverse_problem=self.inverse_problem, 
                                             image_size=self.image_size, 
                                             split='train', 
                                             max_samples=train_size, 
                                             transform_type='standard')

            val_data = HighResImageDataset(data_dir=dataset_path, 
                                           inverse_problem=self.inverse_problem, 
                                           image_size=self.image_size, 
                                           split='val', 
                                           max_samples=val_size, 
                                           transform_type='standard')

            test_data = HighResImageDataset(data_dir=dataset_path, 
                                            inverse_problem=self.inverse_problem, 
                                            image_size=self.image_size, 
                                            split='test', 
                                            max_samples=len(all_paths) - train_size - val_size, 
                                            transform_type='standard')

            return train_data, val_data, test_data
        else:
            print("Fallback to synthetic data")
            synth = self._generate_synthetic_data(10000)
            return synth, synth[:1000], synth[:1000]
            
    def _generate_synthetic_data(self, num_samples: int) -> torch.Tensor:
        c, h, w = 1 if self.dataset_name == 'mnist' else 3, self.image_size[0], self.image_size[1]
        data = torch.randn(num_samples, c, h, w)
        kernel = torch.ones(c, 1, 5, 5) / 25.0
        data = torch.nn.functional.conv2d(data, kernel, padding=2, groups=c)
        data = (data - data.mean()) / data.std()
        return torch.clamp(data, -1, 1)
    
    def _setup_inverse_problem(self):
        if self.problem_type == 'inpainting': 
            return InpaintingProblem(image_shape=(3 if self.dataset_name != 'mnist' else 1, *self.image_size), mask_ratio=0.7, noise_level=0.01)
        elif self.problem_type == 'super_resolution': 
            return SuperResolutionProblem(scale_factor=4, method='bilinear', noise_level=0.01)
        elif self.problem_type == 'deblurring': 
            return DeblurringProblem(kernel_size=61, sigma=3.0, noise_level=0.01)
        elif self.problem_type == 'nonlinear': 
            return NonlinearProblem()
        elif self.problem_type == 'compressed_sensing': 
            return CompressedSensingProblem(measurement_ratio=0.25, image_shape=self.image_size, noise_level=0.01)
        else: raise ValueError(f"Unknown problem type: {self.problem_type}")
    
    def run_experiment(self, method: str = 'proximal', 
                       sde_type: str = 've', 
                       num_epochs: int = 50, 
                       learning_rate: float = 1e-4, 
                       data_weight: float = 10.0, 
                       experiment_name: Optional[str] = None) -> Dict:
        experiment_name = experiment_name or f"{method}_{sde_type}_{self.problem_type}"
        print(f"\nRunning Experiment: {experiment_name}")
        print("=" * 60)
        
        train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=False)
        val_loader = torch.utils.data.DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)
        
        results = self._run_proximal_experiment(train_loader, 
                                                val_loader, 
                                                sde_type, 
                                                num_epochs, 
                                                learning_rate, 
                                                data_weight, 
                                                experiment_name)
        self.results[experiment_name] = results
        self._save_results(experiment_name, results)
        return results
    
    def _run_proximal_experiment(self, train_loader, 
                                 val_loader, 
                                 sde_type: str, 
                                 num_epochs: int, 
                                 learning_rate: float, 
                                 data_weight: float, 
                                 experiment_name: str) -> Dict:
        input_channels = 1 if self.dataset_name == 'mnist' else 3
        
        sample_img = self.train_data[0][0] if isinstance(self.train_data[0], tuple) else self.train_data[0]
        measurement_dim = sample_img.numel()
        
        print(f"GPU memory: {memory_allocated(self.device) / 1024**2:.2f} MB")
        model = EnhancedProximalNetwork(input_channels=input_channels, 
                                        hidden_channels=128, 
                                        time_dim=128, 
                                        measurement_dim=measurement_dim, 
                                        num_encoder_blocks=4, 
                                        num_decoder_blocks=4, 
                                        use_attention=False, 
                                        dropout=0.1)

        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        trainer = InverseProblemTrainer(model=model, 
                                        inverse_problem=self.inverse_problem, 
                                        sde_type=sde_type, 
                                        device=str(self.device))
        trainer.setup_optimizer(lr=learning_rate, scheduler_type='warmup_cosine', warmup_steps=1000)
        
        train_history = trainer.train(train_loader=train_loader, 
                                      val_loader=val_loader, 
                                      num_epochs=num_epochs, 
                                      initial_zeta=1.0, 
                                      final_zeta=0.01, 
                                      zeta_decay_epochs=int(num_epochs * 0.8), 
                                      eval_freq=5, 
                                      checkpoint_dir=str(self.base_dir / 'checkpoints' / experiment_name), 
                                      use_checkpoint=True)
        
        sampler = InverseProblemSampler(model=model, 
                                        inverse_problem=self.inverse_problem, 
                                        sde_type=sde_type, 
                                        device=str(self.device))
        num_samples = 1
        test_samples = torch.stack([self.test_data[i][0] for i in range(num_samples)]).to(self.device)
        test_measurements = torch.stack([self.inverse_problem.get_degraded_image(img) for img in test_samples])
        
        start_time = time.time()
        samples, _, _ = sampler.sample_with_data_guidance_and_optimization(test_measurements, 
                                                                           num_samples=num_samples, 
                                                                           num_steps=100, 
                                                                           data_weight=5, 
                                                                           use_ode=False, 
                                                                           apply_optimization=True)

        print(f"Sampling time: {time.time() - start_time:.2f} seconds with num_samples = {num_samples}")
        
        reconstructions = torch.clamp(samples, -1, 1).to(self.device)
        recon_metrics = self._compute_reconstruction_metrics(test_samples, reconstructions, test_measurements)
        
        return {'method': 'proximal', 
                'sde_type': sde_type, 
                'train_history': train_history, 
                'reconstruction_metrics': recon_metrics, 
                'reconstructions': {'original': test_samples.cpu(), 
                                    'degraded': test_measurements.cpu(), 
                                    'reconstructed': reconstructions.cpu()}}
    
    def _compute_reconstruction_metrics(self, originals: torch.Tensor, 
                                        reconstructions: torch.Tensor, 
                                        measurements: torch.Tensor) -> Dict:
        metrics = {}
        originals = (originals - originals.min()) / (originals.max() - originals.min())
        reconstructions = (reconstructions - reconstructions.min()) / (reconstructions.max() - reconstructions.min())
        
        mse = torch.mean((originals - reconstructions) ** 2, dim=[1, 2, 3])
        metrics['psnr'] = (10 * torch.log10(1.0 / mse)).tolist()
        
        def ssim(img1, img2):
            C1, C2 = 0.01**2, 0.03**2
            mu1, mu2 = torch.mean(img1, dim=[1, 2, 3]), torch.mean(img2, dim=[1, 2, 3])
            sigma1, sigma2 = torch.var(img1, dim=[1, 2, 3]), torch.var(img2, dim=[1, 2, 3])
            sigma12 = torch.mean((img1 - mu1.view(-1, 1, 1, 1)) * (img2 - mu2.view(-1, 1, 1, 1)), dim=[1, 2, 3])
            return ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1 + sigma2 + C2))
        
        metrics['ssim'] = ssim(originals, reconstructions).tolist()
        recon_measurements = self.inverse_problem.forward(reconstructions)
        metrics['data_consistency'] = torch.mean((recon_measurements - measurements) ** 2, 
                                                 dim=tuple(range(1, recon_measurements.dim()))).tolist()
        return metrics
    
    def _save_results(self, experiment_name: str, results: Dict):
        exp_dir = self.base_dir / experiment_name
        exp_dir.mkdir(exist_ok=True)
        with open(exp_dir / 'metrics.json', 'w') as f: 
            json.dump({'reconstruction_metrics': results.get('reconstruction_metrics', {})}, f, indent=2)
        print(f"Results saved to {exp_dir}")

    def _save_results(self, experiment_name: str, results: Dict):
        """Save experiment results"""
        exp_dir = self.base_dir / experiment_name
        exp_dir.mkdir(exist_ok=True)
        
        # Save metrics
        metrics_file = exp_dir / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump({
                'test_metrics': results.get('test_metrics', {}),
                'reconstruction_metrics': results.get('reconstruction_metrics', {})
            }, f, indent=2)
        
        # Save training history
        if 'train_history' in results:
            history_file = exp_dir / 'history.json'
            with open(history_file, 'w') as f:
                json.dump(results['train_history'], f, indent=2)
        
        # Save reconstructions as images
        if 'reconstructions' in results:
            self._save_reconstruction_images(
                exp_dir / 'reconstructions.png',
                results['reconstructions']
            )
        
        print(f"Results saved to {exp_dir}")
    
    def _save_reconstruction_images(self, filepath: Path, reconstructions: Dict):
        """Save reconstruction images for visualization"""
        import matplotlib.pyplot as plt
        
        def denormalize(tensor):
            if tensor.is_cuda:
                tensor = tensor.cpu()
            if tensor.requires_grad:
                tensor = tensor.detach()
            tensor = tensor.clone()
            mean = 0.5
            std = 0.5
            # if tensor.dim() == 4:  # [B, C, H, W]
            #     mean = self.mean.unsqueeze(0)
            #     std = self.std.unsqueeze(0)
            # else:  # [C, H, W]
            #     mean = self.mean
            #     std = self.std
            tensor = tensor * std + mean
            tensor = torch.clamp(tensor, 0, 1)
            return tensor

        originals = reconstructions['original']
        degraded = reconstructions['degraded']
        reconstructed = reconstructions['reconstructed']
        
        n_samples = min(5, len(originals))
        
        # fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
        fig, axes = plt.subplots(3, n_samples, figsize=(4 * n_samples, 12))
        
        for i in range(n_samples):
            # Original
            ax = axes[0, i] if n_samples > 1 else axes[0]
            # img = originals[i].permute(1, 2, 0).numpy()
            img = originals[i].permute(1, 2, 0)
            if img.shape[-1] == 1:
                img = img.squeeze(-1)
            if img.ndim == 2:
                ax.imshow(img, cmap='gray', vmin=-1, vmax=1)
            else:
                ax.imshow(denormalize(img))
            # ax.set_title('Original')
            ax.axis('off')
            
            # Degraded
            ax = axes[1, i] if n_samples > 1 else axes[1]
            img = degraded[i].permute(1, 2, 0)
            if img.shape[-1] == 1:
                img = img.squeeze(-1)
            if img.ndim == 2:
                ax.imshow(img, cmap='gray', vmin=-1, vmax=1)
            else:
                ax.imshow(denormalize(img))
            # ax.set_title('Degraded')
            ax.axis('off')
            
            # Reconstructed
            ax = axes[2, i] if n_samples > 1 else axes[2]
            img = reconstructed[i].permute(1, 2, 0)
            if img.shape[-1] == 1:
                img = img.squeeze(-1)
            if img.ndim == 2:
                ax.imshow(img, cmap='gray', vmin=-1, vmax=1)
            else:
                ax.imshow(denormalize(img))
            # ax.imshow(img, cmap='gray' if img.ndim == 2 else None, vmin=-1, vmax=1)
            # ax.set_title('Reconstructed')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
