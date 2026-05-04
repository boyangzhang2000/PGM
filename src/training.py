"""
Training Framework for Inverse Problems
Implements specialized training procedures for inverse problems
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from typing import Optional, Tuple, Dict, List, Callable
from tqdm import tqdm
from pathlib import Path
import PIL.Image
import warnings
warnings.filterwarnings('ignore')


class InverseProblemDataset(Dataset):
    def __init__(self, clean_images: torch.Tensor, 
                 inverse_problem, 
                 num_pairs: Optional[int] = None, 
                 transform: Optional[Callable] = None):
        self.clean_images = clean_images
        self.inverse_problem = inverse_problem
        self.transform = transform
        
        if num_pairs is not None and num_pairs < len(clean_images):
            indices = torch.randperm(len(clean_images))[:num_pairs]
            self.clean_images = clean_images[indices]
    
    def __len__(self) -> int:
        return len(self.clean_images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        clean_img = self.clean_images[idx]
        if self.transform is not None:
            clean_img = self.transform(clean_img)
        degraded = self.inverse_problem.get_degraded_image(clean_img)
        metadata = {'problem_type': self.inverse_problem.problem_type, 'noise_level': self.inverse_problem.noise_level}
        return clean_img, degraded, metadata

class HighResImageDataset(Dataset):
    def __init__(self, data_dir: Path, 
                 inverse_problem, 
                 image_size: Tuple[int, int] = (256, 256), 
                 split: str = 'train', 
                 max_samples: Optional[int] = None, 
                 transform_type: str = 'standard'):
        self.data_dir = Path(data_dir)
        self.inverse_problem = inverse_problem
        self.image_size = image_size
        self.split = split
        self.transform_type = transform_type
        
        self.image_paths = []
        for ext in ['jpg', 'jpeg', 'png', 'webp']:
            self.image_paths.extend(list(self.data_dir.glob(f'*.{ext}')))
        
        if max_samples is not None:
            self.image_paths = self.image_paths[:max_samples]
        
        self.transform = self._get_transforms()
        print(f"Loaded {len(self.image_paths)} images from {data_dir}")
    
    def _get_transforms(self):
        if self.transform_type == 'standard':
            return transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        elif self.transform_type == 'augmentation':
            return transforms.Compose([
                transforms.Resize(int(self.image_size[0] * 1.1)),
                transforms.RandomCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        image = PIL.Image.open(img_path).convert('RGB')
        if self.transform: image = self.transform(image)
        degraded = self.inverse_problem.get_degraded_image(image)
        metadata = {'problem_type': self.inverse_problem.problem_type, 'noise_level': self.inverse_problem.noise_level}
        return image, degraded, metadata


class InverseProblemTrainer:
    def __init__(self, model: nn.Module, inverse_problem, sde_type: str = 've', device: str = 'auto', **sde_kwargs):
        self.model = model
        self.inverse_problem = inverse_problem
        self.sde_type = sde_type.lower()
        self.device = torch.device(device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.to(self.device)
        
        if self.sde_type == 've': self.sde = VESDE(**sde_kwargs)
        elif self.sde_type == 'vp': self.sde = VPSDE(**sde_kwargs)
        elif self.sde_type == 'my': self.sde = MYSDE(**sde_kwargs)
        else: raise ValueError(f"Unknown SDE type: {sde_type}")
        
        self.optimizer = None
        self.scheduler = None
        self.train_history = {'losses': [], 'psnr_values': [], 'ssim_values': [], 'learning_rates': []}
    
    def setup_optimizer(self, lr: float = 1e-4, 
                        weight_decay: float = 1e-5, 
                        scheduler_type: str = 'cosine', 
                        warmup_steps: int = 1000):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
        
        if scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10000, eta_min=1e-6)
        elif scheduler_type == 'warmup_cosine':
            def lr_lambda(step):
                if step < warmup_steps: return float(step) / float(max(1, warmup_steps))
                progress = float(step - warmup_steps) / float(max(1, 10000 - warmup_steps))
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def proximal_matching_loss_with_measurements(self, clean_images: torch.Tensor, 
                                                 measurements: torch.Tensor, 
                                                 zeta: float = 0.1) -> Tuple[torch.Tensor, Dict]:
        batch_size = clean_images.shape[0]
        t = torch.rand(batch_size, 1, device=self.device)
        lambda_val = self.sde.lambda_val(t)
        
        noise = torch.randn_like(clean_images)
        noisy_images = clean_images + torch.sqrt(lambda_val.view(-1, 1, 1, 1)) * noise
        pred_prox = self.model(noisy_images, lambda_val)
        
        squared_dist = torch.sum((pred_prox - clean_images) ** 2, dim=(1, 2, 3), keepdim=True)
        pixel_count = clean_images.shape[1] * clean_images.shape[2] * clean_images.shape[3]
        exp_loss = 1 - torch.exp(-squared_dist / (pixel_count * zeta ** 2))
        pm_loss = torch.mean(exp_loss)
        
        if measurements is not None:
            consistent_measurements = self.inverse_problem.forward(pred_prox)
            dc_loss = F.mse_loss(consistent_measurements, measurements)
            total_loss = pm_loss + 0.1 * dc_loss
        else:
            dc_loss = torch.tensor(0.0)
            total_loss = pm_loss
        
        with torch.no_grad():
            psnr = self._compute_psnr(pred_prox, clean_images)
            ssim = self._compute_ssim(pred_prox, clean_images)
        
        info = {'pm_loss': pm_loss.item(), 
                'dc_loss': dc_loss.item(), 
                'total_loss': total_loss.item(), 
                'psnr': psnr.item(), 
                'ssim': ssim.item(), 
                'zeta': zeta}
        return total_loss, info
    
    def _compute_psnr(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = torch.mean((pred - target) ** 2)
        if mse == 0: return torch.tensor(float('inf'))
        return 20 * torch.log10(1.0 / torch.sqrt(mse))
    
    def _compute_ssim(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        mu_x = torch.mean(pred, dim=(1, 2, 3))
        mu_y = torch.mean(target, dim=(1, 2, 3))
        sigma_x = torch.var(pred, dim=(1, 2, 3), unbiased=False)
        sigma_y = torch.var(target, dim=(1, 2, 3), unbiased=False)
        sigma_xy = torch.mean((pred - mu_x[:, None, None, None]) * (target - mu_y[:, None, None, None]), dim=(1, 2, 3))
        ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
        return torch.mean(ssim)
    
    def train_epoch(self, data_loader: DataLoader, zeta: float = 0.1, epoch: int = 0) -> Dict:
        self.model.train()
        epoch_losses = []
        epoch_metrics = {'psnr': [], 'ssim': [], 'pm_loss': [], 'dc_loss': []}
        
        progress_bar = tqdm(data_loader, desc=f'Epoch {epoch}')
        for clean_imgs, measurements, _ in progress_bar:
            clean_imgs, measurements = clean_imgs.to(self.device), measurements.to(self.device)
            loss, info = self.proximal_matching_loss_with_measurements(clean_imgs, None, zeta)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            if self.scheduler is not None: self.scheduler.step()
            
            epoch_losses.append(loss.item())
            epoch_metrics['psnr'].append(info['psnr'])
            epoch_metrics['ssim'].append(info['ssim'])
            epoch_metrics['pm_loss'].append(info['pm_loss'])
            epoch_metrics['dc_loss'].append(info['dc_loss'])
            
            progress_bar.set_postfix({'loss': loss.item(), 'psnr': info['psnr'], 'zeta': zeta})
        
        epoch_stats = {
            'total_loss': np.mean(epoch_losses), 'psnr': np.mean(epoch_metrics['psnr']), 'ssim': np.mean(epoch_metrics['ssim']),
            'pm_loss': np.mean(epoch_metrics['pm_loss']), 'dc_loss': np.mean(epoch_metrics['dc_loss'])
        }
        if self.optimizer is not None: epoch_stats['learning_rate'] = self.optimizer.param_groups[0]['lr']
        return epoch_stats
    
    def train(self, train_loader: DataLoader, 
              val_loader: Optional[DataLoader] = None, 
              num_epochs: int = 100, 
              initial_zeta: float = 0.5, 
              final_zeta: float = 0.01, 
              zeta_decay_epochs: int = 80, 
              eval_freq: int = 5, 
              checkpoint_dir: str = './checkpoints', 
              use_checkpoint: bool = False):

        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Starting training on {self.device} | Problem: {self.inverse_problem.problem_type} | SDE: {self.sde_type.upper()}")
        
        if use_checkpoint:
            try: self.load_checkpoint(os.path.join(checkpoint_dir, 'best_model.pth'))
            except: print("No checkpoints founded.")
            return 

        def zeta_schedule(epoch):
            if epoch < zeta_decay_epochs:
                alpha = epoch / zeta_decay_epochs
                return initial_zeta * (final_zeta / initial_zeta) ** alpha
            return final_zeta
        
        best_val_psnr = 0.0
        for epoch in range(num_epochs):
            current_zeta = zeta_schedule(epoch)
            train_stats = self.train_epoch(train_loader, current_zeta, epoch)
            
            self.train_history['losses'].append(train_stats['total_loss'])
            self.train_history['psnr_values'].append(train_stats['psnr'])
            self.train_history['ssim_values'].append(train_stats['ssim'])
            if 'learning_rate' in train_stats: self.train_history['learning_rates'].append(train_stats['learning_rate'])
            
            if val_loader is not None and (epoch % eval_freq == 0 or epoch == num_epochs - 1):
                val_metrics = self.evaluate(val_loader)
                val_psnr = val_metrics['psnr']
                print(f"Epoch {epoch}: Train PSNR = {train_stats['psnr']:.2f}, Val PSNR = {val_psnr:.2f}, zeta = {current_zeta:.4f}")
                
                if val_psnr > best_val_psnr:
                    best_val_psnr = val_psnr
                    self.save_checkpoint(os.path.join(checkpoint_dir, 'best_model.pth'), epoch=epoch, metrics=val_metrics)
        
        self.save_checkpoint(os.path.join(checkpoint_dir, 'final_model.pth'), epoch=num_epochs, metrics=train_stats)
        print(f"Training completed. Best validation PSNR: {best_val_psnr:.2f}")
        return self.train_history
    
    def evaluate(self, data_loader: DataLoader) -> Dict:
        self.model.eval()
        total_psnr, total_ssim, total_samples = 0, 0, 0
        with torch.no_grad():
            for clean_imgs, measurements, _ in data_loader:
                clean_imgs, measurements = clean_imgs.to(self.device), measurements.to(self.device)
                batch_size = clean_imgs.shape[0]
                t = torch.rand(batch_size, 1, device=self.device)
                lambda_val = self.sde.lambda_val(t)
                
                noise = torch.randn_like(clean_imgs)
                noisy_imgs = clean_imgs + torch.sqrt(lambda_val.view(-1, 1, 1, 1)) * noise
                pred_prox = self.model(noisy_imgs, lambda_val)
                
                total_psnr += self._compute_psnr(pred_prox, clean_imgs).item() * batch_size
                total_ssim += self._compute_ssim(pred_prox, clean_imgs).item() * batch_size
                total_samples += batch_size
        self.model.train()
        return {'psnr': total_psnr / total_samples, 'ssim': total_ssim / total_samples}
    
    def save_checkpoint(self, filepath: str, epoch: int, metrics: Dict):
        checkpoint = {'epoch': epoch, 
                      'model_state_dict': self.model.state_dict(), 
                      'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None, 
                      'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None, 
                      'train_history': self.train_history, 
                      'metrics': metrics, 
                      'sde_type': self.sde_type, 
                      'problem_type': self.inverse_problem.problem_type}

        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer and checkpoint['optimizer_state_dict']: self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']: self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_history = checkpoint.get('train_history', self.train_history)
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")


class VESDE:
    def __init__(self, sigma_min: float = 0.01, sigma_max: float = 50.0):
        self.sigma_min, self.sigma_max = sigma_min, sigma_max
        self.log_ratio = math.log(sigma_max / sigma_min)
    def sigma(self, t: torch.Tensor) -> torch.Tensor: 
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    def sigma_derivative(self, t: torch.Tensor) -> torch.Tensor: 
        return self.sigma(t) * self.log_ratio
    def mean_weight(self, t: torch.Tensor) -> torch.Tensor: 
        return torch.tensor(1.0)
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
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0):
        self.beta_min, self.beta_max = beta_min, beta_max
    def beta(self, t: torch.Tensor) -> torch.Tensor: 
        return self.beta_min + t * (self.beta_max - self.beta_min)
    def beta_integral(self, t: torch.Tensor) -> torch.Tensor: 
        return t * self.beta_min + 0.5 * t ** 2 * (self.beta_max - self.beta_min)
    def mean_weight(self, t: torch.Tensor) -> torch.Tensor: 
        return torch.exp(- self.beta_integral(t))
    def variance(self, t: torch.Tensor) -> torch.Tensor: 
        return 1 - torch.exp(-2*self.beta_integral(t))
    def sigma(self, t: torch.Tensor) -> torch.Tensor: 
        return torch.sqrt(self.variance(t))
    def lambda_val(self, t: torch.Tensor) -> torch.Tensor: 
        return self.variance(t)/self.mean_weight(t)**2
    def forward_process(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = torch.randn_like(x0)
        return self.mean_weight(t) * x0 + self.sigma(t) * z, z
    def reverse_drift_ode(self, x: torch.Tensor, score: torch.Tensor, t: torch.Tensor) -> torch.Tensor: 
        return -self.beta(t) * x - self.beta(t) * score
    def reverse_drift_sde(self, x: torch.Tensor, score: torch.Tensor, t: torch.Tensor) -> torch.Tensor: 
        return -self.beta(t) * x - 2*self.beta(t) * score
    def reverse_diffusion(self, t: torch.Tensor) -> torch.Tensor: 
        return torch.sqrt(2*self.beta(t))


class MYSDE:
    def __init__(self, alpha: float = 10.0, beta: float = 8.0):
        self.alpha, self.beta = alpha, beta
    def mean_weight(self, t: torch.Tensor) -> torch.Tensor: 
        return torch.ones_like(t)
    def lambda_val(self, t: torch.Tensor) -> torch.Tensor: 
        return torch.exp(self.alpha*t-self.beta)
    def variance(self, t: torch.Tensor) -> torch.Tensor: 
        return self.lambda_val(t)*self.mean_weight(t)**2
    def sigma(self, t: torch.Tensor) -> torch.Tensor: 
        return torch.sqrt(self.variance(t))
    def forward_process(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = torch.randn_like(x0)
        return self.mean_weight(t) * x0 + self.sigma(t) * z, z
    def reverse_coeff_ode(self, x: torch.Tensor, 
                          score: torch.Tensor, 
                          t: torch.Tensor, 
                          dt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        lambd_t, lambd_t_prev = self.lambda_val(t), self.lambda_val(t - dt)
        mu_t, mu_t_prev = self.mean_weight(t), self.mean_weight(t - dt)
        coeff1 = torch.sqrt(lambd_t_prev*mu_t_prev/lambd_t/mu_t)
        return coeff1, 1 - coeff1

    def reverse_coeff_sde(self, x: torch.Tensor, 
                          prox: torch.Tensor, 
                          t: torch.Tensor, 
                          dt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        lambd_t, lambd_t_prev = self.lambda_val(t), self.lambda_val(t - dt)
        mu_t, mu_t_prev = self.mean_weight(t), self.mean_weight(t - dt)
        coeff1 = lambd_t_prev*mu_t_prev/lambd_t/mu_t
        coeff2 = mu_t_prev*(1-lambd_t_prev/lambd_t)
        coeff3 = mu_t_prev*torch.sqrt(lambd_t_prev*(1-lambd_t_prev/lambd_t))
        return coeff1, coeff2, coeff3