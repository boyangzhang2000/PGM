"""
Inverse Problem Definitions and Forward Operators
Defines various inverse problems including inpainting, super-resolution, deblurring, etc.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List, Callable
import torch.nn.functional as F
import math
import yaml
from blur.models.kernel_encoding.kernel_wizard import KernelWizard

class InverseProblem:
    """Base class for inverse problems"""
    def __init__(self, 
                 forward_operator: Callable,
                 backward_operator: Optional[Callable] = None,
                 noise_level: float = 0.05,
                 problem_type: str = 'inpainting'):
        self.forward = forward_operator
        self.backward = backward_operator
        self.noise_level = noise_level
        self.problem_type = problem_type
        
    def add_noise(self, measurements: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(measurements) * self.noise_level
        return measurements + noise
    
    def data_consistency_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        Ax = self.forward(x)
        loss = torch.mean((y - Ax)**2)
        return loss
    
    def gradient_data_consistency(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.backward is not None:
            residual = self.forward(x) - y
            print(torch.mean(residual**2))
            return 2.0 * self.backward(residual) / x.shape[0]
        else:
            if x.grad is not None:
                x.grad.zero_()
            x.requires_grad_(True)
            with torch.enable_grad():
                loss = self.data_consistency_loss(x, y)
                gradient = torch.autograd.grad(loss, x, create_graph=True)[0]
            x.requires_grad_(False)
            return gradient


class InpaintingProblem(InverseProblem):
    def __init__(self, 
                 image_shape: Tuple[int, int, int],
                 mask_ratio: float = 0.5,
                 noise_level: float = 0.05):
        self.image_shape = image_shape
        self.mask_ratio = mask_ratio
        self.mask = self._generate_mask(image_shape, mask_ratio)
        
        forward_op = lambda x: x * self.mask.to(x.device)
        super().__init__(forward_operator=forward_op, 
                        backward_operator=forward_op,
                        noise_level=noise_level,
                        problem_type='inpainting')
    
    def _generate_mask(self, shape: Tuple[int, int, int], ratio: float) -> torch.Tensor:
        _, h, w = shape
        mask = torch.rand(1, h, w) > ratio
        mask = mask.float()
        mask = mask.repeat(shape[0], 1, 1)
        return mask
    
    def get_degraded_image(self, clean_image: torch.Tensor) -> torch.Tensor:
        y = self.forward(clean_image)
        y = self.add_noise(y)
        return y


class SuperResolutionProblem(InverseProblem):
    def __init__(self, 
                 scale_factor: int = 4,
                 method: str = 'bicubic',
                 noise_level: float = 0.05):
        self.scale_factor = scale_factor
        self.method = method
        
        def forward_op(x):
            if x.dim() == 4:
                b, c, h, w = x.shape
                new_h, new_w = h // scale_factor, w // scale_factor
                return F.interpolate(x, size=(new_h, new_w), mode=method, align_corners=False)
            else:
                c, h, w = x.shape
                new_h, new_w = h // scale_factor, w // scale_factor
                return F.interpolate(x.unsqueeze(0), size=(new_h, new_w), mode=method, align_corners=False).squeeze(0)
        
        def backward_op(y):
            if y.dim() == 4:
                b, c, h, w = y.shape
                new_h, new_w = h * scale_factor, w * scale_factor
                return F.interpolate(y, size=(new_h, new_w), mode=method, align_corners=False)
            else:
                c, h, w = y.shape
                new_h, new_w = h * scale_factor, w * scale_factor
                return F.interpolate(y.unsqueeze(0), size=(new_h, new_w), mode=method, align_corners=False).squeeze(0)
        
        super().__init__(forward_operator=forward_op,
                        backward_operator=backward_op,
                        noise_level=noise_level,
                        problem_type='super_resolution')
    
    def get_degraded_image(self, clean_image: torch.Tensor) -> torch.Tensor:
        y = self.forward(clean_image)
        y = self.add_noise(y)
        return y


class DeblurringProblem(InverseProblem):
    def __init__(self,
                 kernel_size: int = 15,
                 sigma: float = 2.5,
                 noise_level: float = 0.05):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.kernel = self._create_gaussian_kernel(kernel_size, sigma)
        
        def forward_op(x):
            if x.dim() == 4:
                return self._apply_blur_batch(x, self.kernel)
            else:
                return self._apply_blur_single(x, self.kernel)
        
        super().__init__(forward_operator=forward_op,
                        backward_operator=forward_op,
                        noise_level=noise_level,
                        problem_type='deblurring')
    
    def _create_gaussian_kernel(self, size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        x = coords.view(1, -1).repeat(size, 1)
        y = coords.view(-1, 1).repeat(1, size)
        kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)
    
    def _apply_blur_batch(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        kernel = kernel.to(x.device)
        kernel = kernel.repeat(c, 1, 1, 1)
        blurred = F.conv2d(x, kernel, groups=c, padding=self.kernel_size//2)
        return blurred
    
    def _apply_blur_single(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        return self._apply_blur_batch(x.unsqueeze(0), kernel).squeeze(0)
    
    def get_degraded_image(self, clean_image: torch.Tensor) -> torch.Tensor:
        y = self.forward(clean_image)
        y = self.add_noise(y)
        return y


class CompressedSensingProblem(InverseProblem):
    def __init__(self,
                 measurement_ratio: float = 0.25,
                 image_shape: Optional[Tuple[int, int]] = None,
                 noise_level: float = 0.05):
        self.measurement_ratio = measurement_ratio
        
        if image_shape is not None:
            h, w = image_shape
            self.original_dim = h * w
            self.measurement_dim = int(measurement_ratio * h * w)
            self.measurement_matrix = torch.randn(self.measurement_dim, self.original_dim)
            self.measurement_matrix = self._orthonormalize(self.measurement_matrix)
        else:
            self.measurement_matrix = None
        
        def forward_op(x):
            if self.measurement_matrix is None:
                h, w = x.shape[-2:]
                self.original_dim = h * w
                self.measurement_dim = int(measurement_ratio * h * w)
                self.measurement_matrix = torch.randn(self.measurement_dim, self.original_dim)
                self.measurement_matrix = self._orthonormalize(self.measurement_matrix)
            
            if x.dim() == 4:
                b, c, h, w = x.shape
                x_flat = x.view(b, c, -1)
                measurements = torch.matmul(x_flat, self.measurement_matrix.T.to(x.device))
                return measurements.view(b, c, -1)
            else:
                c, h, w = x.shape
                x_flat = x.view(c, -1)
                measurements = torch.matmul(x_flat, self.measurement_matrix.T.to(x.device))
                return measurements.view(c, -1)
        
        def backward_op(y):
            if y.dim() == 4:
                b, c, m = y.shape
                reconstruction = torch.matmul(y, self.measurement_matrix.to(y.device))
                h = w = int(math.sqrt(self.original_dim))
                return reconstruction.view(b, c, h, w)
            else:
                c, m = y.shape
                reconstruction = torch.matmul(y, self.measurement_matrix.to(y.device))
                h = w = int(math.sqrt(self.original_dim))
                return reconstruction.view(c, h, w)
        
        super().__init__(forward_operator=forward_op,
                        backward_operator=backward_op,
                        noise_level=noise_level,
                        problem_type='compressed_sensing')
    
    def _orthonormalize(self, matrix: torch.Tensor) -> torch.Tensor:
        q, _ = torch.linalg.qr(matrix.T)
        return q.T
    
    def get_degraded_measurements(self, clean_image: torch.Tensor) -> torch.Tensor:
        y = self.forward(clean_image)
        y = self.add_noise(y)
        return y


class NonlinearProblem(InverseProblem):
    def __init__(self, yml_path: str = "blur/options/generate_blur/default.yml"):
        self.yml_path = yml_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        with open(yml_path, "r") as f:
            opt = yaml.safe_load(f)["KernelWizard"]
            model_path = opt["pretrained"]
        model = KernelWizard(opt)
        model.eval()
        model.load_state_dict(torch.load(model_path))
        model = model.to(self.device)
        self.model = model
        self.kernel = torch.randn((1, 512, 2, 2)).to(self.device) * 1.2

        def forward_op(x):
            if x.dim() == 4:
                return self._apply_blur_batch(x, self.kernel)
            else:
                return self._apply_blur_single(x, self.kernel)
        
        super().__init__(forward_operator=forward_op, noise_level=0, problem_type='nonlinear')
    
    def _apply_blur_batch(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        kernel = kernel.to(x.device)
        blurred = []
        for i in range(b):
            single = (x[i:i+1]+1)/2
            LQ_tensor = self.model.adaptKernel(single, kernel)
            blurred.append(LQ_tensor*2-1)
        blurred = torch.cat(blurred, dim=0)
        return blurred
    
    def _apply_blur_single(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        return self._apply_blur_batch(x.unsqueeze(0), kernel).squeeze(0)
    
    def get_degraded_image(self, clean_image: torch.Tensor) -> torch.Tensor:
        y = self.forward(clean_image)
        return y