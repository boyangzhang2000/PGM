"""
Enhanced Network Architectures for Inverse Problems
Improved UNet-based architectures with attention and conditioning mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math

class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embedding for time/conditioning"""
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        embeddings = math.log(self.max_period) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        
        if self.dim % 2 == 1:
            embeddings = torch.cat([embeddings, torch.zeros_like(embeddings[:, :1])], dim=-1)
        return embeddings


class AdaptiveGroupNorm(nn.Module):
    """Adaptive group normalization with conditioning"""
    def __init__(self, num_channels: int, cond_dim: int, num_groups: int = 32):
        super().__init__()
        self.num_groups = num_groups
        self.norm = nn.GroupNorm(num_groups, num_channels, affine=False)
        self.cond_proj = nn.Linear(cond_dim, num_channels * 2)
        
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        scale_shift = self.cond_proj(cond)
        scale_shift = scale_shift.unsqueeze(-1).unsqueeze(-1)
        scale, shift = torch.chunk(scale_shift, 2, dim=1)
        return x * (1 + scale) + shift


class ResidualBlockWithConditioning(nn.Module):
    """Residual block with multiple conditioning inputs"""
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, problem_dim: Optional[int] = None, dropout: float = 0.1, use_attention: bool = False):
        super().__init__()
        self.norm1 = AdaptiveGroupNorm(in_channels, time_dim)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels * 2))
        
        if problem_dim is not None:
            self.problem_mlp = nn.Sequential(nn.Linear(problem_dim, out_channels), nn.SiLU(), nn.Linear(out_channels, out_channels * 2))
        
        self.norm2 = AdaptiveGroupNorm(out_channels, time_dim)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.use_attention = use_attention
        if use_attention:
            self.attention = nn.MultiheadAttention(embed_dim=out_channels, num_heads=4, dropout=dropout, batch_first=True)
            self.attn_norm = nn.GroupNorm(32, out_channels)
        
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
            
        self.dropout = nn.Dropout2d(dropout)
        
    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, problem_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x = self.norm1(x, t_embed)
        x = F.silu(x)
        x = self.conv1(x)
        
        t_scale_shift = self.time_mlp(t_embed).unsqueeze(-1).unsqueeze(-1)
        t_scale, t_shift = torch.chunk(t_scale_shift, 2, dim=1)
        x = x * (1 + t_scale) + t_shift
        
        if problem_cond is not None and hasattr(self, 'problem_mlp'):
            p_scale_shift = self.problem_mlp(problem_cond).unsqueeze(-1).unsqueeze(-1)
            p_scale, p_shift = torch.chunk(p_scale_shift, 2, dim=1)
            x = x * (1 + p_scale) + p_shift
        
        x = self.norm2(x, t_embed)
        x = F.silu(x)
        x = self.conv2(x)
        x = self.dropout(x)
        
        if self.use_attention:
            b, c, h, w = x.shape
            x_attn = x.view(b, c, -1).permute(2, 0, 1)
            x_attn, _ = self.attention(x_attn, x_attn, x_attn)
            x_attn = x_attn.permute(1, 2, 0).view(b, c, h, w)
            x = self.attn_norm(x + x_attn)
            
        return self.skip(residual) + x


class UNetEncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, problem_dim: Optional[int] = None, num_layers: int = 1, downsample: bool = True, use_attention: bool = False):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            block_in = in_channels if i == 0 else out_channels
            self.blocks.append(ResidualBlockWithConditioning(block_in, out_channels, time_dim, problem_dim, use_attention=use_attention))
        self.downsample = downsample
        if downsample:
            self.down = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
            
    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, problem_cond: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        skips = []
        for block in self.blocks:
            x = block(x, t_embed, problem_cond)
            skips.append(x)
        if self.downsample:
            x = self.down(x)
        return x, skips


class UNetDecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, problem_dim: Optional[int] = None, num_layers: int = 1, upsample: bool = True, use_attention: bool = False):
        super().__init__()
        if upsample:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, 3, stride=2, padding=1, output_padding=1)
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            block_in = in_channels
            self.blocks.append(ResidualBlockWithConditioning(block_in, out_channels, time_dim, problem_dim, use_attention=use_attention))
            
    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_embed: torch.Tensor, problem_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        if hasattr(self, 'up'):
            x = self.up(x)
        if x.shape != skip.shape:
            skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = x + skip
        for block in self.blocks:
            x = block(x, t_embed, problem_cond)
        return x


class MeasurementConditioning(nn.Module):
    def __init__(self, measurement_dim: int, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(measurement_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, measurements: torch.Tensor) -> torch.Tensor:
        if measurements.dim() == 4 or measurements.dim() == 3:
            b = measurements.shape[0]
            measurements = measurements.view(b, -1)
        return self.encoder(measurements)


class EnhancedProximalNetwork(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int = 64, time_dim: int = 128, measurement_dim: Optional[int] = None, num_encoder_blocks: int = 4, num_decoder_blocks: int = 4, use_attention: bool = True, dropout: float = 0.1):
        super().__init__()
        self.input_channels = input_channels
        self.time_embed = nn.Sequential(SinusoidalPositionEmbedding(time_dim), nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        
        if measurement_dim is not None:
            self.measurement_cond = MeasurementConditioning(measurement_dim, output_dim=time_dim)
            
        self.conv_in = nn.Conv2d(input_channels, hidden_channels, 3, padding=1)
        
        self.encoders = nn.ModuleList()
        channels = hidden_channels
        for i in range(num_encoder_blocks):
            encoder = UNetEncoderBlock(channels, channels * 2 if i < num_encoder_blocks - 1 else channels, time_dim, problem_dim=time_dim if measurement_dim is not None else None, num_layers=1, downsample=(i < num_encoder_blocks - 1), use_attention=use_attention)
            self.encoders.append(encoder)
            if i < num_encoder_blocks - 1: channels *= 2
            
        self.bottleneck = nn.ModuleList([ResidualBlockWithConditioning(channels, channels, time_dim, problem_dim=time_dim if measurement_dim is not None else None, use_attention=True) for _ in range(2)])
        
        self.decoders = nn.ModuleList()
        for i in range(num_decoder_blocks):
            decoder = UNetDecoderBlock(channels, channels // 2 if i > 0 else channels, time_dim, problem_dim=time_dim if measurement_dim is not None else None, num_layers=1, upsample=(i > 0), use_attention=use_attention)
            self.decoders.append(decoder)
            if i > 0: channels //= 2
            
        self.conv_out = nn.Conv2d(hidden_channels, input_channels, 3, padding=1)
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
    def forward(self, x: torch.Tensor, lambda_val: torch.Tensor, measurements: Optional[torch.Tensor] = None) -> torch.Tensor:
        t_embed = self.time_embed(lambda_val.squeeze(1))
        problem_cond = None
        if measurements is not None and hasattr(self, 'measurement_cond'):
            problem_cond = self.measurement_cond(measurements)
            
        h = self.conv_in(x)
        skips = []
        for encoder in self.encoders:
            h, encoder_skips = encoder(h, t_embed, problem_cond)
            skips.extend(encoder_skips)

        for block in self.bottleneck: h = block(h, t_embed, problem_cond)
        
        for decoder in self.decoders:
            skip = skips.pop() if skips else None
            if skip is not None:
                h = decoder(h, skip, t_embed, problem_cond)
            else:
                for block in decoder.blocks: h = block(h, t_embed, problem_cond)
                
        output = self.conv_out(h)
        return output