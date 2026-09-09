"""
=============================================================================
 Rectified Flow — U-Net (Velocity Prediction Network)
 
 SAME architecture as the DDPM ConditionalUNet, with two key adaptations:
 
 1. Timestep input: continuous t ∈ [0, 1] instead of discrete t ∈ {0, ..., 999}
    → The sinusoidal embedding handles this naturally (just pass float * 1000)
 
 2. Output semantics: predicts velocity v = ε - x₀ (not noise ε)
    → No architecture change needed! The output tensor has the same shape.
    → The ONLY difference is in the training loss target.
 
 The architecture is: Encoder → Bottleneck → Decoder with skip connections
   Encoder:    64 → 128 → 256 → 512
   Bottleneck: 512 (with self-attention)
   Decoder:    512 → 256 → 128 → 64
=============================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════
# BUILDING BLOCKS
# ═══════════════════════════════════════════════════════════════════════

class SinusoidalPositionEmbedding(nn.Module):
    """
    Timestep embedding using sinusoidal functions (like Transformers).
    
    For Rectified Flow, t is continuous in [0, 1].
    We scale it to [0, 1000] range before embedding for numerical stability
    (matching the DDPM embedding range).
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        Args:
            t: (B,) timesteps, continuous in [0, 1]
        Returns:
            emb: (B, dim) sinusoidal embeddings
        """
        device = t.device
        # Scale to [0, 1000] range for numerical stability
        t_scaled = t.float() * 1000.0

        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t_scaled[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class SelfAttention(nn.Module):
    """
    Self-attention for spatial feature maps.
    Uses F.scaled_dot_product_attention for Flash Attention support (PyTorch 2.0+).
    """

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)

        q = self.q(h).view(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
        k = self.k(h).view(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
        v = self.v(h).view(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)

        # Use PyTorch 2.0+ SDPA for Flash Attention when available
        out = F.scaled_dot_product_attention(q, k, v)       # (B, HW, C)
        out = out.permute(0, 2, 1).view(B, C, H, W)

        return x + self.proj(out)


class ResBlock(nn.Module):
    """Residual block with time and class conditioning."""

    def __init__(self, in_ch, out_ch, time_emb_dim, class_emb_dim, use_attention=False):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch),
        )

        self.class_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(class_emb_dim, out_ch),
        )

        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.attention = SelfAttention(out_ch) if use_attention else nn.Identity()

    def forward(self, x, t_emb, c_emb):
        h = self.conv1(x)
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = h + self.class_mlp(c_emb)[:, :, None, None]
        h = self.conv2(h)
        h = h + self.skip(x)
        h = self.attention(h)
        return h


# ═══════════════════════════════════════════════════════════════════════
# CONDITIONAL U-NET (VELOCITY PREDICTION)
# ═══════════════════════════════════════════════════════════════════════

class ConditionalUNet(nn.Module):
    """
    U-Net for VELOCITY prediction (Rectified Flow).
    
    Architecture identical to the DDPM noise-prediction U-Net.
    The only semantic difference is:
      - DDPM: output = predicted noise ε̂
      - RF:   output = predicted velocity v̂ = ε - x₀
    
    The architecture doesn't need to know — it's the loss function that differs.
    
    Parameters: ~19M with base_channels=64
    """

    def __init__(self, img_channels=1, base_channels=64,
                 time_emb_dim=256, num_classes=3, class_emb_dim=256):
        super().__init__()

        # ── Embeddings ──
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # num_classes + 1 for the "unconditional" class (CFG)
        self.class_embed = nn.Embedding(num_classes + 1, class_emb_dim)

        ch = base_channels  # 64

        # ── Encoder ──
        self.enc_conv0 = nn.Conv2d(img_channels, ch, 3, padding=1)

        self.enc1 = ResBlock(ch, ch, time_emb_dim, class_emb_dim)               # 64
        self.down1 = nn.Conv2d(ch, ch, 4, stride=2, padding=1)                  # /2

        self.enc2 = ResBlock(ch, ch * 2, time_emb_dim, class_emb_dim)           # 128
        self.down2 = nn.Conv2d(ch * 2, ch * 2, 4, stride=2, padding=1)         # /4

        self.enc3 = ResBlock(ch * 2, ch * 4, time_emb_dim, class_emb_dim,
                             use_attention=True)                                 # 256
        self.down3 = nn.Conv2d(ch * 4, ch * 4, 4, stride=2, padding=1)         # /8

        # ── Bottleneck ──
        self.bot1 = ResBlock(ch * 4, ch * 8, time_emb_dim, class_emb_dim,
                             use_attention=True)                                 # 512
        self.bot2 = ResBlock(ch * 8, ch * 8, time_emb_dim, class_emb_dim,
                             use_attention=True)
        self.bot3 = ResBlock(ch * 8, ch * 4, time_emb_dim, class_emb_dim)      # Back to 256

        # ── Decoder ──
        self.up3 = nn.ConvTranspose2d(ch * 4, ch * 4, 4, stride=2, padding=1)   # ×2
        self.dec3 = ResBlock(ch * 8, ch * 2, time_emb_dim, class_emb_dim,
                             use_attention=True)                                  # 128

        self.up2 = nn.ConvTranspose2d(ch * 2, ch * 2, 4, stride=2, padding=1)   # ×2
        self.dec2 = ResBlock(ch * 4, ch, time_emb_dim, class_emb_dim)            # 64

        self.up1 = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)            # ×2
        self.dec1 = ResBlock(ch * 2, ch, time_emb_dim, class_emb_dim)            # 64

        # ── Output ──
        self.out = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, img_channels, 3, padding=1),
        )

    def forward(self, x, t, c):
        """
        Args:
            x: (B, 1, H, W) — noisy image x_t
            t: (B,) — continuous timestep in [0, 1]
            c: (B,) — class label (0=Benign, 1=Malignant, 2=Normal, 3=unconditional)
        
        Returns:
            v: (B, 1, H, W) — predicted velocity (same shape as input)
        """
        # Embeddings
        t_emb = self.time_embed(t)
        c_emb = self.class_embed(c)

        # Encoder
        x0 = self.enc_conv0(x)
        e1 = self.enc1(x0, t_emb, c_emb)
        e1_down = self.down1(e1)

        e2 = self.enc2(e1_down, t_emb, c_emb)
        e2_down = self.down2(e2)

        e3 = self.enc3(e2_down, t_emb, c_emb)
        e3_down = self.down3(e3)

        # Bottleneck
        b = self.bot1(e3_down, t_emb, c_emb)
        b = self.bot2(b, t_emb, c_emb)
        b = self.bot3(b, t_emb, c_emb)

        # Decoder with skip connections
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3, t_emb, c_emb)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2, t_emb, c_emb)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1, t_emb, c_emb)

        return self.out(d1)
