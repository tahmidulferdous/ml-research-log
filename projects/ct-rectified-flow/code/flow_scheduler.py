"""
=============================================================================
 Rectified Flow Scheduler
 
 The CORE math change from DDPM. Instead of a curved noise schedule
 (sqrt_alpha_cumprod, betas, etc.), RF uses a simple straight line:
 
   DDPM:  x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε       (curved path)
   RF:    x_t = (1-t) · x_0 + t · ε                (straight line!)
 
   DDPM predicts: noise ε
   RF   predicts: velocity v = ε - x_0
 
 This is what FLUX, SD3, and Seedream all use.
 Reference: "Flow Matching for Generative Modeling" (Lipman et al., 2023)
            "Scaling Rectified Flow Transformers" (Esser et al., ICML 2024)
=============================================================================
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm


class RectifiedFlowScheduler:
    """
    Rectified Flow noise scheduler.
    
    Key differences from DDPM DiffusionScheduler:
    ─────────────────────────────────────────────────────────────────
    │ Aspect              │ DDPM                    │ Rectified Flow │
    │─────────────────────│─────────────────────────│────────────────│
    │ Time range          │ t ∈ {0, ..., T-1}       │ t ∈ [0, 1]     │
    │ Forward process     │ x_t = √ᾱ·x0 + √(1-ᾱ)·ε │ x_t = (1-t)·x0 + t·ε │
    │ Model predicts      │ noise ε                 │ velocity v = ε - x0   │
    │ Training loss       │ MSE(ε̂, ε)               │ MSE(v̂, ε - x0)       │
    │ Sampling            │ DDPM/DDIM reverse        │ Euler ODE solver      │
    │ Pre-computed tables │ betas, alphas_cumprod... │ NONE needed!          │
    ─────────────────────────────────────────────────────────────────
    """

    def __init__(self, sigma_min=1e-4, device='cuda'):
        self.sigma_min = sigma_min
        self.device = device

    def sample_timesteps(self, batch_size, method="logit_normal", mean=0.0, std=1.0):
        """
        Sample timesteps t ∈ [sigma_min, 1] for training.
        
        Args:
            batch_size: Number of timesteps to sample.
            method: "uniform" or "logit_normal"
              - uniform: t ~ U[sigma_min, 1]
              - logit_normal: t = sigmoid(N(mean, std²))
                This is SD3's technique — emphasizes mid-timesteps (t≈0.5)
                where the model has the hardest learning signal.
        
        Returns:
            t: (B,) tensor of timesteps in [sigma_min, 1]
        """
        if method == "logit_normal":
            # Sample from logit-normal distribution
            # t = sigmoid(z), z ~ N(mean, std²)
            z = torch.randn(batch_size, device=self.device) * std + mean
            t = torch.sigmoid(z)
            # Clamp to [sigma_min, 1 - sigma_min] for numerical stability
            t = t.clamp(self.sigma_min, 1.0 - self.sigma_min)
        else:
            # Simple uniform sampling
            t = torch.rand(batch_size, device=self.device)
            t = t * (1.0 - self.sigma_min) + self.sigma_min  # [sigma_min, 1]

        return t

    def add_noise(self, x0, t, noise=None):
        """
        Forward process: interpolate between clean image and noise.
        
        DDPM:  x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε       (curved)
        RF:    x_t = (1-t) · x_0 + t · ε                (straight!)
        
        Args:
            x0: (B, C, H, W) clean images in [-1, 1]
            t:  (B,) timesteps in [0, 1]
            noise: (B, C, H, W) optional pre-generated noise
            
        Returns:
            x_t: (B, C, H, W) noisy image
            noise: (B, C, H, W) the noise used
            velocity: (B, C, H, W) target velocity v = noise - x0
        """
        if noise is None:
            noise = torch.randn_like(x0)

        # Reshape t for broadcasting: (B,) → (B, 1, 1, 1)
        t_view = t.view(-1, 1, 1, 1)

        # Straight-line interpolation (this is the ENTIRE math change from DDPM)
        x_t = (1.0 - t_view) * x0 + t_view * noise

        # Target: velocity field v = noise - x0
        velocity = noise - x0

        return x_t, noise, velocity

    @torch.no_grad()
    def sample_euler(self, model, shape, class_labels, num_steps=30,
                     cfg_scale=3.0, num_classes=3):
        """
        Generate images using Euler ODE solver.
        
        Starting from pure noise (t=1), step backwards to clean image (t=0)
        using the predicted velocity field.
        
        This replaces DDPM's reverse sampling and DDIM.
        
        Algorithm:
            x_1 = noise ~ N(0, I)
            for t from 1 to 0 in num_steps:
                v = model(x_t, t, class)       # predict velocity
                x_{t-dt} = x_t - dt * v        # Euler step
            return x_0
        
        Args:
            model: velocity prediction network (U-Net)
            shape: (B, C, H, W) output shape
            class_labels: (B,) class indices
            num_steps: number of Euler integration steps
            cfg_scale: classifier-free guidance scale
            num_classes: number of classes (for unconditional label)
            
        Returns:
            x_0: (B, C, H, W) generated images in [-1, 1]
        """
        model.eval()
        b = shape[0]
        device = next(model.parameters()).device

        # Start from pure noise at t=1
        x = torch.randn(shape, device=device)

        # Uniform timestep schedule from 1 → 0
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
        dt = 1.0 / num_steps  # Step size

        for i in tqdm(range(num_steps), desc="RF Euler Sampling", disable=not True):
            t_current = timesteps[i]
            t_batch = torch.full((b,), t_current, device=device)

            # ── Classifier-Free Guidance ──
            # Conditional prediction
            v_cond = model(x, t_batch, class_labels)
            # Unconditional prediction
            uncond_labels = torch.full_like(class_labels, num_classes)
            v_uncond = model(x, t_batch, uncond_labels)
            # Guided velocity
            v = v_uncond + cfg_scale * (v_cond - v_uncond)

            # ── Euler step: x_{t-dt} = x_t - dt * v ──
            x = x - dt * v

        # Clamp to valid range
        x = torch.clamp(x, -1.0, 1.0)
        return x

    @torch.no_grad()
    def sample_midpoint(self, model, shape, class_labels, num_steps=20,
                        cfg_scale=3.0, num_classes=3):
        """
        Generate images using Midpoint method (2nd-order ODE solver).
        Better quality than Euler at the same number of steps, at 2× compute cost.
        
        Algorithm:
            x_1 = noise ~ N(0, I)
            for t from 1 to 0:
                v1 = model(x_t, t)               # velocity at current point
                x_mid = x_t - (dt/2) * v1        # half step
                v2 = model(x_mid, t - dt/2)      # velocity at midpoint
                x_{t-dt} = x_t - dt * v2         # full step with midpoint velocity
        """
        model.eval()
        b = shape[0]
        device = next(model.parameters()).device

        x = torch.randn(shape, device=device)
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
        dt = 1.0 / num_steps

        for i in tqdm(range(num_steps), desc="RF Midpoint Sampling", disable=not True):
            t_current = timesteps[i]
            t_mid = t_current - dt / 2

            # ── Step 1: velocity at current point ──
            t_batch = torch.full((b,), t_current, device=device)
            v_cond = model(x, t_batch, class_labels)
            uncond_labels = torch.full_like(class_labels, num_classes)
            v_uncond = model(x, t_batch, uncond_labels)
            v1 = v_uncond + cfg_scale * (v_cond - v_uncond)

            # ── Step 2: velocity at midpoint ──
            x_mid = x - (dt / 2) * v1
            t_mid_batch = torch.full((b,), t_mid.item(), device=device)
            v_cond_mid = model(x_mid, t_mid_batch, class_labels)
            v_uncond_mid = model(x_mid, t_mid_batch, uncond_labels)
            v2 = v_uncond_mid + cfg_scale * (v_cond_mid - v_uncond_mid)

            # ── Full step with midpoint velocity ──
            x = x - dt * v2

        x = torch.clamp(x, -1.0, 1.0)
        return x
