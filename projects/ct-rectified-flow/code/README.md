# Rectified Flow — CT Scan Augmentation

Class-conditional Rectified Flow model for generating synthetic lung cancer CT scans.
Uses the **same U-Net architecture** as the DDPM baseline but with straight-line flow matching — the technique behind FLUX, Stable Diffusion 3, and Seedream.

## What Changed From DDPM?

```
DDPM (Old)                              Rectified Flow (New)
──────────────────────────────────────────────────────────────
Noise schedule:  β₁...βT (1000 steps)  → t ∈ [0, 1] continuous
Forward:         x_t = √ᾱ·x₀ + √(1-ᾱ)·ε  → x_t = (1-t)·x₀ + t·ε
Model predicts:  noise ε                → velocity v = ε - x₀
Training loss:   MSE(ε̂, ε)              → MSE(v̂, ε - x₀)
Sampling:        DDPM/DDIM reverse      → Euler ODE solver
Timestep dist:   Uniform {0,...,999}     → Logit-normal (SD3-style)
Architecture:    U-Net (same!)          → U-Net (same!)
```

Only ~5 lines of math change. All infrastructure (DDP, AMP, W&B, checkpointing) is identical.

## Quick Start

```bash
# Multi-GPU training (2× T4 on Kaggle)
torchrun --nproc_per_node=2 rectified_flow/train.py

# Single GPU
python rectified_flow/train.py

# Generate synthetic images
python rectified_flow/generate.py --checkpoint /path/to/best_rf_model.pt

# With midpoint solver (better quality, 2× slower)
python rectified_flow/generate.py --checkpoint best_rf_model.pt --midpoint --euler_steps 20

# Evaluate quality (FID, t-SNE, nearest-neighbor)
python rectified_flow/evaluate.py --synthetic_dir /path/to/synthetic_images_rf
```

## File Structure

```
rectified_flow/
├── __init__.py          # Package marker
├── config.py            # Configuration (224×224, RF params, GPU auto-detect)
├── flow_scheduler.py    # RectifiedFlowScheduler (the CORE math change)
├── unet.py              # ConditionalUNet (velocity prediction, Flash Attention)
├── ema.py               # Exponential Moving Average
├── train.py             # Training loop (DDP, AMP, W&B, checkpointing)
├── generate.py          # Generate synthetic images from trained model
├── evaluate.py          # FID, t-SNE, nearest-neighbor analysis
└── README.md            # This file
```

## Configuration

Edit `config.py` to change:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `IMG_SIZE` | 224 | Image resolution |
| `EPOCHS` | 3000 | Training epochs |
| `TIMESTEP_SAMPLING` | `logit_normal` | `uniform` or `logit_normal` (SD3-style) |
| `EULER_STEPS` | 30 | ODE solver steps for generation |
| `CFG_SCALE` | 3.0 | Classifier-free guidance strength |
| `BATCH_SIZE` | Auto | Based on GPU VRAM |

## Why Rectified Flow?

| Metric | DDPM | Rectified Flow |
|--------|------|----------------|
| Training convergence | Slower | ~1.3-1.5× faster |
| Inference steps | 50 (DDIM) | 20-30 (Euler) |
| Image quality (FID) | 40-60 | 25-45 (expected) |
| Used by | Original papers | FLUX, SD3, Seedream |
| Code complexity | Complex schedules | Simple linear math |
