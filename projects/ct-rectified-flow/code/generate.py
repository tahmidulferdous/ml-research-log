"""
=============================================================================
 Rectified Flow — Generate Synthetic CT Images
 
 Usage:
   python rectified_flow/generate.py --checkpoint /path/to/best_rf_model.pt
   
 Loads a trained RF model and generates synthetic CT images using the
 Euler ODE solver. Images are saved organized by class.
=============================================================================
"""

import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path

# Local imports
from config import cfg, gpu_cfg, is_main_process
from unet import ConditionalUNet
from flow_scheduler import RectifiedFlowScheduler


def generate_synthetic_images(checkpoint_path, output_dir=None,
                              num_benign=300, num_normal=100,
                              euler_steps=30, cfg_scale=3.0,
                              batch_size=16, use_midpoint=False):
    """
    Generate synthetic CT images using a trained Rectified Flow model.
    
    Args:
        checkpoint_path: Path to .pt checkpoint file
        output_dir: Where to save images (default: cfg.SYNTHETIC_DIR)
        num_benign: Number of benign images to generate
        num_normal: Number of normal images to generate
        euler_steps: Number of ODE solver steps
        cfg_scale: Classifier-free guidance scale
        batch_size: Generation batch size
        use_midpoint: Use midpoint solver (2nd order, better quality)
    """
    device = cfg.DEVICE
    output_dir = output_dir or cfg.SYNTHETIC_DIR

    print(f"\n{'='*60}")
    print(f"GENERATION — Rectified Flow")
    print(f"  Checkpoint:    {checkpoint_path}")
    print(f"  Output:        {output_dir}")
    print(f"  Resolution:    {cfg.IMG_SIZE}×{cfg.IMG_SIZE}")
    print(f"  Solver:        {'Midpoint (2nd order)' if use_midpoint else 'Euler (1st order)'}")
    print(f"  Steps:         {euler_steps}")
    print(f"  CFG scale:     {cfg_scale}")
    print(f"  Benign:        {num_benign}")
    print(f"  Normal:        {num_normal}")
    print(f"{'='*60}")

    # ── Load Model ──
    model = ConditionalUNet(
        img_channels=cfg.IMG_CHANNELS,
        base_channels=cfg.BASE_CHANNELS,
        time_emb_dim=cfg.TIME_EMB_DIM,
        num_classes=cfg.NUM_CLASSES,
        class_emb_dim=cfg.CLASS_EMB_DIM,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Try loading EMA weights first (better quality), fallback to model weights
    if 'ema_state_dict' in ckpt:
        model.load_state_dict(ckpt['ema_state_dict'])
        print("✅ Loaded EMA weights (best quality)")
    else:
        model.load_state_dict(ckpt['model_state_dict'])
        print("✅ Loaded model weights")

    model.eval()
    print(f"  Epoch: {ckpt.get('epoch', '?')} | Loss: {ckpt.get('best_loss', '?')}")

    # ── Scheduler ──
    scheduler = RectifiedFlowScheduler(sigma_min=cfg.SIGMA_MIN, device=device)

    # ── Choose solver ──
    sample_fn = scheduler.sample_midpoint if use_midpoint else scheduler.sample_euler

    # ── Generate by class ──
    generation_plan = {
        0: ("Benign", num_benign),
        2: ("Normal", num_normal),
    }

    for class_idx, (class_name, num_images) in generation_plan.items():
        if num_images <= 0:
            continue

        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        print(f"\n🎨 Generating {num_images} {class_name} images...")

        generated = 0
        while generated < num_images:
            current_batch = min(batch_size, num_images - generated)
            labels = torch.full((current_batch,), class_idx, device=device, dtype=torch.long)
            shape = (current_batch, cfg.IMG_CHANNELS, cfg.IMG_SIZE, cfg.IMG_SIZE)

            with torch.inference_mode():
                samples = sample_fn(
                    model, shape, labels,
                    num_steps=euler_steps,
                    cfg_scale=cfg_scale,
                    num_classes=cfg.NUM_CLASSES
                )

            # Convert [-1, 1] → [0, 255] and save
            samples = (samples.clamp(-1, 1) + 1) / 2 * 255
            samples = samples.cpu().numpy().astype(np.uint8)

            for j in range(current_batch):
                img = samples[j, 0]  # Remove channel dim
                img_path = os.path.join(class_dir, f"{class_name.lower()}_{generated:04d}.png")
                Image.fromarray(img, mode='L').save(img_path)
                generated += 1

        print(f"  ✅ {generated} {class_name} images saved to {class_dir}")

    print(f"\n{'='*60}")
    print(f"✅ Generation complete! All images saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic CT images with Rectified Flow")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained RF model checkpoint (.pt)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: config SYNTHETIC_DIR)")
    parser.add_argument("--num_benign", type=int, default=300)
    parser.add_argument("--num_normal", type=int, default=100)
    parser.add_argument("--euler_steps", type=int, default=30)
    parser.add_argument("--cfg_scale", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--midpoint", action="store_true",
                        help="Use midpoint solver (2nd order, better quality)")
    args = parser.parse_args()

    generate_synthetic_images(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        num_benign=args.num_benign,
        num_normal=args.num_normal,
        euler_steps=args.euler_steps,
        cfg_scale=args.cfg_scale,
        batch_size=args.batch_size,
        use_midpoint=args.midpoint,
    )
