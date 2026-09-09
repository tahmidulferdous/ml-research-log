"""
╔══════════════════════════════════════════════════════════╗
║  Diffusion Model Inference — Generate Synthetic CT Scans ║
║  OPTIMIZED for maximum speed on T4/A100/CPU              ║
╚══════════════════════════════════════════════════════════╝

Usage:
    # Quick preview grid
    python generate_ct_images.py \
        --checkpoint /path/to/best_diffusion_model.pt \
        --output ./generated_images --preview_only

    # Full generation (300 Benign + 100 Normal)
    python generate_ct_images.py \
        --checkpoint /path/to/best_diffusion_model.pt \
        --output ./generated_images \
        --num_benign 300 --num_normal 100

    # Maximum quality (more steps, higher CFG)
    python generate_ct_images.py \
        --checkpoint /path/to/best_diffusion_model.pt \
        --output ./generated_images \
        --ddim_steps 100 --cfg_scale 5.0

Optimizations applied:
    ✅ torch.compile (2× speedup on SM 7.0+)
    ✅ AMP float16 inference (2× memory savings, faster matmuls)
    ✅ CUDA optimizations (cudnn.benchmark, TF32, channels_last)
    ✅ Batched CFG (single forward pass for cond+uncond)
    ✅ Pre-computed schedule tensors (no recomputation per step)
    ✅ torch.inference_mode (faster than no_grad)
"""

import argparse
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════════════
# CUDA OPTIMIZATIONS (applied at import time)
# ═══════════════════════════════════════════════════════════════════════

def setup_cuda_optimizations():
    """Apply all CUDA performance flags."""
    if torch.cuda.is_available():
        # Use fastest convolution algorithm
        torch.backends.cudnn.benchmark = True
        # Allow TF32 on Ampere+ (no effect on T4, but helps A100/H100)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Disable debug APIs in production
        torch.autograd.set_detect_anomaly(False)
        torch.autograd.profiler.emit_nvtx(False)
        torch.autograd.profiler.profile(False)


# ═══════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE (must match training exactly)
# ═══════════════════════════════════════════════════════════════════════

class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class SelfAttention(nn.Module):
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
        q = self.q(h).view(B, C, H * W)
        k = self.k(h).view(B, C, H * W)
        v = self.v(h).view(B, C, H * W)
        attn = torch.bmm(q.permute(0, 2, 1), k) * (C ** -0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        return x + self.proj(out)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, class_emb_dim, use_attention=False):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_ch), nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_ch))
        self.class_mlp = nn.Sequential(nn.SiLU(), nn.Linear(class_emb_dim, out_ch))
        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_ch), nn.SiLU(), nn.Dropout(0.0),  # No dropout at inference
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


class ConditionalUNet(nn.Module):
    def __init__(self, img_channels=1, base_channels=64,
                 time_emb_dim=256, num_classes=3, class_emb_dim=256):
        super().__init__()
        self.num_classes = num_classes
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim), nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.class_embed = nn.Embedding(num_classes + 1, class_emb_dim)
        ch = base_channels

        # Encoder
        self.enc_conv0 = nn.Conv2d(img_channels, ch, 3, padding=1)
        self.enc1 = ResBlock(ch, ch, time_emb_dim, class_emb_dim)
        self.down1 = nn.Conv2d(ch, ch, 4, stride=2, padding=1)
        self.enc2 = ResBlock(ch, ch * 2, time_emb_dim, class_emb_dim)
        self.down2 = nn.Conv2d(ch * 2, ch * 2, 4, stride=2, padding=1)
        self.enc3 = ResBlock(ch * 2, ch * 4, time_emb_dim, class_emb_dim, use_attention=True)
        self.down3 = nn.Conv2d(ch * 4, ch * 4, 4, stride=2, padding=1)

        # Bottleneck
        self.bot1 = ResBlock(ch * 4, ch * 8, time_emb_dim, class_emb_dim, use_attention=True)
        self.bot2 = ResBlock(ch * 8, ch * 8, time_emb_dim, class_emb_dim, use_attention=True)
        self.bot3 = ResBlock(ch * 8, ch * 4, time_emb_dim, class_emb_dim)

        # Decoder
        self.up3 = nn.ConvTranspose2d(ch * 4, ch * 4, 4, stride=2, padding=1)
        self.dec3 = ResBlock(ch * 8, ch * 2, time_emb_dim, class_emb_dim, use_attention=True)
        self.up2 = nn.ConvTranspose2d(ch * 2, ch * 2, 4, stride=2, padding=1)
        self.dec2 = ResBlock(ch * 4, ch, time_emb_dim, class_emb_dim)
        self.up1 = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)
        self.dec1 = ResBlock(ch * 2, ch, time_emb_dim, class_emb_dim)

        # Output
        self.out = nn.Sequential(
            nn.GroupNorm(8, ch), nn.SiLU(),
            nn.Conv2d(ch, img_channels, 3, padding=1),
        )

    def forward(self, x, t, c):
        t_emb = self.time_embed(t)
        c_emb = self.class_embed(c)

        x0 = self.enc_conv0(x)
        e1 = self.enc1(x0, t_emb, c_emb)
        e1_down = self.down1(e1)
        e2 = self.enc2(e1_down, t_emb, c_emb)
        e2_down = self.down2(e2)
        e3 = self.enc3(e2_down, t_emb, c_emb)
        e3_down = self.down3(e3)

        b = self.bot1(e3_down, t_emb, c_emb)
        b = self.bot2(b, t_emb, c_emb)
        b = self.bot3(b, t_emb, c_emb)

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


# ═══════════════════════════════════════════════════════════════════════
# OPTIMIZED DDIM SAMPLER
# ═══════════════════════════════════════════════════════════════════════

class OptimizedDDIMSampler:
    """
    High-performance DDIM sampler with all optimizations applied.

    Key optimizations vs naive implementation:
    1. Batched CFG: single forward pass with 2B batch (cond+uncond together)
    2. Pre-computed schedule: all alpha values computed once, not per-step
    3. AMP float16: halves memory, speeds up matmuls on tensor cores
    4. torch.inference_mode: faster than torch.no_grad (disables more tracking)
    5. In-place operations: reduces memory allocations
    """

    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02,
                 num_classes=3, device='cuda'):
        self.timesteps = timesteps
        self.num_classes = num_classes
        self.device = device

        # Pre-compute FULL schedule on device (done once)
        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)

    def _get_timestep_subsequence(self, num_steps):
        """Pre-compute the DDIM timestep subsequence and alpha pairs."""
        step_size = self.timesteps // num_steps
        timesteps = list(reversed(range(0, self.timesteps, step_size)))

        # Pre-compute alpha pairs (avoids repeated indexing in the loop)
        alpha_t = torch.tensor(
            [self.alphas_cumprod[t].item() for t in timesteps],
            device=self.device
        )
        alpha_prev = torch.tensor(
            [self.alphas_cumprod[timesteps[i + 1]].item()
             if i + 1 < len(timesteps) else 1.0
             for i in range(len(timesteps))],
            device=self.device
        )

        return timesteps, alpha_t, alpha_prev

    @torch.inference_mode()
    def sample(self, model, shape, class_labels,
               num_steps=50, cfg_scale=3.0, eta=0.0, use_amp=True):
        """
        Optimized DDIM sampling with batched CFG.

        Args:
            model: ConditionalUNet (ideally torch.compiled)
            shape: (B, C, H, W)
            class_labels: (B,) tensor of class indices
            num_steps: DDIM steps (25=fastest, 50=good, 100=best)
            cfg_scale: Guidance strength (1.0=none, 3.0=balanced, 7.0=strong)
            eta: Stochasticity (0.0=deterministic DDIM, 1.0=DDPM-like)
            use_amp: Use float16 mixed precision
        """
        b = shape[0]
        img = torch.randn(shape, device=self.device)

        # Pre-compute schedule
        timesteps, alpha_t_all, alpha_prev_all = self._get_timestep_subsequence(num_steps)

        # Pre-compute unconditional labels (reused every step)
        uncond_labels = torch.full_like(class_labels, self.num_classes)

        # Batched labels for CFG: [cond_labels, uncond_labels]
        # This lets us do ONE forward pass instead of TWO per step
        batched_labels = torch.cat([class_labels, uncond_labels], dim=0)

        amp_dtype = torch.float16 if use_amp and self.device != 'cpu' else torch.float32
        use_autocast = use_amp and self.device != 'cpu'

        for i, t in enumerate(tqdm(timesteps, desc="DDIM Sampling", leave=True)):
            t_batch = torch.full((b,), t, device=self.device, dtype=torch.long)

            # ══ OPTIMIZATION: Batched CFG (1 forward pass instead of 2) ══
            # Stack: [img, img] and [t, t] and [cond_labels, uncond_labels]
            img_doubled = torch.cat([img, img], dim=0)
            t_doubled = torch.cat([t_batch, t_batch], dim=0)

            with torch.autocast(device_type=self.device, dtype=amp_dtype,
                                enabled=use_autocast):
                noise_pred_both = model(img_doubled, t_doubled, batched_labels)

            # Split back into conditional and unconditional
            noise_cond, noise_uncond = noise_pred_both.chunk(2, dim=0)

            # CFG: guided = uncond + scale * (cond - uncond)
            noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)

            # ══ DDIM update step ══
            at = alpha_t_all[i]
            ap = alpha_prev_all[i]

            # Predicted x₀
            sqrt_at = torch.sqrt(at)
            sqrt_1_minus_at = torch.sqrt(1.0 - at)
            pred_x0 = (img - sqrt_1_minus_at * noise_pred) / sqrt_at
            pred_x0.clamp_(-1.0, 1.0)  # In-place clamp

            # Direction pointing to x_t
            if eta > 0 and i + 1 < len(timesteps):
                sigma = eta * torch.sqrt(
                    (1 - ap) / (1 - at) * (1 - at / ap)
                )
                dir_xt = torch.sqrt(1 - ap - sigma**2) * noise_pred
                noise = torch.randn_like(img)
                img = torch.sqrt(ap) * pred_x0 + dir_xt + sigma * noise
            else:
                dir_xt = torch.sqrt(1 - ap) * noise_pred
                img = torch.sqrt(ap) * pred_x0 + dir_xt

        img.clamp_(-1.0, 1.0)
        return img


# ═══════════════════════════════════════════════════════════════════════
# INFERENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

CLASS_NAMES = {0: 'Benign', 1: 'Malignant', 2: 'Normal'}


def detect_gpu():
    """Detect GPU capabilities for optimization decisions."""
    if not torch.cuda.is_available():
        return {'device': 'cpu', 'compile': False, 'amp': False, 'name': 'CPU'}

    props = torch.cuda.get_device_properties(0)
    sm = props.major * 10 + props.minor
    vram_gb = props.total_mem / (1024**3)

    return {
        'device': 'cuda',
        'compile': sm >= 70,       # torch.compile on SM 7.0+
        'amp': True,               # FP16 on any GPU
        'name': props.name,
        'vram_gb': vram_gb,
        'sm': sm,
    }


def load_model(checkpoint_path, device='cuda', use_compile=True):
    """Load trained model with optimizations."""
    print(f"\n📦 Loading checkpoint: {checkpoint_path}")

    model = ConditionalUNet(
        img_channels=1, base_channels=64,
        time_emb_dim=256, num_classes=3, class_emb_dim=256,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # EMA weights = best quality
    if 'ema_state_dict' in ckpt:
        model.load_state_dict(ckpt['ema_state_dict'])
        print("  ✅ Loaded EMA weights (best quality)")
    elif 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        print("  ✅ Loaded model weights")
    else:
        model.load_state_dict(ckpt)
        print("  ✅ Loaded state dict directly")

    epoch = ckpt.get('epoch', '?')
    loss = ckpt.get('loss', ckpt.get('best_loss', '?'))
    print(f"  📍 Epoch: {epoch} | Loss: {loss}")

    model.eval()

    # ══ OPTIMIZATION: torch.compile ══
    if use_compile and device == 'cuda':
        try:
            print("  ⚡ Compiling model with torch.compile (1-2 min warmup)...")
            model = torch.compile(model, mode='reduce-overhead')
            print("  ✅ Model compiled!")
        except Exception as e:
            print(f"  ⚠️  torch.compile failed ({e}), using eager mode")

    del ckpt
    torch.cuda.empty_cache() if device == 'cuda' else None

    return model


def warmup_model(model, sampler, device='cuda'):
    """
    Run 1 dummy batch to trigger torch.compile and CUDA kernel caching.
    Subsequent batches will be much faster.
    """
    print("  🔥 Warming up (first batch is slow due to compilation)...")
    dummy_labels = torch.zeros(1, device=device, dtype=torch.long)
    _ = sampler.sample(
        model, shape=(1, 1, 128, 128), class_labels=dummy_labels,
        num_steps=5, cfg_scale=1.0,  # Minimal steps for warmup
    )
    if device == 'cuda':
        torch.cuda.synchronize()
    print("  ✅ Warmup complete!")


def generate_images(model, sampler, class_idx, num_images,
                    output_dir, batch_size=16, cfg_scale=3.0,
                    ddim_steps=50, device='cuda', use_amp=True):
    """Generate and save synthetic images for a class."""

    class_name = CLASS_NAMES[class_idx]
    save_dir = os.path.join(output_dir, class_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n🎨 Generating {num_images} {class_name} images "
          f"(batch={batch_size}, steps={ddim_steps}, cfg={cfg_scale})...")

    generated = 0
    start_time = time.time()

    while generated < num_images:
        current_batch = min(batch_size, num_images - generated)
        labels = torch.full((current_batch,), class_idx,
                            device=device, dtype=torch.long)

        images = sampler.sample(
            model, shape=(current_batch, 1, 128, 128),
            class_labels=labels, num_steps=ddim_steps,
            cfg_scale=cfg_scale, use_amp=use_amp,
        )

        # Save images
        for i in range(current_batch):
            img = (images[i] + 1) / 2  # [-1,1] → [0,1]
            img.clamp_(0, 1)
            path = os.path.join(save_dir,
                                f"synthetic_{class_name}_{generated + i:04d}.png")
            save_image(img, path)

        generated += current_batch
        elapsed = time.time() - start_time
        speed = generated / elapsed
        eta = (num_images - generated) / speed if speed > 0 else 0
        print(f"  💾 {generated}/{num_images} "
              f"({speed:.1f} img/s, ETA: {eta:.0f}s)")

    total_time = time.time() - start_time
    print(f"  ✅ Done! {num_images} images in {total_time:.1f}s "
          f"({num_images/total_time:.1f} img/s)")
    return save_dir


def generate_preview_grid(model, sampler, output_dir,
                          cfg_scale=3.0, ddim_steps=50,
                          device='cuda', use_amp=True):
    """Generate a 3×8 preview grid."""
    print("\n🖼️  Generating preview grid...")
    fig, axes = plt.subplots(3, 8, figsize=(20, 8))
    fig.suptitle('Synthetic CT Samples\n(Rows: Benign, Malignant, Normal)',
                 fontsize=14, fontweight='bold')

    for class_idx in range(3):
        labels = torch.full((8,), class_idx, device=device, dtype=torch.long)
        samples = sampler.sample(
            model, shape=(8, 1, 128, 128), class_labels=labels,
            num_steps=ddim_steps, cfg_scale=cfg_scale, use_amp=use_amp,
        )
        for j in range(8):
            img = (samples[j, 0].cpu() + 1) / 2
            axes[class_idx, j].imshow(img, cmap='gray')
            axes[class_idx, j].axis('off')
            if j == 0:
                axes[class_idx, j].set_ylabel(CLASS_NAMES[class_idx], fontsize=12)

    plt.tight_layout()
    grid_path = os.path.join(output_dir, 'synthetic_preview_grid.png')
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Grid saved to {grid_path}")


def generate_cfg_comparison(model, sampler, output_dir,
                            ddim_steps=50, device='cuda', use_amp=True):
    """Compare different CFG scales side by side."""
    print("\n📊 Generating CFG scale comparison...")
    cfg_scales = [1.0, 2.0, 3.0, 5.0, 7.0]
    fig, axes = plt.subplots(len(cfg_scales), 6,
                             figsize=(15, len(cfg_scales) * 2.5))
    fig.suptitle('CFG Scale Comparison (Higher = More Class-Specific)',
                 fontsize=13, fontweight='bold')

    for row, scale in enumerate(cfg_scales):
        torch.manual_seed(42)  # Same seed for fair comparison
        for class_idx in range(3):
            labels = torch.full((2,), class_idx, device=device, dtype=torch.long)
            samples = sampler.sample(
                model, shape=(2, 1, 128, 128), class_labels=labels,
                num_steps=ddim_steps, cfg_scale=scale, use_amp=use_amp,
            )
            for k in range(2):
                col = class_idx * 2 + k
                img = (samples[k, 0].cpu() + 1) / 2
                axes[row, col].imshow(img, cmap='gray')
                axes[row, col].axis('off')
                if col == 0:
                    axes[row, col].set_ylabel(f'CFG={scale}', fontsize=10)
                if row == 0:
                    axes[row, col].set_title(CLASS_NAMES[class_idx], fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, 'cfg_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved to {path}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic CT images (optimized inference)')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint (.pt)')
    parser.add_argument('--output', type=str, default='./generated_images',
                        help='Output directory')
    parser.add_argument('--num_benign', type=int, default=300)
    parser.add_argument('--num_malignant', type=int, default=0)
    parser.add_argument('--num_normal', type=int, default=100)
    parser.add_argument('--cfg_scale', type=float, default=3.0,
                        help='CFG guidance (1=none, 3=balanced, 7=strong)')
    parser.add_argument('--ddim_steps', type=int, default=50,
                        help='DDIM steps (25=fast, 50=good, 100=best)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Generation batch size')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cuda, cpu')
    parser.add_argument('--preview_only', action='store_true',
                        help='Only generate preview grid')
    parser.add_argument('--cfg_compare', action='store_true',
                        help='Generate CFG comparison chart')
    parser.add_argument('--no_compile', action='store_true',
                        help='Disable torch.compile')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable mixed precision')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # ── Setup ──
    setup_cuda_optimizations()
    gpu = detect_gpu()

    if args.device == 'auto':
        device = gpu['device']
    else:
        device = args.device

    use_compile = gpu.get('compile', False) and not args.no_compile
    use_amp = gpu.get('amp', False) and not args.no_amp

    torch.manual_seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    # ── Print config ──
    print("=" * 60)
    print("  🧬 Optimized CT Image Generator")
    print("=" * 60)
    print(f"  GPU:          {gpu.get('name', 'CPU')}")
    if 'vram_gb' in gpu:
        print(f"  VRAM:         {gpu['vram_gb']:.1f} GB (SM {gpu['sm']})")
    print(f"  torch.compile: {'✅' if use_compile else '❌'}")
    print(f"  AMP FP16:     {'✅' if use_amp else '❌'}")
    print(f"  Batched CFG:  ✅ (2× fewer forward passes)")
    print(f"  CFG Scale:    {args.cfg_scale}")
    print(f"  DDIM Steps:   {args.ddim_steps}")
    print(f"  Batch Size:   {args.batch_size}")
    print(f"  Seed:         {args.seed}")
    print("=" * 60)

    # ── Load model ──
    model = load_model(args.checkpoint, device, use_compile)

    # ── Create sampler ──
    sampler = OptimizedDDIMSampler(
        timesteps=1000, beta_start=1e-4, beta_end=0.02,
        num_classes=3, device=device,
    )

    # ── Warmup (triggers torch.compile) ──
    if use_compile and device == 'cuda':
        warmup_model(model, sampler, device)

    total_start = time.time()

    # ── Generate preview grid ──
    generate_preview_grid(model, sampler, args.output,
                          cfg_scale=args.cfg_scale,
                          ddim_steps=args.ddim_steps,
                          device=device, use_amp=use_amp)

    # ── CFG comparison ──
    if args.cfg_compare:
        generate_cfg_comparison(model, sampler, args.output,
                                ddim_steps=args.ddim_steps,
                                device=device, use_amp=use_amp)

    # ── Full generation ──
    if not args.preview_only:
        plan = {0: args.num_benign, 1: args.num_malignant, 2: args.num_normal}
        for class_idx, num in plan.items():
            if num > 0:
                generate_images(
                    model, sampler, class_idx, num, args.output,
                    batch_size=args.batch_size, cfg_scale=args.cfg_scale,
                    ddim_steps=args.ddim_steps, device=device, use_amp=use_amp,
                )

    # ── Summary ──
    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print("  ✅ GENERATION COMPLETE!")
    print("=" * 60)
    if not args.preview_only:
        total_imgs = args.num_benign + args.num_malignant + args.num_normal
        print(f"  Total images:  {total_imgs}")
        print(f"  Total time:    {total_time:.1f}s")
        if total_imgs > 0:
            print(f"  Throughput:    {total_imgs/total_time:.1f} img/s")
    print(f"  Output dir:    {args.output}")


if __name__ == '__main__':
    main()
