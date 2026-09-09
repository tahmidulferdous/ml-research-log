"""
=============================================================================
 Class-Conditional Diffusion Model (DDPM) for CT Scan Augmentation
 Dataset: IQ-OTH/NCCD Lung Cancer CT
=============================================================================

 DISTRIBUTED TRAINING (DDP):
 ────────────────────────────
 This script supports PyTorch DistributedDataParallel (DDP) for training
 across multiple GPUs on a single node. Automatically falls back to
 single-GPU when launched with plain `python`.

 HOW TO RUN:
 ───────────
 Single GPU:
   python conditional_diffusion_ct_augmentation.py

 Multi-GPU (DDP):
   torchrun --nproc_per_node=2 conditional_diffusion_ct_augmentation.py

 FEATURES:
 ─────────
 • DDP: Data-parallel training across multiple GPUs (NCCL backend)
 • torch.compile: Inductor backend for SM ≥ 7.0 (T4 ✅, ~10-20% speedup)
 • Mixed Precision: BF16 on Ampere+, FP16 with GradScaler on Turing/Volta
 • EMA: Exponential moving average for better generation quality
 • CFG: Classifier-Free Guidance with 10% unconditional dropout
 • DDIM: Fast 50-step sampling for generation

 GPU REQUIREMENTS:
 ┌──────────────────────────────────────────────────────────────────────────┐
 │  GPU              │ VRAM   │ Precision │ Batch │ Time/Epoch │  Total   │
 │───────────────────│────────│───────────│───────│────────────│──────────│
 │  T4 ×2 (DDP)     │ 2×16GB │ FP16      │ 2×16  │ ~1.3 min   │ ~4 hrs   │
 │  T4 (Free Kaggle) │ 16 GB  │ FP16      │ 16    │ ~2.5 min   │ ~8 hrs   │
 │  A100             │ 40 GB  │ BF16      │ 64    │ ~30 sec    │ ~1.5 hrs │
 │  H100             │ 80 GB  │ BF16      │ 128   │ ~15 sec    │ ~50 min  │
 └──────────────────────────────────────────────────────────────────────────┘
"""


import os
import math
import copy
import random
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from collections import Counter
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.utils import save_image, make_grid

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Kaggle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("⚠️  wandb not installed. Run: pip install wandb")

# Global training start time for Kaggle time guard
TRAINING_START_TIME = time.time()

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ═══════════════════════════════════════════════════════════════════════
# DDP DISTRIBUTED TRAINING SETUP
# ═══════════════════════════════════════════════════════════════════════

def setup_distributed():
    """
    Initialize DDP process group if launched with torchrun.
    Falls back to single-GPU if launched with plain `python`.

    Usage:
      Multi-GPU:  torchrun --nproc_per_node=2 script.py
      Single-GPU: python script.py
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

        return rank, local_rank, world_size
    else:
        # Single-GPU fallback
        return 0, 0, 1


def cleanup_distributed():
    """Destroy DDP process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if this is rank 0 (main process)."""
    return not dist.is_initialized() or dist.get_rank() == 0


# Initialize distributed training
RANK, LOCAL_RANK, WORLD_SIZE = setup_distributed()
IS_DDP = WORLD_SIZE > 1

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION — CHANGE THESE TO MATCH YOUR SETUP
# ═══════════════════════════════════════════════════════════════════════

def detect_gpu_config(local_rank=0):
    """
    Auto-detect GPU and set optimal precision + batch size.

    Precision hierarchy:
      BF16 — H100, A100, RTX 40xx (best: no gradient scaling needed)
      FP16 — T4, V100, RTX 30xx (good: needs GradScaler)
      FP32 — CPU fallback (slow)
    """
    if not torch.cuda.is_available():
        return {
            'device': torch.device('cpu'),
            'precision': 'fp32',
            'amp_dtype': None,
            'use_amp': False,
            'batch_size': 4,
            'compile': False,
            'gpu_name': 'CPU',
            'num_workers': 2,
        }

    gpu_name = torch.cuda.get_device_name(local_rank)
    vram_gb = torch.cuda.get_device_properties(local_rank).total_memory / 1024**3

    compute_cap = torch.cuda.get_device_capability(local_rank)
    major, minor = compute_cap

    # BF16 supported on compute capability >= 8.0 (Ampere+)
    supports_bf16 = major >= 8

    # torch.compile inductor backend supports SM >= 7.0 (T4 = 7.5 ✅)
    supports_compile = major >= 7

    if supports_bf16:
        amp_dtype = torch.bfloat16
        precision = 'bf16'
    else:
        amp_dtype = torch.float16
        precision = 'fp16'

    # Auto batch size based on VRAM (per-GPU)
    if vram_gb >= 70:
        batch_size = 128
    elif vram_gb >= 35:
        batch_size = 64
    elif vram_gb >= 20:
        batch_size = 64
    elif vram_gb >= 14:
        batch_size = 16
    elif vram_gb >= 10:
        batch_size = 8
    else:
        batch_size = 4

    return {
        'device': torch.device('cuda', local_rank),
        'precision': precision,
        'amp_dtype': amp_dtype,
        'use_amp': True,
        'batch_size': batch_size,
        'compile': supports_compile,
        'gpu_name': gpu_name,
        'vram_gb': vram_gb,
        'compute_cap': f"{major}.{minor}",
        'num_workers': 4 if vram_gb >= 35 else 2,
    }


# Auto-detect GPU capabilities (use LOCAL_RANK for correct GPU)
gpu_cfg = detect_gpu_config(local_rank=LOCAL_RANK)


class Config:
    # ── Dataset ──
    DATASET_ROOT = "/kaggle/input/datasets/hamdallak/the-iqothnccd-lung-cancer-dataset/The IQ-OTHNCCD lung cancer dataset"  # Change this!
    OUTPUT_DIR = "/kaggle/working/diffusion_output"
    SYNTHETIC_DIR = "/kaggle/working/synthetic_images"

    # ── Classes ──
    CLASSES = ["Benign cases", "Malignant cases", "Normal cases"]
    CLASS_TO_IDX = {"Benign cases": 0, "Malignant cases": 1, "Normal cases": 2}
    IDX_TO_CLASS = {0: "Benign", 1: "Malignant", 2: "Normal"}
    NUM_CLASSES = 3

    # ── Image ──
    IMG_SIZE = 128          # 128×128 (good balance of quality vs speed)
    IMG_CHANNELS = 1        # Grayscale CT scans

    # ── Diffusion ──
    TIMESTEPS = 1000        # T = 1000 (standard DDPM)
    BETA_START = 1e-4       # β₁
    BETA_END = 0.02         # βT

    # ── Model (U-Net) ──
    BASE_CHANNELS = 64      # Starting channels (64→128→256→512)
    TIME_EMB_DIM = 256      # Timestep embedding dimension
    CLASS_EMB_DIM = 256     # Class embedding dimension

    # ── Training (AUTO-CONFIGURED based on GPU) ──
    EPOCHS = 3000             # ~3h on 2×T4 (~3.5s/epoch × 3000)
    BATCH_SIZE = 24           # Per-GPU batch size (~11GB/GPU, fits in T4 14.6GB)
    LEARNING_RATE = 1.5e-4 * WORLD_SIZE   # Scaled for batch=24 per GPU
    EMA_DECAY = 0.9999      # Exponential moving average
    SAVE_EVERY = 100        # Save checkpoint every N epochs (overwrite to save disk)
    SAMPLE_EVERY = 25       # Generate samples every N epochs

    # ── Mixed Precision (AUTO-DETECTED) ──
    USE_AMP = gpu_cfg['use_amp']         # True for any GPU
    AMP_DTYPE = gpu_cfg['amp_dtype']     # BF16 on H100/A100, FP16 on T4/V100
    USE_COMPILE = gpu_cfg['compile']     # torch.compile on SM 7.0+ (T4 ✅)

    # ── Distributed ──
    WORLD_SIZE = WORLD_SIZE
    RANK = RANK
    LOCAL_RANK = LOCAL_RANK
    IS_DDP = IS_DDP

    # ── W&B (Weights & Biases) ──
    WANDB_PROJECT = "ct-diffusion-augmentation"
    WANDB_RUN_NAME = f"ddpm-{WORLD_SIZE}xT4-ddp" if IS_DDP else "ddpm-1xT4"
    USE_WANDB = HAS_WANDB  # Auto-enable if wandb is installed

    # ── Kaggle Time Guard ──
    MAX_TRAINING_HOURS = 10.0     # Stop training after 10 hrs (Kaggle limit = 11)
    GENERATION_BUDGET_MIN = 30    # Reserve 30 min for generation + quality

    # ── Resume Training ──
    # Set to checkpoint path to resume (e.g. '/kaggle/input/prev-run/checkpoint_epoch_725.pt')
    RESUME_CHECKPOINT = None      # Set path to resume from a previous checkpoint

    # ── Early Stopping (DISABLED — rely on time guard) ──
    PATIENCE = 80
    MIN_DELTA = 0.005
    CONVERGENCE_LOSS = 0.020

    # ── Generation ──
    NUM_SYNTHETIC_BENIGN = 300    # Generate this many benign images
    NUM_SYNTHETIC_NORMAL = 100    # Generate some normal too
    DDIM_STEPS = 50               # Use DDIM for faster sampling (50 vs 1000)

    # ── Reproducibility ──
    SEED = 42

    # ── Device ──
    DEVICE = gpu_cfg['device']
    NUM_WORKERS = gpu_cfg['num_workers']


cfg = Config()

# Seed everything (offset seed per rank for data diversity)
def seed_everything(seed, rank=0):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # Auto-tune convolution algorithms

seed_everything(cfg.SEED, RANK)

# CUDA optimizations
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Create output directories (rank 0 only to avoid race conditions)
if is_main_process():
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.makedirs(cfg.SYNTHETIC_DIR, exist_ok=True)
    for cls_name in cfg.IDX_TO_CLASS.values():
        os.makedirs(os.path.join(cfg.SYNTHETIC_DIR, cls_name), exist_ok=True)
if IS_DDP:
    dist.barrier()  # Wait for rank 0 to create directories

# Print config (rank 0 only)
if is_main_process():
    print(f"{'='*60}")
    print(f"GPU CONFIGURATION (Auto-Detected)")
    print(f"{'='*60}")
    print(f"  Device:        {gpu_cfg['gpu_name']}")
    if torch.cuda.is_available():
        print(f"  VRAM:          {gpu_cfg['vram_gb']:.1f} GB")
        print(f"  Compute Cap:   SM {gpu_cfg['compute_cap']}")
    print(f"  Precision:     {gpu_cfg['precision'].upper()} (Mixed Precision AMP)")
    print(f"  Batch Size:    {cfg.BATCH_SIZE} per GPU × {WORLD_SIZE} GPU(s) = {cfg.BATCH_SIZE * WORLD_SIZE} effective")
    print(f"  Learning Rate: {cfg.LEARNING_RATE} (scaled ×{WORLD_SIZE})")
    print(f"  torch.compile: {'✅ Enabled' if cfg.USE_COMPILE else '❌ Disabled'}")
    print(f"  DDP:           {'✅ Enabled (' + str(WORLD_SIZE) + ' GPUs)' if IS_DDP else '❌ Single GPU'}")
    if gpu_cfg['precision'] == 'bf16':
        print(f"  BF16 Speedup:  ~4× over FP32 (2000 TFLOPS on H100)")
        print(f"  Note:          No GradScaler needed (BF16 has large dynamic range)")
    else:
        print(f"  FP16 Speedup:  ~2× over FP32")
        print(f"  Note:          Using GradScaler for gradient stability")
    print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════════════════
# PART 1: DATA PREPARATION & STRATIFIED SPLITTING
# ═══════════════════════════════════════════════════════════════════════

if is_main_process():
    print("\n" + "="*60)
    print("PART 1: DATA PREPARATION")
    print("="*60)

def load_dataset_paths(root_dir):
    """Load all image paths and labels from the dataset directory."""
    image_paths = []
    labels = []

    for class_name in cfg.CLASSES:
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.exists(class_dir):
            if is_main_process():
                print(f"WARNING: {class_dir} not found!")
            continue

        for img_name in sorted(os.listdir(class_dir)):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                image_paths.append(os.path.join(class_dir, img_name))
                labels.append(cfg.CLASS_TO_IDX[class_name])

    return image_paths, labels


# Load all paths
all_paths, all_labels = load_dataset_paths(cfg.DATASET_ROOT)
if is_main_process():
    print(f"Total images found: {len(all_paths)}")
    print(f"Class distribution: {Counter(all_labels)}")
    for idx, count in sorted(Counter(all_labels).items()):
        print(f"  {cfg.IDX_TO_CLASS[idx]}: {count} images")

# ── Stratified Split: 70/15/15 ──
train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    all_paths, all_labels,
    test_size=0.30,
    stratify=all_labels,
    random_state=cfg.SEED
)

val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths, temp_labels,
    test_size=0.50,
    stratify=temp_labels,
    random_state=cfg.SEED
)

if is_main_process():
    print(f"\nTrain: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")
    print("Train distribution:", Counter(train_labels))
    print("Val distribution:  ", Counter(val_labels))
    print("Test distribution: ", Counter(test_labels))


# ═══════════════════════════════════════════════════════════════════════
# PART 2: DATASET CLASS WITH ALBUMENTATIONS
# ═══════════════════════════════════════════════════════════════════════

class CTDataset(Dataset):
    """CT scan dataset with Albumentations augmentation."""

    def __init__(self, image_paths, labels, img_size=128, augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.img_size = img_size
        self.augment = augment

        if augment:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5, border_mode=0),
                A.ElasticTransform(
                    alpha=120, sigma=120 * 0.05,
                    p=0.3, border_mode=0
                ),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1,
                    contrast_limit=0.1, p=0.3
                ),
                A.GaussNoise(var_limit=(10, 50), p=0.2),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img = Image.open(self.image_paths[idx]).convert('L')  # Grayscale
        img = np.array(img)

        # Apply albumentations
        augmented = self.transform(image=img)
        img = augmented['image']

        # Normalize to [-1, 1] (required for diffusion models)
        img = img.astype(np.float32) / 255.0  # [0, 1]
        img = img * 2.0 - 1.0                 # [-1, 1]

        # Add channel dimension: (H, W) → (1, H, W)
        img = torch.FloatTensor(img).unsqueeze(0)

        label = torch.LongTensor([self.labels[idx]]).squeeze()

        return img, label


# Create datasets
train_dataset = CTDataset(train_paths, train_labels, cfg.IMG_SIZE, augment=True)
val_dataset = CTDataset(val_paths, val_labels, cfg.IMG_SIZE, augment=False)

# ── Sampler setup: DistributedSampler for DDP, WeightedRandomSampler for single-GPU ──
if IS_DDP:
    train_sampler = DistributedSampler(train_dataset, num_replicas=WORLD_SIZE, rank=RANK, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=WORLD_SIZE, rank=RANK, shuffle=False)
else:
    class_counts = Counter(train_labels)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in train_labels]
    train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_labels), replacement=True)
    val_sampler = None

train_loader = DataLoader(
    train_dataset, batch_size=cfg.BATCH_SIZE,
    sampler=train_sampler, num_workers=cfg.NUM_WORKERS,
    pin_memory=True, drop_last=True, persistent_workers=True
)
val_loader = DataLoader(
    val_dataset, batch_size=cfg.BATCH_SIZE,
    sampler=val_sampler, shuffle=False,
    num_workers=cfg.NUM_WORKERS,
    pin_memory=True, persistent_workers=True
)

# Verify a batch (rank 0 only)
if is_main_process():
    sample_batch, sample_labels = next(iter(train_loader))
    print(f"\nBatch shape: {sample_batch.shape}")
    print(f"Label distribution in batch: {Counter(sample_labels.numpy().tolist())}")


# ═══════════════════════════════════════════════════════════════════════
# PART 3: DIFFUSION NOISE SCHEDULER
# ═══════════════════════════════════════════════════════════════════════

class DiffusionScheduler:
    """
    Manages the noise schedule for DDPM.

    Forward:  q(xₜ|x₀) = √(ᾱₜ)·x₀ + √(1-ᾱₜ)·ε
    Reverse:  pθ(xₜ₋₁|xₜ) = μθ(xₜ, t) + σₜ·z
    """

    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cuda'):
        self.timesteps = timesteps
        self.device = device

        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)

        # Pre-compute useful quantities
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # For q(xₜ|x₀) — adding noise
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # For p(xₜ₋₁|xₜ) — removing noise
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def add_noise(self, x0, t, noise=None):
        """Forward process: add noise to x0 at timestep t."""
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        return sqrt_alpha * x0 + sqrt_one_minus_alpha * noise, noise

    @torch.no_grad()
    def sample_ddpm(self, model, shape, class_labels, cfg_scale=3.0):
        """
        Full DDPM reverse sampling (T steps).
        Uses Classifier-Free Guidance for better class conditioning.
        """
        model.eval()
        b = shape[0]
        img = torch.randn(shape, device=self.device)

        for t in tqdm(reversed(range(self.timesteps)), total=self.timesteps, desc="DDPM Sampling"):
            t_batch = torch.full((b,), t, device=self.device, dtype=torch.long)

            # Classifier-Free Guidance
            # Conditional prediction
            noise_pred_cond = model(img, t_batch, class_labels)
            # Unconditional prediction (class = NUM_CLASSES = "no class")
            noise_pred_uncond = model(img, t_batch,
                                       torch.full_like(class_labels, cfg.NUM_CLASSES))

            # Guided prediction
            noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)

            # Compute xₜ₋₁
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            beta = self.betas[t]

            mean = self.sqrt_recip_alphas[t] * (
                img - beta / self.sqrt_one_minus_alphas_cumprod[t] * noise_pred
            )

            if t > 0:
                noise = torch.randn_like(img)
                sigma = torch.sqrt(self.posterior_variance[t])
                img = mean + sigma * noise
            else:
                img = mean

        # Clamp to [-1, 1]
        img = torch.clamp(img, -1.0, 1.0)
        return img

    @torch.no_grad()
    def sample_ddim(self, model, shape, class_labels, num_steps=50, cfg_scale=3.0, eta=0.0):
        """
        DDIM sampling — MUCH faster (50 steps instead of 1000).
        This is what you should use for generation.
        """
        model.eval()
        b = shape[0]
        img = torch.randn(shape, device=self.device)

        # Create sub-sequence of timesteps
        step_size = self.timesteps // num_steps
        timesteps = list(range(0, self.timesteps, step_size))
        timesteps = list(reversed(timesteps))

        for i, t in enumerate(tqdm(timesteps, desc="DDIM Sampling")):
            t_batch = torch.full((b,), t, device=self.device, dtype=torch.long)

            # Classifier-Free Guidance
            noise_pred_cond = model(img, t_batch, class_labels)
            noise_pred_uncond = model(img, t_batch,
                                       torch.full_like(class_labels, cfg.NUM_CLASSES))
            noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)

            # DDIM update
            alpha_cumprod_t = self.alphas_cumprod[t]

            if i + 1 < len(timesteps):
                t_prev = timesteps[i + 1]
                alpha_cumprod_prev = self.alphas_cumprod[t_prev]
            else:
                alpha_cumprod_prev = torch.tensor(1.0, device=self.device)

            # Predicted x₀
            pred_x0 = (img - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

            # Direction pointing to xₜ
            sigma = eta * torch.sqrt(
                (1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_prev)
            )
            dir_xt = torch.sqrt(1 - alpha_cumprod_prev - sigma**2) * noise_pred

            # xₜ₋₁
            if sigma > 0 and i + 1 < len(timesteps):
                noise = torch.randn_like(img)
                img = torch.sqrt(alpha_cumprod_prev) * pred_x0 + dir_xt + sigma * noise
            else:
                img = torch.sqrt(alpha_cumprod_prev) * pred_x0 + dir_xt

        img = torch.clamp(img, -1.0, 1.0)
        return img


# ═══════════════════════════════════════════════════════════════════════
# PART 4: U-NET MODEL (THE DENOISER εθ)
# ═══════════════════════════════════════════════════════════════════════

class SinusoidalPositionEmbedding(nn.Module):
    """Timestep embedding using sinusoidal functions (like Transformers)."""

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
    """Self-attention for spatial feature maps."""

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

        # Add time embedding
        h = h + self.time_mlp(t_emb)[:, :, None, None]

        # Add class embedding
        h = h + self.class_mlp(c_emb)[:, :, None, None]

        h = self.conv2(h)
        h = h + self.skip(x)

        # Self-attention
        h = self.attention(h)

        return h


class ConditionalUNet(nn.Module):
    """
    U-Net for noise prediction, conditioned on timestep t and class label c.

    Architecture:
      Encoder: 64 → 128 → 256 → 512
      Bottleneck: 512 (with self-attention)
      Decoder: 512 → 256 → 128 → 64
      Skip connections between encoder and decoder

    Parameters: ~25M (fits comfortably on T4 16GB)
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

        # num_classes + 1 for the "unconditional" class (used in CFG)
        self.class_embed = nn.Embedding(num_classes + 1, class_emb_dim)

        ch = base_channels  # 64

        # ── Encoder ──
        self.enc_conv0 = nn.Conv2d(img_channels, ch, 3, padding=1)

        self.enc1 = ResBlock(ch, ch, time_emb_dim, class_emb_dim)              # 64
        self.down1 = nn.Conv2d(ch, ch, 4, stride=2, padding=1)                 # /2

        self.enc2 = ResBlock(ch, ch * 2, time_emb_dim, class_emb_dim)          # 128
        self.down2 = nn.Conv2d(ch * 2, ch * 2, 4, stride=2, padding=1)        # /4

        self.enc3 = ResBlock(ch * 2, ch * 4, time_emb_dim, class_emb_dim,
                             use_attention=True)                                # 256
        self.down3 = nn.Conv2d(ch * 4, ch * 4, 4, stride=2, padding=1)        # /8

        # ── Bottleneck ──
        self.bot1 = ResBlock(ch * 4, ch * 8, time_emb_dim, class_emb_dim,
                             use_attention=True)                                # 512
        self.bot2 = ResBlock(ch * 8, ch * 8, time_emb_dim, class_emb_dim,
                             use_attention=True)
        self.bot3 = ResBlock(ch * 8, ch * 4, time_emb_dim, class_emb_dim)     # Back to 256

        # ── Decoder ──
        self.up3 = nn.ConvTranspose2d(ch * 4, ch * 4, 4, stride=2, padding=1)  # ×2
        self.dec3 = ResBlock(ch * 8, ch * 2, time_emb_dim, class_emb_dim,
                             use_attention=True)                                 # 128 (concat skip)

        self.up2 = nn.ConvTranspose2d(ch * 2, ch * 2, 4, stride=2, padding=1)  # ×2
        self.dec2 = ResBlock(ch * 4, ch, time_emb_dim, class_emb_dim)           # 64

        self.up1 = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)           # ×2
        self.dec1 = ResBlock(ch * 2, ch, time_emb_dim, class_emb_dim)           # 64

        # ── Output ──
        self.out = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, img_channels, 3, padding=1),
        )

    def forward(self, x, t, c):
        """
        x: (B, 1, H, W) — noisy image
        t: (B,) — timestep
        c: (B,) — class label (0=Benign, 1=Malignant, 2=Normal, 3=unconditional)
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
        d3 = torch.cat([d3, e3], dim=1)  # Skip connection
        d3 = self.dec3(d3, t_emb, c_emb)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2, t_emb, c_emb)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1, t_emb, c_emb)

        return self.out(d1)


# ═══════════════════════════════════════════════════════════════════════
# PART 5: EXPONENTIAL MOVING AVERAGE (EMA)
# ═══════════════════════════════════════════════════════════════════════

class EMA:
    """
    Exponential Moving Average of model parameters.
    Keeps a smoothed copy of the model for better generation quality.
    """

    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(self.decay).add_(m_param.data, alpha=1 - self.decay)

    def forward(self, *args, **kwargs):
        return self.shadow(*args, **kwargs)


# ═══════════════════════════════════════════════════════════════════════
# PART 6: TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════

def train():
    if is_main_process():
        print("\n" + "="*60)
        print("PART 6: TRAINING THE DIFFUSION MODEL")
        print(f"  Precision: {gpu_cfg['precision'].upper()} Mixed Precision")
        print(f"  Batch Size: {cfg.BATCH_SIZE} per GPU × {WORLD_SIZE} = {cfg.BATCH_SIZE * WORLD_SIZE} effective")
        print(f"  DDP: {'Enabled (' + str(WORLD_SIZE) + ' GPUs)' if IS_DDP else 'Single GPU'}")
        print(f"  Time Guard: {cfg.MAX_TRAINING_HOURS}h max (Kaggle 11h limit)")
        print(f"  Early Stop: patience={cfg.PATIENCE}, converge at loss<{cfg.CONVERGENCE_LOSS}")
        print("="*60)

    # ── W&B Init (rank 0 only) ──
    if is_main_process() and cfg.USE_WANDB:
        try:
            # Try to get API key from Kaggle secrets
            from kaggle_secrets import UserSecretsClient
            secrets = UserSecretsClient()
            wandb_key = secrets.get_secret("WANDB_API_KEY")
            wandb.login(key=wandb_key)
        except Exception:
            # Fall back to environment variable or cached login
            wandb.login()

        wandb.init(
            project=cfg.WANDB_PROJECT,
            name=cfg.WANDB_RUN_NAME,
            config={
                "model": "ConditionalUNet",
                "params": "~19M",
                "epochs": cfg.EPOCHS,
                "batch_size_per_gpu": cfg.BATCH_SIZE,
                "effective_batch_size": cfg.BATCH_SIZE * WORLD_SIZE,
                "learning_rate": cfg.LEARNING_RATE,
                "precision": gpu_cfg['precision'],
                "gpu": gpu_cfg['gpu_name'],
                "num_gpus": WORLD_SIZE,
                "ddp": IS_DDP,
                "torch_compile": cfg.USE_COMPILE,
                "img_size": cfg.IMG_SIZE,
                "timesteps": cfg.TIMESTEPS,
                "ema_decay": cfg.EMA_DECAY,
                "max_training_hours": cfg.MAX_TRAINING_HOURS,
                "patience": cfg.PATIENCE,
                "convergence_loss": cfg.CONVERGENCE_LOSS,
            },
        )
        print("✅ W&B initialized!")

    # Initialize model
    model = ConditionalUNet(
        img_channels=cfg.IMG_CHANNELS,
        base_channels=cfg.BASE_CHANNELS,
        time_emb_dim=cfg.TIME_EMB_DIM,
        num_classes=cfg.NUM_CLASSES,
        class_emb_dim=cfg.CLASS_EMB_DIM,
    ).to(cfg.DEVICE)

    # ── torch.compile ──
    if cfg.USE_COMPILE:
        if is_main_process():
            print("Compiling model with torch.compile (this takes ~1-2 min first time)...")
        model = torch.compile(model, mode='reduce-overhead')
        if is_main_process():
            print("✅ Model compiled!")

    # Get raw model reference (before DDP wrapping)
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model

    # ── DDP wrapping ──
    if IS_DDP:
        model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)
        if is_main_process():
            print(f"✅ Model wrapped with DDP ({WORLD_SIZE} GPUs)")

    # Count parameters
    total_params = sum(p.numel() for p in raw_model.parameters())
    trainable_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    if is_main_process():
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size: {total_params * 4 / 1024**2:.1f} MB (fp32 weights)")

    # Scheduler, optimizer, EMA
    scheduler = DiffusionScheduler(
        cfg.TIMESTEPS, cfg.BETA_START, cfg.BETA_END, cfg.DEVICE
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)
    ema = EMA(raw_model, decay=cfg.EMA_DECAY)

    # ── Mixed Precision Setup ──
    use_scaler = cfg.USE_AMP and cfg.AMP_DTYPE == torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)

    # Training history
    train_losses = []
    best_loss = float('inf')
    stop_reason = "completed"
    start_epoch = 1

    # ═══════════════════════════════════════════════════════════════
    # 🔄 RESUME FROM CHECKPOINT (if configured)
    # ═══════════════════════════════════════════════════════════════
    if cfg.RESUME_CHECKPOINT and os.path.exists(cfg.RESUME_CHECKPOINT):
        if is_main_process():
            print(f"\n🔄 Resuming from checkpoint: {cfg.RESUME_CHECKPOINT}")
        ckpt = torch.load(cfg.RESUME_CHECKPOINT, map_location=cfg.DEVICE, weights_only=False)

        # Load model weights
        raw_model.load_state_dict(ckpt['model_state_dict'])
        if is_main_process():
            print(f"  ✅ Model weights loaded")

        # Load EMA weights
        if 'ema_state_dict' in ckpt:
            ema.shadow.load_state_dict(ckpt['ema_state_dict'])
            if is_main_process():
                print(f"  ✅ EMA weights loaded")

         # Load optimizer state (but reset LR — old schedule is incompatible)
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            # IMPORTANT: Reset LR to current config (old checkpoint had different LR schedule)
            for param_group in optimizer.param_groups:
                param_group['lr'] = cfg.LEARNING_RATE
                param_group['initial_lr'] = cfg.LEARNING_RATE
            if is_main_process():
                print(f"  ✅ Optimizer state loaded (LR reset to {cfg.LEARNING_RATE})")

        # Load scaler state
        if 'scaler_state_dict' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
            if is_main_process():
                print(f"  ✅ GradScaler state loaded")

        # Resume epoch
        start_epoch = ckpt.get('epoch', 0) + 1
        best_loss = ckpt.get('best_loss', ckpt.get('loss', float('inf')))

        # Create FRESH LR scheduler for remaining epochs (don't inherit old schedule)
        remaining_epochs = cfg.EPOCHS - start_epoch + 1
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=remaining_epochs
        )

        if is_main_process():
            print(f"  📍 Resuming from epoch {start_epoch} (best_loss={best_loss:.4f})")
            print(f"  📍 LR: {cfg.LEARNING_RATE} → 0 over {remaining_epochs} epochs (fresh cosine)")
            print(f"  📍 Remaining epochs: {remaining_epochs}")

        del ckpt  # Free memory
        torch.cuda.empty_cache()

    # Classifier-Free Guidance: randomly drop class labels during training
    p_uncond = 0.1

    if is_main_process():
        print(f"\nStarting training for {cfg.EPOCHS - start_epoch + 1} epochs (epoch {start_epoch}→{cfg.EPOCHS})...")
        print(f"  AMP: {cfg.AMP_DTYPE}")
        print(f"  GradScaler: {'Enabled (FP16)' if use_scaler else 'Disabled (BF16)'}")
        print(f"  Classifier-Free Guidance dropout: {p_uncond*100}%")
        print(f"  W&B: {'✅ Logging' if cfg.USE_WANDB else '❌ Disabled'}")
        if start_epoch > 1:
            print(f"  🔄 RESUMED from epoch {start_epoch - 1} (best_loss={best_loss:.4f})")

    epoch_times = []
    final_epoch = 0

    for epoch in range(start_epoch, cfg.EPOCHS + 1):
        final_epoch = epoch

        # ════════════════════════════════════════════════════════════
        # ⏰ KAGGLE TIME GUARD — Check before each epoch
        # ════════════════════════════════════════════════════════════
        elapsed_hours = (time.time() - TRAINING_START_TIME) / 3600
        remaining_hours = cfg.MAX_TRAINING_HOURS - elapsed_hours

        if remaining_hours < (cfg.GENERATION_BUDGET_MIN / 60):
            stop_reason = f"time_guard ({elapsed_hours:.1f}h elapsed, {cfg.MAX_TRAINING_HOURS}h limit)"
            if is_main_process():
                print(f"\n⏰ TIME GUARD: {elapsed_hours:.1f}h elapsed. "
                      f"Stopping to reserve {cfg.GENERATION_BUDGET_MIN}min for generation.")
            break



        model.train()
        epoch_loss = 0.0
        num_batches = 0
        epoch_start = time.time()

        # Set epoch for DistributedSampler
        if IS_DDP and hasattr(train_sampler, 'set_epoch'):
            train_sampler.set_epoch(epoch)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.EPOCHS}",
                    disable=not is_main_process())
        for batch_images, batch_labels in pbar:
            batch_images = batch_images.to(cfg.DEVICE, non_blocking=True)
            batch_labels = batch_labels.to(cfg.DEVICE, non_blocking=True)

            # Classifier-Free Guidance: randomly drop class labels
            mask = torch.rand(batch_labels.shape[0], device=cfg.DEVICE) < p_uncond
            batch_labels_cfg = batch_labels.clone()
            batch_labels_cfg[mask] = cfg.NUM_CLASSES

            # Forward diffusion: add noise
            t = torch.randint(0, cfg.TIMESTEPS, (batch_images.shape[0],), device=cfg.DEVICE)
            noise = torch.randn_like(batch_images)
            x_noisy, _ = scheduler.add_noise(batch_images, t, noise)

            # Mixed Precision Forward Pass
            with torch.amp.autocast('cuda', dtype=cfg.AMP_DTYPE, enabled=cfg.USE_AMP):
                noise_pred = model(x_noisy, t, batch_labels_cfg)
                loss = F.mse_loss(noise_pred, noise)

            # Backprop
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(raw_model)

            epoch_loss += loss.item()
            num_batches += 1
            if is_main_process():
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # ── Epoch summary ──
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
        lr_scheduler.step()


        elapsed_hours = (time.time() - TRAINING_START_TIME) / 3600

        if is_main_process():
            eta_minutes = (cfg.EPOCHS - epoch) * np.mean(epoch_times) / 60
            print(f"Epoch {epoch}/{cfg.EPOCHS} | Loss: {avg_loss:.4f} | "
                  f"Best: {best_loss:.4f} | Time: {epoch_time:.1f}s | "
                  f"Elapsed: {elapsed_hours:.1f}h | ETA: {eta_minutes:.0f}m | "
                  f"LR: {lr_scheduler.get_last_lr()[0]:.6f}")

            # ── W&B Logging ──
            if cfg.USE_WANDB:
                log_dict = {
                    "train/loss": avg_loss,
                    "train/best_loss": min(best_loss, avg_loss),
                    "train/learning_rate": lr_scheduler.get_last_lr()[0],
                    "train/epoch_time_s": epoch_time,
                    "train/eta_minutes": eta_minutes,
                    "train/elapsed_hours": elapsed_hours,
                    "train/remaining_hours": cfg.MAX_TRAINING_HOURS - elapsed_hours,

                    "train/epoch": epoch,
                }
                # Log GPU memory
                if torch.cuda.is_available():
                    log_dict["gpu/memory_allocated_gb"] = torch.cuda.memory_allocated() / 1024**3
                    log_dict["gpu/memory_reserved_gb"] = torch.cuda.memory_reserved() / 1024**3
                wandb.log(log_dict, step=epoch)

        # ── Save best model (rank 0 only) ──
        if is_main_process() and avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': raw_model.state_dict(),
                'ema_state_dict': ema.shadow.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss': best_loss,
                'best_loss': best_loss,
                'precision': gpu_cfg['precision'],
            }, os.path.join(cfg.OUTPUT_DIR, 'best_diffusion_model.pt'))

        # ── Save periodic checkpoint (rank 0 only) ── OVERWRITE to save disk
        if is_main_process() and epoch % cfg.SAVE_EVERY == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': raw_model.state_dict(),
                'ema_state_dict': ema.shadow.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'loss': avg_loss,
                'best_loss': best_loss,
            }, os.path.join(cfg.OUTPUT_DIR, 'latest_checkpoint.pt'))

        # Synchronize all ranks
        if IS_DDP:
            dist.barrier()

        # ── Generate sample images + log to W&B (rank 0 only) ──
        if is_main_process() and epoch % cfg.SAMPLE_EVERY == 0:
            print(f"  Generating samples at epoch {epoch}...")
            ema.shadow.eval()

            with torch.amp.autocast('cuda', dtype=cfg.AMP_DTYPE, enabled=cfg.USE_AMP):
                wandb_images = []
                for class_idx in range(cfg.NUM_CLASSES):
                    labels = torch.full((4,), class_idx, device=cfg.DEVICE, dtype=torch.long)
                    samples = scheduler.sample_ddim(
                        ema.shadow,
                        shape=(4, cfg.IMG_CHANNELS, cfg.IMG_SIZE, cfg.IMG_SIZE),
                        class_labels=labels,
                        num_steps=cfg.DDIM_STEPS,
                        cfg_scale=3.0,
                    )

                    grid = make_grid((samples + 1) / 2, nrow=4)
                    save_path = os.path.join(cfg.OUTPUT_DIR,
                                             f'samples_epoch{epoch}_{cfg.IDX_TO_CLASS[class_idx]}.png')
                    save_image(grid, save_path)

                    # Log to W&B
                    if cfg.USE_WANDB:
                        # Convert grid to numpy for W&B
                        grid_np = grid.cpu().permute(1, 2, 0).numpy()
                        if grid_np.shape[2] == 1:
                            grid_np = grid_np.squeeze(2)
                        wandb_images.append(
                            wandb.Image(grid_np, caption=f"{cfg.IDX_TO_CLASS[class_idx]} (epoch {epoch})")
                        )

                if cfg.USE_WANDB and wandb_images:
                    wandb.log({"samples": wandb_images}, step=epoch)

        # Synchronize after sampling
        if IS_DDP:
            dist.barrier()

    # ═══════════════════════════════════════════════════════════════
    # TRAINING FINISHED — Save final state
    # ═══════════════════════════════════════════════════════════════
    if is_main_process():
        # Save final checkpoint
        torch.save({
            'epoch': final_epoch,
            'model_state_dict': raw_model.state_dict(),
            'ema_state_dict': ema.shadow.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'loss': train_losses[-1] if train_losses else float('inf'),
            'best_loss': best_loss,
            'stop_reason': stop_reason,
        }, os.path.join(cfg.OUTPUT_DIR, 'final_checkpoint.pt'))

        # Plot training loss
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(train_losses, label='Training Loss', color='#58a6ff')
        ax1.axhline(y=cfg.CONVERGENCE_LOSS, color='#3fb950', linestyle='--',
                    label=f'Convergence ({cfg.CONVERGENCE_LOSS})')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MSE Loss')
        ax1.set_title('Diffusion Model Training Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(epoch_times, label='Epoch Time (s)', color='#3fb950')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Time (seconds)')
        ax2.set_title(f'Epoch Time ({gpu_cfg["precision"].upper()} on {gpu_cfg["gpu_name"]})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        loss_plot_path = os.path.join(cfg.OUTPUT_DIR, 'training_loss.png')
        plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Log final plot to W&B
        if cfg.USE_WANDB:
            wandb.log({"training_loss_plot": wandb.Image(loss_plot_path)})

        total_time = sum(epoch_times)
        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"  Stop reason: {stop_reason}")
        print(f"  Epochs trained: {final_epoch}")
        print(f"  Best loss: {best_loss:.4f}")
        print(f"  Final loss: {train_losses[-1]:.4f}" if train_losses else "")
        print(f"  Total time: {total_time/60:.1f} minutes")
        print(f"  Avg epoch: {np.mean(epoch_times):.1f}s")
        print(f"  Precision: {gpu_cfg['precision'].upper()}")
        print(f"  DDP: {WORLD_SIZE} GPU(s)")
        print(f"{'='*60}")

        # W&B summary + alert
        if cfg.USE_WANDB:
            wandb.summary["best_loss"] = best_loss
            wandb.summary["final_loss"] = train_losses[-1] if train_losses else float('inf')
            wandb.summary["epochs_trained"] = final_epoch
            wandb.summary["total_time_minutes"] = total_time / 60
            wandb.summary["stop_reason"] = stop_reason

            # Send alert so you get notified!
            wandb.alert(
                title="🎉 Diffusion Training Complete!",
                text=(f"Stop: {stop_reason}\n"
                      f"Best loss: {best_loss:.4f}\n"
                      f"Epochs: {final_epoch}\n"
                      f"Time: {total_time/60:.1f} min"),
                level=wandb.AlertLevel.INFO,
            )

    return raw_model, ema, scheduler


# ═══════════════════════════════════════════════════════════════════════
# PART 7: GENERATE SYNTHETIC IMAGES
# ═══════════════════════════════════════════════════════════════════════

def generate_synthetic_images(ema_model, scheduler):
    """Generate synthetic CT images for underrepresented classes."""

    print("\n" + "="*60)
    print("PART 7: GENERATING SYNTHETIC IMAGES")
    print("="*60)

    generation_plan = {
        0: cfg.NUM_SYNTHETIC_BENIGN,   # Benign: generate 300
        2: cfg.NUM_SYNTHETIC_NORMAL,   # Normal: generate 100
    }

    ema_model.eval()

    for class_idx, num_to_generate in generation_plan.items():
        class_name = cfg.IDX_TO_CLASS[class_idx]
        os.makedirs(os.path.join(cfg.SYNTHETIC_DIR, class_name), exist_ok=True)
        print(f"\nGenerating {num_to_generate} synthetic {class_name} images...")

        generated_count = 0
        batch_size = min(16, num_to_generate)

        while generated_count < num_to_generate:
            current_batch = min(batch_size, num_to_generate - generated_count)

            labels = torch.full(
                (current_batch,), class_idx,
                device=cfg.DEVICE, dtype=torch.long
            )

            # Use DDIM for fast generation (50 steps vs 1000)
            synthetic_images = scheduler.sample_ddim(
                ema_model,
                shape=(current_batch, cfg.IMG_CHANNELS, cfg.IMG_SIZE, cfg.IMG_SIZE),
                class_labels=labels,
                num_steps=cfg.DDIM_STEPS,
                cfg_scale=3.0,
                eta=0.0,
            )

            # Save individual images
            for i in range(current_batch):
                img = (synthetic_images[i] + 1) / 2  # [-1,1] → [0,1]
                img = img.clamp(0, 1)
                save_path = os.path.join(
                    cfg.SYNTHETIC_DIR, class_name,
                    f"synthetic_{class_name}_{generated_count + i:04d}.png"
                )
                save_image(img, save_path)

            generated_count += current_batch
            print(f"  Generated {generated_count}/{num_to_generate}...")

        print(f"✅ Saved {num_to_generate} synthetic {class_name} images to "
              f"{os.path.join(cfg.SYNTHETIC_DIR, class_name)}")

    # ── Generate a quality preview grid ──
    print("\nGenerating quality preview grid...")
    fig, axes = plt.subplots(3, 8, figsize=(20, 8))
    fig.suptitle('Synthetic CT Samples (Rows: Benign, Malignant, Normal)', fontsize=14)

    for class_idx in range(3):
        labels = torch.full((8,), class_idx, device=cfg.DEVICE, dtype=torch.long)
        samples = scheduler.sample_ddim(
            ema_model,
            shape=(8, cfg.IMG_CHANNELS, cfg.IMG_SIZE, cfg.IMG_SIZE),
            class_labels=labels,
            num_steps=cfg.DDIM_STEPS,
            cfg_scale=3.0,
        )

        for j in range(8):
            img = (samples[j, 0].cpu() + 1) / 2
            axes[class_idx, j].imshow(img, cmap='gray')
            axes[class_idx, j].axis('off')
            if j == 0:
                axes[class_idx, j].set_ylabel(cfg.IDX_TO_CLASS[class_idx], fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, 'synthetic_preview_grid.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Preview grid saved!")


# ═══════════════════════════════════════════════════════════════════════
# PART 8: QUALITY ASSURANCE — FID SCORE
# ═══════════════════════════════════════════════════════════════════════

def compute_fid_features(dataloader, model_inception=None, device='cuda'):
    """Extract features for FID computation using a simple feature extractor."""
    # We use a simple approach: flatten and compute statistics
    # For production, use the `pytorch-fid` package with InceptionV3
    all_features = []

    for images, _ in tqdm(dataloader, desc="Extracting features"):
        images = images.to(device)
        # Simple feature: adaptive average pool to 8x8 then flatten
        feat = F.adaptive_avg_pool2d(images, (8, 8)).view(images.size(0), -1)
        all_features.append(feat.cpu().numpy())

    all_features = np.concatenate(all_features, axis=0)
    mu = np.mean(all_features, axis=0)
    sigma = np.cov(all_features, rowvar=False)
    return mu, sigma


def calculate_fid(mu1, sigma1, mu2, sigma2):
    """Calculate Fréchet Inception Distance."""
    from scipy.linalg import sqrtm

    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


def quality_assessment():
    """Run quality checks on generated synthetic images."""

    print("\n" + "="*60)
    print("PART 8: QUALITY ASSURANCE")
    print("="*60)

    # ── 1. Visual check: save a comparison grid ──
    print("\n[Gate 1] Visual Comparison — Real vs Synthetic")
    print("  → Check the saved grid: synthetic_preview_grid.png")
    print("  → Look for: anatomical plausibility, no checkerboard artifacts")

    # ── 2. FID Score ──
    print("\n[Gate 2] FID Score")

    # Load synthetic images
    synthetic_paths = []
    synthetic_labels = []
    for class_name in ["Benign", "Normal"]:
        class_dir = os.path.join(cfg.SYNTHETIC_DIR, class_name)
        if os.path.exists(class_dir):
            for img_name in os.listdir(class_dir):
                if img_name.endswith('.png'):
                    synthetic_paths.append(os.path.join(class_dir, img_name))
                    synthetic_labels.append(
                        cfg.CLASS_TO_IDX.get(f"{class_name} cases",
                                              {"Benign": 0, "Normal": 2}[class_name])
                    )

    if len(synthetic_paths) > 0:
        synth_dataset = CTDataset(synthetic_paths, synthetic_labels, cfg.IMG_SIZE, augment=False)
        synth_loader = DataLoader(synth_dataset, batch_size=32, shuffle=False)

        # Compute FID between real training data and synthetic data
        mu_real, sigma_real = compute_fid_features(train_loader, device=cfg.DEVICE)
        mu_synth, sigma_synth = compute_fid_features(synth_loader, device=cfg.DEVICE)

        fid_score = calculate_fid(mu_real, sigma_real, mu_synth, sigma_synth)
        print(f"  FID Score: {fid_score:.2f}")
        if fid_score < 50:
            print("  ✅ PASS — FID < 50 (good quality)")
        elif fid_score < 100:
            print("  ⚠️ MARGINAL — FID 50-100 (acceptable, could improve)")
        else:
            print("  ❌ FAIL — FID > 100 (poor quality, need more training)")

    # ── 3. t-SNE Diversity Check ──
    print("\n[Gate 3] t-SNE Diversity Visualization")

    all_features = []
    all_types = []  # 'real' or 'synthetic'
    all_class_labels = []

    # Real features
    for images, labels in train_loader:
        feat = F.adaptive_avg_pool2d(images, (8, 8)).view(images.size(0), -1)
        all_features.append(feat.numpy())
        all_types.extend(['real'] * images.size(0))
        all_class_labels.extend(labels.numpy().tolist())
        if len(all_features) > 10:
            break

    # Synthetic features
    if len(synthetic_paths) > 0:
        for images, labels in synth_loader:
            feat = F.adaptive_avg_pool2d(images, (8, 8)).view(images.size(0), -1)
            all_features.append(feat.numpy())
            all_types.extend(['synthetic'] * images.size(0))
            all_class_labels.extend(labels.numpy().tolist())

    all_features = np.concatenate(all_features, axis=0)

    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=cfg.SEED, perplexity=min(30, len(all_features) - 1))
    features_2d = tsne.fit_transform(all_features)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Plot 1: Real vs Synthetic
    colors = ['#58a6ff' if t == 'real' else '#f85149' for t in all_types]
    ax1.scatter(features_2d[:, 0], features_2d[:, 1], c=colors, alpha=0.5, s=10)
    ax1.set_title('t-SNE: Real (blue) vs Synthetic (red)')
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')

    # Plot 2: By class
    class_colors = {0: '#3fb950', 1: '#f85149', 2: '#58a6ff'}
    colors2 = [class_colors[c] for c in all_class_labels]
    ax2.scatter(features_2d[:, 0], features_2d[:, 1], c=colors2, alpha=0.5, s=10)
    ax2.set_title('t-SNE: Benign (green) / Malignant (red) / Normal (blue)')
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')

    plt.tight_layout()
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, 'tsne_quality_check.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  → t-SNE plot saved: tsne_quality_check.png")
    print("  → Synthetic points should OVERLAP with real points, not form separate clusters")


# ═══════════════════════════════════════════════════════════════════════
# PART 9: TRAIN CLASSIFIER ON AUGMENTED DATA
# ═══════════════════════════════════════════════════════════════════════

def train_classifier_comparison():
    """
    Train a simple classifier twice:
    1. WITHOUT synthetic data (baseline)
    2. WITH synthetic data (augmented)
    Compare results to validate the utility of synthetic generation.
    """

    print("\n" + "="*60)
    print("PART 9: CLASSIFIER COMPARISON (Baseline vs Augmented)")
    print("="*60)

    from torchvision import models

    class SimpleClassifier(nn.Module):
        """Lightweight classifier for comparison."""
        def __init__(self, num_classes=3):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(4),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256 * 4 * 4, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    def train_and_evaluate(train_loader, val_loader, name, epochs=30):
        """Train classifier and return metrics."""
        model = SimpleClassifier(cfg.NUM_CLASSES).to(cfg.DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Class-weighted loss
        class_counts = Counter(train_labels)
        total = sum(class_counts.values())
        weights = torch.FloatTensor([
            total / (cfg.NUM_CLASSES * class_counts.get(i, 1))
            for i in range(cfg.NUM_CLASSES)
        ]).to(cfg.DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weights)

        best_val_acc = 0
        for epoch in range(1, epochs + 1):
            # Train
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE)
                loss = criterion(model(images), labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validate
            model.eval()
            correct = 0
            total_samples = 0
            class_correct = {i: 0 for i in range(cfg.NUM_CLASSES)}
            class_total = {i: 0 for i in range(cfg.NUM_CLASSES)}

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE)
                    preds = model(images).argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total_samples += labels.size(0)

                    for i in range(cfg.NUM_CLASSES):
                        mask = labels == i
                        class_correct[i] += (preds[mask] == labels[mask]).sum().item()
                        class_total[i] += mask.sum().item()

            val_acc = correct / total_samples
            best_val_acc = max(best_val_acc, val_acc)

        # Per-class recall
        per_class_recall = {}
        for i in range(cfg.NUM_CLASSES):
            if class_total[i] > 0:
                per_class_recall[cfg.IDX_TO_CLASS[i]] = class_correct[i] / class_total[i]
            else:
                per_class_recall[cfg.IDX_TO_CLASS[i]] = 0.0

        print(f"\n  [{name}] Best Val Accuracy: {best_val_acc*100:.1f}%")
        for cls_name, recall in per_class_recall.items():
            print(f"    {cls_name} Recall: {recall*100:.1f}%")

        return best_val_acc, per_class_recall

    # ── Experiment 1: Baseline (no synthetic data) ──
    print("\n[Experiment 1] Baseline — Original data only")
    baseline_acc, baseline_recall = train_and_evaluate(
        train_loader, val_loader, "Baseline", epochs=30
    )

    # ── Experiment 2: With synthetic data ──
    print("\n[Experiment 2] Augmented — Original + Synthetic data")

    # Build augmented dataset
    aug_paths = list(train_paths)
    aug_labels = list(train_labels)

    for class_name in ["Benign", "Normal"]:
        synth_dir = os.path.join(cfg.SYNTHETIC_DIR, class_name)
        if os.path.exists(synth_dir):
            class_idx = {"Benign": 0, "Normal": 2}[class_name]
            for img_name in os.listdir(synth_dir):
                if img_name.endswith('.png'):
                    aug_paths.append(os.path.join(synth_dir, img_name))
                    aug_labels.append(class_idx)

    print(f"  Augmented dataset: {len(aug_paths)} images")
    print(f"  Class distribution: {Counter(aug_labels)}")

    aug_dataset = CTDataset(aug_paths, aug_labels, cfg.IMG_SIZE, augment=True)

    # Weighted sampler for augmented data
    aug_class_counts = Counter(aug_labels)
    aug_class_weights = {cls: 1.0 / count for cls, count in aug_class_counts.items()}
    aug_sample_weights = [aug_class_weights[label] for label in aug_labels]
    aug_sampler = WeightedRandomSampler(aug_sample_weights, len(aug_labels), replacement=True)

    aug_loader = DataLoader(
        aug_dataset, batch_size=cfg.BATCH_SIZE,
        sampler=aug_sampler, num_workers=2, pin_memory=True, drop_last=True
    )

    augmented_acc, augmented_recall = train_and_evaluate(
        aug_loader, val_loader, "Augmented", epochs=30
    )

    # ── Comparison ──
    print("\n" + "="*60)
    print("COMPARISON: Baseline vs Augmented")
    print("="*60)
    print(f"{'Metric':<25} {'Baseline':>10} {'Augmented':>10} {'Δ':>10}")
    print("-" * 58)
    print(f"{'Overall Accuracy':<25} {baseline_acc*100:>9.1f}% {augmented_acc*100:>9.1f}% "
          f"{(augmented_acc - baseline_acc)*100:>+9.1f}%")
    for cls_name in ["Benign", "Malignant", "Normal"]:
        b_recall = baseline_recall.get(cls_name, 0)
        a_recall = augmented_recall.get(cls_name, 0)
        print(f"{cls_name + ' Recall':<25} {b_recall*100:>9.1f}% {a_recall*100:>9.1f}% "
              f"{(a_recall - b_recall)*100:>+9.1f}%")


# ═══════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        if is_main_process():
            print("╔" + "═"*58 + "╗")
            print("║  Class-Conditional DDPM for CT Scan Augmentation          ║")
            print("║  Dataset: IQ-OTH/NCCD Lung Cancer                        ║")
            print(f"║  DDP: {WORLD_SIZE} GPU(s) | Compile: {'ON' if cfg.USE_COMPILE else 'OFF'} | W&B: {'ON' if cfg.USE_WANDB else 'OFF'}            ║")
            print(f"║  Time budget: {cfg.MAX_TRAINING_HOURS}h train + {cfg.GENERATION_BUDGET_MIN}min gen           ║")
            print("╚" + "═"*58 + "╝")

        # Step 1: Train the diffusion model (all ranks participate)
        model, ema, scheduler = train()

        # Steps 2-4: Only rank 0 does generation, evaluation, and comparison
        if is_main_process():
            # Check remaining time before generation
            elapsed_hours = (time.time() - TRAINING_START_TIME) / 3600
            remaining_min = (cfg.MAX_TRAINING_HOURS + (cfg.GENERATION_BUDGET_MIN / 60) - elapsed_hours) * 60

            if remaining_min > 5:  # Need at least 5 min for generation
                # Step 2: Generate synthetic images
                generate_synthetic_images(ema.shadow, scheduler)

                # Step 3: Quality assessment
                quality_assessment()

                # Step 4: Train & compare classifiers (skip if < 10 min left)
                if remaining_min > 15:
                    train_classifier_comparison()
                else:
                    print("\n⏰ Skipping classifier comparison (not enough time)")
            else:
                print("\n⏰ Skipping generation (not enough time remaining)")

            print("\n" + "="*60)
            print("ALL DONE! 🎉")
            print("="*60)
            print(f"\nOutputs saved to:")
            print(f"  Checkpoints: {cfg.OUTPUT_DIR}/")
            print(f"  Synthetic images: {cfg.SYNTHETIC_DIR}/")
            print(f"  Total elapsed: {elapsed_hours:.1f} hours")

            if cfg.USE_WANDB:
                print(f"\n📊 View results: https://wandb.ai — project '{cfg.WANDB_PROJECT}'")

    except KeyboardInterrupt:
        if is_main_process():
            print("\n⚠️  Training interrupted by user.")
    finally:
        # Close W&B
        if is_main_process() and cfg.USE_WANDB:
            try:
                wandb.finish()
            except Exception:
                pass
        cleanup_distributed()
