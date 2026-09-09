"""
=============================================================================
 Rectified Flow — Configuration
 Same infrastructure as DDPM (DDP, AMP, W&B) but with RF-specific parameters.
=============================================================================
"""

import os
import torch
import torch.distributed as dist


# ═══════════════════════════════════════════════════════════════════════
# DDP SETUP (identical to DDPM)
# ═══════════════════════════════════════════════════════════════════════

def setup_distributed():
    """Initialize DDP if launched with torchrun, else single-GPU."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    else:
        return 0, 0, 1


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


# Initialize
RANK, LOCAL_RANK, WORLD_SIZE = setup_distributed()
IS_DDP = WORLD_SIZE > 1


# ═══════════════════════════════════════════════════════════════════════
# GPU AUTO-DETECTION
# ═══════════════════════════════════════════════════════════════════════

def detect_gpu_config(local_rank=0):
    """Auto-detect GPU capabilities and set optimal precision + batch size."""
    if not torch.cuda.is_available():
        return {
            'device': torch.device('cpu'), 'precision': 'fp32',
            'amp_dtype': torch.float32, 'use_amp': False,
            'batch_size': 4, 'compile': False, 'gpu_name': 'CPU',
            'vram_gb': 0, 'compute_cap': '0.0', 'num_workers': 2,
        }

    gpu_name = torch.cuda.get_device_name(local_rank)
    vram_gb = torch.cuda.get_device_properties(local_rank).total_memory / 1024**3
    major, minor = torch.cuda.get_device_capability(local_rank)

    supports_bf16 = major >= 8
    supports_compile = major >= 7

    amp_dtype = torch.bfloat16 if supports_bf16 else torch.float16
    precision = 'bf16' if supports_bf16 else 'fp16'

    # Auto batch size — 224×224 uses ~3× more VRAM than 128×128
    if vram_gb >= 70:
        batch_size = 64
    elif vram_gb >= 35:
        batch_size = 32
    elif vram_gb >= 20:
        batch_size = 24
    elif vram_gb >= 14:
        batch_size = 12   # T4 16GB — 224×224 needs more VRAM than 128
    elif vram_gb >= 10:
        batch_size = 6
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


gpu_cfg = detect_gpu_config(local_rank=LOCAL_RANK)


# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

class Config:
    # ── Dataset ──
    DATASET_ROOT = "/kaggle/input/datasets/hamdallak/the-iqothnccd-lung-cancer-dataset/The IQ-OTHNCCD lung cancer dataset"
    OUTPUT_DIR = "/kaggle/working/rf_output"
    SYNTHETIC_DIR = "/kaggle/working/synthetic_images_rf"

    # ── Classes ──
    CLASSES = ["Benign cases", "Malignant cases", "Normal cases"]
    CLASS_TO_IDX = {"Benign cases": 0, "Malignant cases": 1, "Normal cases": 2}
    IDX_TO_CLASS = {0: "Benign", 1: "Malignant", 2: "Normal"}
    NUM_CLASSES = 3

    # ── Image ──
    IMG_SIZE = 224              # ← Upgraded from 128 for better quality
    IMG_CHANNELS = 1            # Grayscale CT scans

    # ── Rectified Flow (replaces DDPM beta schedule) ──
    # No beta_start / beta_end needed — RF uses t ∈ [0, 1] directly
    SIGMA_MIN = 1e-4            # Small offset to avoid division by zero at t=0
    TIMESTEP_SAMPLING = "logit_normal"   # "uniform" or "logit_normal" (SD3-style)
    LOGIT_NORMAL_MEAN = 0.0     # Center of logit-normal distribution
    LOGIT_NORMAL_STD = 1.0      # Spread — emphasizes mid-timesteps

    # ── Model (U-Net — same architecture as DDPM) ──
    BASE_CHANNELS = 64          # Channel progression: 64→128→256→512
    TIME_EMB_DIM = 256          # Timestep embedding dimension
    CLASS_EMB_DIM = 256         # Class embedding dimension

    # ── Training ──
    EPOCHS = 3000               # RF converges faster, but we keep same budget
    BATCH_SIZE = gpu_cfg['batch_size']
    LEARNING_RATE = 1.5e-4 * WORLD_SIZE
    EMA_DECAY = 0.9999
    SAVE_EVERY = 100
    SAMPLE_EVERY = 25

    # ── Mixed Precision ──
    USE_AMP = gpu_cfg['use_amp']
    AMP_DTYPE = gpu_cfg['amp_dtype']
    USE_COMPILE = gpu_cfg['compile']

    # ── Distributed ──
    WORLD_SIZE = WORLD_SIZE
    RANK = RANK
    LOCAL_RANK = LOCAL_RANK
    IS_DDP = IS_DDP

    # ── W&B ──
    WANDB_PROJECT = "ct-rectified-flow"
    WANDB_RUN_NAME = f"rf-{WORLD_SIZE}xGPU-ddp" if IS_DDP else "rf-1xGPU"
    USE_WANDB = True

    # ── Kaggle Time Guard ──
    MAX_TRAINING_HOURS = 10.0
    GENERATION_BUDGET_MIN = 30

    # ── Resume ──
    RESUME_CHECKPOINT = None

    # ── Early Stopping ──
    PATIENCE = 80
    MIN_DELTA = 0.005
    CONVERGENCE_LOSS = 0.010    # RF loss converges to lower values than DDPM

    # ── Generation ──
    NUM_SYNTHETIC_BENIGN = 300
    NUM_SYNTHETIC_NORMAL = 100
    EULER_STEPS = 30            # ODE solver steps (RF needs fewer than DDPM's 50)
    CFG_SCALE = 3.0             # Classifier-free guidance scale

    # ── Reproducibility ──
    SEED = 42

    # ── Device ──
    DEVICE = gpu_cfg['device']
    NUM_WORKERS = gpu_cfg['num_workers']


cfg = Config()
