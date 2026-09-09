"""
=============================================================================
 Rectified Flow — Training Script
 
 Usage:
   Multi-GPU:   torchrun --nproc_per_node=2 rectified_flow/train.py
   Single-GPU:  python rectified_flow/train.py
 
 All infrastructure (DDP, AMP, W&B, checkpointing, time guard) is identical
 to the DDPM script. The ONLY training difference is:
 
   DDPM training step:
     t ~ Uniform{0, ..., 999}
     x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
     loss = MSE(model(x_t, t, c), ε)
 
   RF training step:
     t ~ LogitNormal(0, 1)          ← continuous, emphasizes mid-timesteps
     x_t = (1-t) · x_0 + t · ε      ← straight line (not curved)
     v = ε - x_0                     ← velocity target
     loss = MSE(model(x_t, t, c), v) ← predict velocity, not noise
=============================================================================
"""

import os
import sys
import time
import random
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler
from sklearn.model_selection import train_test_split

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    import albumentations as A
    HAS_ALBUM = True
except ImportError:
    HAS_ALBUM = False

# DDP imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# ── Local imports ──
from config import cfg, gpu_cfg, is_main_process, cleanup_distributed, IS_DDP, RANK, LOCAL_RANK, WORLD_SIZE
from unet import ConditionalUNet
from flow_scheduler import RectifiedFlowScheduler
from ema import EMA

TRAINING_START_TIME = time.time()


# ═══════════════════════════════════════════════════════════════════════
# SEEDING & CUDA SETUP
# ═══════════════════════════════════════════════════════════════════════

def seed_everything(seed, rank=0):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(cfg.SEED, RANK)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Create output directories (rank 0 only)
if is_main_process():
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.makedirs(cfg.SYNTHETIC_DIR, exist_ok=True)
    for cls_name in cfg.IDX_TO_CLASS.values():
        os.makedirs(os.path.join(cfg.SYNTHETIC_DIR, cls_name), exist_ok=True)
if IS_DDP:
    dist.barrier()

if is_main_process():
    print(f"{'='*60}")
    print(f"RECTIFIED FLOW — CT Scan Augmentation")
    print(f"{'='*60}")
    print(f"  GPU:           {gpu_cfg['gpu_name']}")
    if torch.cuda.is_available():
        print(f"  VRAM:          {gpu_cfg['vram_gb']:.1f} GB")
    print(f"  Precision:     {gpu_cfg['precision'].upper()}")
    print(f"  Batch Size:    {cfg.BATCH_SIZE} per GPU × {WORLD_SIZE} = {cfg.BATCH_SIZE * WORLD_SIZE} effective")
    print(f"  Image Size:    {cfg.IMG_SIZE}×{cfg.IMG_SIZE}")
    print(f"  Timestep:      {cfg.TIMESTEP_SAMPLING} sampling")
    print(f"  DDP:           {'✅ ' + str(WORLD_SIZE) + ' GPUs' if IS_DDP else '❌ Single'}")
    print(f"  torch.compile: {'✅' if cfg.USE_COMPILE else '❌'}")
    print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════════════════
# DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════

def load_dataset_paths(root_dir):
    """Load all image paths and labels from the dataset directory."""
    image_paths, labels = [], []
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


class CTDataset(Dataset):
    """CT scan dataset with optional augmentation."""

    def __init__(self, image_paths, labels, img_size=224, augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.img_size = img_size
        self.augment = augment

        if HAS_ALBUM:
            if augment:
                self.transform = A.Compose([
                    A.Resize(img_size, img_size),
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=15, p=0.5, border_mode=0),
                    A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.3, border_mode=0),
                    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
                    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
                    A.GaussNoise(var_limit=(10, 50), p=0.2),
                ])
            else:
                self.transform = A.Compose([A.Resize(img_size, img_size)])
        else:
            self.transform = None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('L')
        img = np.array(img)

        if self.transform is not None:
            img = self.transform(image=img)['image']
        else:
            from torchvision.transforms.functional import resize
            img = np.array(resize(Image.fromarray(img), (self.img_size, self.img_size)))

        # Normalize to [-1, 1]
        img = img.astype(np.float32) / 255.0
        img = img * 2.0 - 1.0
        img = torch.FloatTensor(img).unsqueeze(0)  # (1, H, W)
        label = torch.LongTensor([self.labels[idx]]).squeeze()
        return img, label


# Load and split data
all_paths, all_labels = load_dataset_paths(cfg.DATASET_ROOT)
if is_main_process():
    print(f"\nTotal images: {len(all_paths)}")
    for idx, count in sorted(Counter(all_labels).items()):
        print(f"  {cfg.IDX_TO_CLASS[idx]}: {count}")

train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    all_paths, all_labels, test_size=0.30, stratify=all_labels, random_state=cfg.SEED
)
val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths, temp_labels, test_size=0.50, stratify=temp_labels, random_state=cfg.SEED
)

if is_main_process():
    print(f"Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")

# Create datasets and loaders
train_dataset = CTDataset(train_paths, train_labels, cfg.IMG_SIZE, augment=True)
val_dataset = CTDataset(val_paths, val_labels, cfg.IMG_SIZE, augment=False)

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


# ═══════════════════════════════════════════════════════════════════════
# TRAINING FUNCTION
# ═══════════════════════════════════════════════════════════════════════

def train():
    if is_main_process():
        print(f"\n{'='*60}")
        print(f"TRAINING — Rectified Flow")
        print(f"  Precision:     {gpu_cfg['precision'].upper()} AMP")
        print(f"  Batch:         {cfg.BATCH_SIZE} × {WORLD_SIZE} = {cfg.BATCH_SIZE * WORLD_SIZE}")
        print(f"  Resolution:    {cfg.IMG_SIZE}×{cfg.IMG_SIZE}")
        print(f"  Timestep:      {cfg.TIMESTEP_SAMPLING}")
        print(f"  Time Guard:    {cfg.MAX_TRAINING_HOURS}h")
        print(f"{'='*60}")

    # ── W&B Init ──
    if is_main_process() and cfg.USE_WANDB and HAS_WANDB:
        try:
            from kaggle_secrets import UserSecretsClient
            wandb.login(key=UserSecretsClient().get_secret("WANDB_API_KEY"))
        except Exception:
            wandb.login()

        wandb.init(
            project=cfg.WANDB_PROJECT,
            name=cfg.WANDB_RUN_NAME,
            config={
                "method": "Rectified Flow",
                "model": "ConditionalUNet",
                "img_size": cfg.IMG_SIZE,
                "epochs": cfg.EPOCHS,
                "batch_size_per_gpu": cfg.BATCH_SIZE,
                "effective_batch_size": cfg.BATCH_SIZE * WORLD_SIZE,
                "learning_rate": cfg.LEARNING_RATE,
                "precision": gpu_cfg['precision'],
                "gpu": gpu_cfg['gpu_name'],
                "num_gpus": WORLD_SIZE,
                "timestep_sampling": cfg.TIMESTEP_SAMPLING,
                "euler_steps": cfg.EULER_STEPS,
                "cfg_scale": cfg.CFG_SCALE,
                "ema_decay": cfg.EMA_DECAY,
            },
        )
        print("✅ W&B initialized")

    # ── Model ──
    model = ConditionalUNet(
        img_channels=cfg.IMG_CHANNELS,
        base_channels=cfg.BASE_CHANNELS,
        time_emb_dim=cfg.TIME_EMB_DIM,
        num_classes=cfg.NUM_CLASSES,
        class_emb_dim=cfg.CLASS_EMB_DIM,
    ).to(cfg.DEVICE)

    # torch.compile
    if cfg.USE_COMPILE:
        if is_main_process():
            print("Compiling model with torch.compile...")
        model = torch.compile(model, mode='reduce-overhead')
        if is_main_process():
            print("✅ Compiled")

    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model

    # DDP wrapping
    if IS_DDP:
        model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)
        if is_main_process():
            print(f"✅ DDP with {WORLD_SIZE} GPUs")

    total_params = sum(p.numel() for p in raw_model.parameters())
    if is_main_process():
        print(f"Parameters: {total_params:,} ({total_params / 1e6:.2f}M)")

    # ── Scheduler, Optimizer, EMA ──
    scheduler = RectifiedFlowScheduler(sigma_min=cfg.SIGMA_MIN, device=cfg.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)
    ema = EMA(raw_model, decay=cfg.EMA_DECAY)

    # AMP scaler (only for FP16, not BF16)
    use_scaler = cfg.USE_AMP and cfg.AMP_DTYPE == torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)

    # Training state
    train_losses = []
    best_loss = float('inf')
    stop_reason = "completed"
    start_epoch = 1
    p_uncond = 0.1  # CFG label dropout rate

    # ═══════════════════════════════════════════════════════════════
    # RESUME FROM CHECKPOINT
    # ═══════════════════════════════════════════════════════════════
    if cfg.RESUME_CHECKPOINT and os.path.exists(cfg.RESUME_CHECKPOINT):
        if is_main_process():
            print(f"\n🔄 Resuming from: {cfg.RESUME_CHECKPOINT}")
        ckpt = torch.load(cfg.RESUME_CHECKPOINT, map_location=cfg.DEVICE, weights_only=False)

        raw_model.load_state_dict(ckpt['model_state_dict'])
        if 'ema_state_dict' in ckpt:
            ema.load_state_dict(ckpt['ema_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            for pg in optimizer.param_groups:
                pg['lr'] = cfg.LEARNING_RATE
                pg['initial_lr'] = cfg.LEARNING_RATE
        if 'scaler_state_dict' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state_dict'])

        start_epoch = ckpt.get('epoch', 0) + 1
        best_loss = ckpt.get('best_loss', float('inf'))

        remaining = cfg.EPOCHS - start_epoch + 1
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining)

        if is_main_process():
            print(f"  Resumed at epoch {start_epoch}, best_loss={best_loss:.4f}")
        del ckpt
        torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════════
    # TRAINING LOOP
    # ═══════════════════════════════════════════════════════════════
    if is_main_process():
        print(f"\n🚀 Training for {cfg.EPOCHS - start_epoch + 1} epochs ({start_epoch}→{cfg.EPOCHS})")
        print(f"  AMP dtype:      {cfg.AMP_DTYPE}")
        print(f"  GradScaler:     {'ON' if use_scaler else 'OFF'}")
        print(f"  CFG dropout:    {p_uncond * 100}%")
        print(f"  Timestep dist:  {cfg.TIMESTEP_SAMPLING}")

    epoch_times = []
    patience_counter = 0

    for epoch in range(start_epoch, cfg.EPOCHS + 1):

        # ── Time Guard ──
        elapsed_h = (time.time() - TRAINING_START_TIME) / 3600
        remaining_h = cfg.MAX_TRAINING_HOURS - elapsed_h
        if remaining_h < (cfg.GENERATION_BUDGET_MIN / 60):
            stop_reason = f"time_guard ({elapsed_h:.1f}h)"
            if is_main_process():
                print(f"\n⏰ TIME GUARD: {elapsed_h:.1f}h elapsed. Stopping.")
            break

        model.train()
        epoch_loss = 0.0
        num_batches = 0
        epoch_start = time.time()

        if IS_DDP and hasattr(train_sampler, 'set_epoch'):
            train_sampler.set_epoch(epoch)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.EPOCHS}",
                    disable=not is_main_process())

        for batch_images, batch_labels in pbar:
            batch_images = batch_images.to(cfg.DEVICE, non_blocking=True)
            batch_labels = batch_labels.to(cfg.DEVICE, non_blocking=True)
            B = batch_images.shape[0]

            # ── CFG: randomly drop class labels ──
            mask = torch.rand(B, device=cfg.DEVICE) < p_uncond
            batch_labels_cfg = batch_labels.clone()
            batch_labels_cfg[mask] = cfg.NUM_CLASSES  # "unconditional" label

            # ══════════════════════════════════════════════════════
            # THIS IS THE CORE DIFFERENCE FROM DDPM
            # ══════════════════════════════════════════════════════

            # 1. Sample continuous timesteps t ∈ [0, 1]
            #    (DDPM uses discrete t ∈ {0, ..., 999})
            t = scheduler.sample_timesteps(
                B, method=cfg.TIMESTEP_SAMPLING,
                mean=cfg.LOGIT_NORMAL_MEAN, std=cfg.LOGIT_NORMAL_STD
            )

            # 2. Create noisy images via straight-line interpolation
            #    (DDPM uses x_t = √ᾱ·x0 + √(1-ᾱ)·ε)
            noise = torch.randn_like(batch_images)
            x_noisy, _, velocity_target = scheduler.add_noise(batch_images, t, noise)

            # 3. Predict velocity, compute loss
            #    (DDPM predicts noise: loss = MSE(ε̂, ε))
            with torch.amp.autocast('cuda', dtype=cfg.AMP_DTYPE, enabled=cfg.USE_AMP):
                v_pred = model(x_noisy, t, batch_labels_cfg)
                loss = F.mse_loss(v_pred, velocity_target)

            # ══════════════════════════════════════════════════════
            # END OF CORE DIFFERENCE — rest is identical to DDPM
            # ══════════════════════════════════════════════════════

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

        # ── Epoch Summary ──
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        avg_loss = epoch_loss / max(num_batches, 1)
        train_losses.append(avg_loss)
        lr_scheduler.step()

        elapsed_h = (time.time() - TRAINING_START_TIME) / 3600

        if is_main_process():
            eta_min = (cfg.EPOCHS - epoch) * np.mean(epoch_times) / 60
            print(f"Epoch {epoch}/{cfg.EPOCHS} | Loss: {avg_loss:.4f} | "
                  f"Best: {best_loss:.4f} | {epoch_time:.1f}s | "
                  f"Elapsed: {elapsed_h:.1f}h | ETA: {eta_min:.0f}m | "
                  f"LR: {lr_scheduler.get_last_lr()[0]:.6f}")

            # W&B logging
            if cfg.USE_WANDB and HAS_WANDB:
                log_dict = {
                    "train/loss": avg_loss,
                    "train/best_loss": min(best_loss, avg_loss),
                    "train/learning_rate": lr_scheduler.get_last_lr()[0],
                    "train/epoch_time_s": epoch_time,
                    "train/eta_minutes": eta_min,
                    "train/elapsed_hours": elapsed_h,
                    "train/epoch": epoch,
                }
                if torch.cuda.is_available():
                    log_dict["gpu/memory_allocated_gb"] = torch.cuda.memory_allocated() / 1024**3
                    log_dict["gpu/memory_reserved_gb"] = torch.cuda.memory_reserved() / 1024**3
                wandb.log(log_dict)

        # ── Save Best Model ──
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            if is_main_process():
                save_path = os.path.join(cfg.OUTPUT_DIR, "best_rf_model.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': raw_model.state_dict(),
                    'ema_state_dict': ema.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_loss': best_loss,
                    'train_losses': train_losses,
                    'config': {
                        'method': 'rectified_flow',
                        'img_size': cfg.IMG_SIZE,
                        'base_channels': cfg.BASE_CHANNELS,
                        'num_classes': cfg.NUM_CLASSES,
                        'timestep_sampling': cfg.TIMESTEP_SAMPLING,
                    },
                }, save_path)
                print(f"  ★ Saved best model (loss={best_loss:.4f})")
        else:
            patience_counter += 1

        # ── Periodic Checkpoint ──
        if epoch % cfg.SAVE_EVERY == 0 and is_main_process():
            ckpt_path = os.path.join(cfg.OUTPUT_DIR, f"checkpoint_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': raw_model.state_dict(),
                'ema_state_dict': ema.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_loss': best_loss,
                'train_losses': train_losses,
            }, ckpt_path)
            print(f"  💾 Checkpoint saved: epoch {epoch}")

        # ── Sample Images ──
        if epoch % cfg.SAMPLE_EVERY == 0 and is_main_process():
            _generate_samples(ema.shadow, scheduler, epoch)

        # ── Early Stopping ──
        if patience_counter >= cfg.PATIENCE:
            stop_reason = f"early_stop (patience={cfg.PATIENCE})"
            if is_main_process():
                print(f"\n⏹️ Early stopping at epoch {epoch}")
            break

        if IS_DDP:
            dist.barrier()

    # ── Final Save ──
    if is_main_process():
        final_path = os.path.join(cfg.OUTPUT_DIR, "final_checkpoint.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': raw_model.state_dict(),
            'ema_state_dict': ema.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_loss': best_loss,
            'train_losses': train_losses,
            'stop_reason': stop_reason,
        }, final_path)

        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE")
        print(f"  Stop reason:  {stop_reason}")
        print(f"  Final epoch:  {epoch}")
        print(f"  Best loss:    {best_loss:.4f}")
        print(f"  Total time:   {(time.time() - TRAINING_START_TIME) / 3600:.1f}h")
        print(f"  Best model:   {os.path.join(cfg.OUTPUT_DIR, 'best_rf_model.pt')}")
        print(f"{'='*60}")

        # Plot training loss
        _plot_loss(train_losses)

        if cfg.USE_WANDB and HAS_WANDB:
            wandb.finish()

    return ema, scheduler


def _generate_samples(model, scheduler, epoch, num_per_class=4):
    """Generate sample images for visual inspection during training."""
    model.eval()
    for class_idx, class_name in cfg.IDX_TO_CLASS.items():
        labels = torch.full((num_per_class,), class_idx, device=cfg.DEVICE, dtype=torch.long)
        shape = (num_per_class, cfg.IMG_CHANNELS, cfg.IMG_SIZE, cfg.IMG_SIZE)

        samples = scheduler.sample_euler(
            model, shape, labels,
            num_steps=cfg.EULER_STEPS,
            cfg_scale=cfg.CFG_SCALE,
            num_classes=cfg.NUM_CLASSES
        )

        # Save as grid
        samples = (samples.clamp(-1, 1) + 1) / 2  # [-1,1] → [0,1]
        import torchvision.utils as vutils
        grid = vutils.make_grid(samples, nrow=num_per_class, padding=2)
        grid_path = os.path.join(cfg.OUTPUT_DIR, f"samples_epoch{epoch}_{class_name}.png")
        vutils.save_image(grid, grid_path)

        if cfg.USE_WANDB and HAS_WANDB:
            wandb.log({f"samples/{class_name}": wandb.Image(grid_path)})

    model.train()


def _plot_loss(train_losses):
    """Plot and save training loss curve."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(train_losses, label='Training Loss', linewidth=1)
        ax.set(xlabel='Epoch', ylabel='MSE Loss (velocity)',
               title='Rectified Flow Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(cfg.OUTPUT_DIR, "training_loss.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()

        if cfg.USE_WANDB and HAS_WANDB:
            wandb.log({"loss_curve": wandb.Image(plot_path)})
    except ImportError:
        print("matplotlib not available — skipping loss plot")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        ema, scheduler = train()
    except Exception as e:
        if is_main_process():
            print(f"\n❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
    finally:
        cleanup_distributed()
