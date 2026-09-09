"""
=============================================================================
Class-Conditional Rectified Flow for CT Scan Augmentation — JAX/Flax/TPU
=============================================================================
Single file. All config at the top. Edit the Config class, run the script.
Native JAX for Kaggle TPU v5e-8 using jax.pmap on all 8 cores.
Architecture matches GPU EXACTLY.
=============================================================================
"""

import os, sys, math, time, functools, json
from pathlib import Path
from collections import defaultdict
from typing import Any

import numpy as np
from PIL import Image

import jax
import jax.numpy as jnp
from jax import random as jrandom
from jax import lax

import flax.linen as nn
from flax.training import train_state, checkpoints
from flax import jax_utils
import optax

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

TRAINING_START_TIME = time.time()

# ── Full reproducibility ──
# Note: uses 42 here because cfg is not loaded yet.
# Change this if you change cfg.SEED.
_SEED = 42
import random
random.seed(_SEED)
np.random.seed(_SEED)
os.environ['PYTHONHASHSEED'] = str(_SEED)

print(f"JAX devices: {jax.devices()}")
print(f"Device count: {jax.device_count()}")
NUM_DEVICES = jax.local_device_count()


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  TWEAK EVERYTHING HERE — single config section                      ║
# ╚═══════════════════════════════════════════════════════════════════════╝

class Config:
    # ── Paths ──
    DATASET_ROOT     = "/kaggle/input/datasets/hamdallak/the-iqothnccd-lung-cancer-dataset/The IQ-OTHNCCD lung cancer dataset"
    OUTPUT_DIR       = "/kaggle/working/rf_output"
    SYNTHETIC_DIR    = "/kaggle/working/synthetic_images_rf"
    RESUME_CHECKPOINT = None       # set to path to resume from JAX checkpoint
    PT_CHECKPOINT    = None        # PyTorch .pt path (info only, not loadable)

    # ── Dataset ──
    CLASSES      = ["Bengin cases", "Malignant cases", "Normal cases"]
    CLASS_TO_IDX = {"Bengin cases": 0, "Malignant cases": 1, "Normal cases": 2}
    IDX_TO_CLASS = {0: "Benign", 1: "Malignant", 2: "Normal"}
    NUM_CLASSES  = 3
    TRAIN_RATIO  = 0.70
    VAL_RATIO    = 0.15

    # ── Image ──
    IMG_SIZE     = 224
    IMG_CHANNELS = 1

    # ── Augmentation (offline: applied once at data loading) ──
    AUG_ENABLED           = True
    AUG_HORIZONTAL_FLIP_P = 0.5
    AUG_ROTATION_LIMIT    = 15
    AUG_ROTATION_P        = 0.5
    AUG_ELASTIC_ALPHA     = 120
    AUG_ELASTIC_SIGMA_R   = 0.05
    AUG_ELASTIC_P         = 0.3
    AUG_CLAHE_CLIP        = 2.0
    AUG_CLAHE_GRID        = (8, 8)
    AUG_CLAHE_P           = 0.3
    AUG_BRIGHTNESS_LIMIT  = 0.1
    AUG_CONTRAST_LIMIT    = 0.1
    AUG_BRIGHT_CONTRAST_P = 0.3
    AUG_GAUSS_NOISE_P     = 0.2

    # ── Online augmentation (inside train_step, random per batch) ──
    ONLINE_FLIP_ENABLED   = True
    ONLINE_FLIP_P         = 0.5

    # ── Model ──
    BASE_CHANNELS  = 128           # channel progression: ch → 2ch → 4ch → 8ch
    TIME_EMB_DIM   = 256
    CLASS_EMB_DIM  = 256
    NUM_GROUPS     = 8             # GroupNorm groups
    DROPOUT_RATE   = 0.1
    USE_DROPOUT    = False         # Set True to enable dropout (requires dropout RNG)

    # ── Rectified Flow ──
    SIGMA_MIN           = 1e-4
    TIMESTEP_SAMPLING   = "logit_normal"   # "logit_normal" or "uniform"
    LOGIT_NORMAL_MEAN   = 0.0
    LOGIT_NORMAL_STD    = 1.0
    CFG_DROPOUT_RATE    = 0.1              # probability of unconditional training

    # ── Training ──
    EPOCHS           = 3000
    PER_DEVICE_BATCH = 3                   # × 8 devices = 24 effective
    LEARNING_RATE    = 1.5e-4
    WEIGHT_DECAY     = 1e-4
    GRAD_CLIP_NORM   = 1.0
    EMA_DECAY        = 0.9999
    LR_SCHEDULE      = "cosine_decay"      # "cosine_decay", "warmup_cosine", "constant"
    LR_WARMUP_STEPS  = 0                   # only used if LR_SCHEDULE = "warmup_cosine"
    SEED             = 42

    # ── Checkpointing & Logging ──
    SAVE_EVERY          = 100              # save checkpoint every N epochs
    SAMPLE_EVERY        = 100              # generate samples every N epochs
    NUM_SAMPLES_PER_CLASS = 4              # images per class when sampling
    LOG_EVERY           = 1                # print loss every N epochs

    # ── Generation ──
    EULER_STEPS          = 50
    CFG_SCALE            = 4.0
    NUM_SYNTHETIC_BENIGN = 300
    NUM_SYNTHETIC_MALIGNANT = 0
    NUM_SYNTHETIC_NORMAL = 100
    GEN_BATCH_SIZE       = 16

    # ── Runtime ──
    MAX_TRAINING_HOURS    = 7.5
    GENERATION_BUDGET_MIN = 30
    USE_WANDB             = True
    WANDB_PROJECT         = "ct-rectified-flow"
    WANDB_RUN_NAME        = f"rf-JAX-{NUM_DEVICES}xTPUv5e"

    # ── Derived (auto-calculated) ──
    EFFECTIVE_BATCH = PER_DEVICE_BATCH * NUM_DEVICES

cfg = Config()


# ═══════════════════════════════════════════════════════════════════════
# BEST-LOSS PERSISTENCE (JSON sidecar)
# ═══════════════════════════════════════════════════════════════════════

def _meta_path():
    return os.path.join(cfg.OUTPUT_DIR, "training_meta.json")


def save_training_meta(best_loss, epoch):
    with open(_meta_path(), 'w') as f:
        json.dump({"best_loss": best_loss, "epoch": epoch}, f)


def load_training_meta():
    p = _meta_path()
    if os.path.exists(p):
        with open(p) as f:
            d = json.load(f)
        return d.get("best_loss", float('inf')), d.get("epoch", 0)
    return float('inf'), 0


# ═══════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════

def load_dataset_paths(root_dir):
    image_paths, labels = [], []
    for class_name in cfg.CLASSES:
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.exists(class_dir):
            continue
        for img_name in sorted(os.listdir(class_dir)):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                image_paths.append(os.path.join(class_dir, img_name))
                labels.append(cfg.CLASS_TO_IDX[class_name])
    return image_paths, labels


def sequential_split(paths, labels):
    """Sequential split 70/15/15 — preserves original sorted order
    to keep the same train/val/test partition across runs."""
    class_files = defaultdict(list)
    for p, l in zip(paths, labels):
        class_files[l].append(p)
    train_p, train_l, val_p, val_l, test_p, test_l = [], [], [], [], [], []
    for cls_idx, cls_paths in sorted(class_files.items()):
        n = len(cls_paths)
        n_train = int(n * cfg.TRAIN_RATIO)
        n_val = int(n * cfg.VAL_RATIO)
        train_p.extend(cls_paths[:n_train]); train_l.extend([cls_idx] * n_train)
        val_p.extend(cls_paths[n_train:n_train + n_val]); val_l.extend([cls_idx] * n_val)
        test_p.extend(cls_paths[n_train + n_val:]); test_l.extend([cls_idx] * (n - n_train - n_val))
    return train_p, val_p, test_p, train_l, val_l, test_l


def build_augmentation():
    if not HAS_ALBUM:
        return None
    if not cfg.AUG_ENABLED:
        return A.Compose([A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE)])
    transforms = [A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE)]
    if cfg.AUG_HORIZONTAL_FLIP_P > 0:
        transforms.append(A.HorizontalFlip(p=cfg.AUG_HORIZONTAL_FLIP_P))
    if cfg.AUG_ROTATION_P > 0:
        transforms.append(A.Rotate(limit=cfg.AUG_ROTATION_LIMIT, p=cfg.AUG_ROTATION_P, border_mode=0))
    if cfg.AUG_ELASTIC_P > 0:
        transforms.append(A.ElasticTransform(
            alpha=cfg.AUG_ELASTIC_ALPHA,
            sigma=cfg.AUG_ELASTIC_ALPHA * cfg.AUG_ELASTIC_SIGMA_R,
            p=cfg.AUG_ELASTIC_P, border_mode=0))
    if cfg.AUG_CLAHE_P > 0:
        transforms.append(A.CLAHE(clip_limit=cfg.AUG_CLAHE_CLIP,
                                  tile_grid_size=cfg.AUG_CLAHE_GRID, p=cfg.AUG_CLAHE_P))
    if cfg.AUG_BRIGHT_CONTRAST_P > 0:
        transforms.append(A.RandomBrightnessContrast(
            brightness_limit=cfg.AUG_BRIGHTNESS_LIMIT,
            contrast_limit=cfg.AUG_CONTRAST_LIMIT, p=cfg.AUG_BRIGHT_CONTRAST_P))
    if cfg.AUG_GAUSS_NOISE_P > 0:
        transforms.append(A.GaussNoise(p=cfg.AUG_GAUSS_NOISE_P))
    return A.Compose(transforms)


def load_images_to_memory(paths, labels):
    transform = build_augmentation()
    images, labs = [], []
    for p, l in zip(paths, labels):
        img = Image.open(p).convert('L')
        img = np.array(img)
        if transform is not None:
            img = transform(image=img)['image']
        else:
            img = np.array(Image.fromarray(img).resize((cfg.IMG_SIZE, cfg.IMG_SIZE)))
        img = img.astype(np.float32) / 255.0 * 2.0 - 1.0
        images.append(img[..., np.newaxis])
        labs.append(l)
    return np.array(images, dtype=np.float32), np.array(labs, dtype=np.int32)


def make_batches(images, labels, per_device_batch, rng, num_devices):
    """Shuffle and reshape into (n_batches, num_devices, per_device_batch, ...)."""
    total_batch = per_device_batch * num_devices
    n = len(images)
    perm = rng.permutation(n)
    images = images[perm]
    labels = labels[perm]
    n_batches = n // total_batch
    if n_batches == 0:
        n_batches = 1
        pad_n = total_batch - n
        images = np.concatenate([images, images[:pad_n]])
        labels = np.concatenate([labels, labels[:pad_n]])
    images = images[:n_batches * total_batch].reshape(
        n_batches, num_devices, per_device_batch, *images.shape[1:])
    labels = labels[:n_batches * total_batch].reshape(
        n_batches, num_devices, per_device_batch)
    return images, labels, n_batches


# ═══════════════════════════════════════════════════════════════════════
# U-NET MODEL — exact match of GPU architecture
# ═══════════════════════════════════════════════════════════════════════

class SinusoidalPosEmbed(nn.Module):
    dim: int
    @nn.compact
    def __call__(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        freqs = jnp.exp(jnp.arange(half, dtype=jnp.float32) * -emb)
        args = t[:, None].astype(jnp.float32) * 1000.0 * freqs[None, :]
        return jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)


class SelfAttention(nn.Module):
    channels: int
    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        h = nn.GroupNorm(num_groups=cfg.NUM_GROUPS)(x)
        q = nn.Conv(C, (1, 1), use_bias=True)(h)
        k = nn.Conv(C, (1, 1), use_bias=True)(h)
        v = nn.Conv(C, (1, 1), use_bias=True)(h)
        # Reshape to (B, H*W, C) for attention
        q = q.reshape(B, H * W, C)
        k = k.reshape(B, H * W, C)
        v = v.reshape(B, H * W, C)
        # FIX #12: Use JAX's efficient dot_product_attention (auto flash-attn on TPU)
        out = jax.nn.dot_product_attention(q, k, v)
        out = out.reshape(B, H, W, C)
        out = nn.Conv(C, (1, 1), use_bias=True)(out)
        return x + out


class ResBlock(nn.Module):
    in_ch: int
    out_ch: int
    time_emb_dim: int
    class_emb_dim: int
    use_attention: bool = False

    @nn.compact
    def __call__(self, x, t_emb, c_emb, deterministic: bool = True):
        h = nn.GroupNorm(num_groups=cfg.NUM_GROUPS)(x)
        h = nn.silu(h)
        h = nn.Conv(self.out_ch, (3, 3), padding='SAME')(h)
        h = h + nn.Dense(self.out_ch)(nn.silu(t_emb))[:, None, None, :]
        h = h + nn.Dense(self.out_ch)(nn.silu(c_emb))[:, None, None, :]
        h = nn.GroupNorm(num_groups=cfg.NUM_GROUPS)(h)
        h = nn.silu(h)
        # FIX #5: Use deterministic arg properly so dropout works when enabled
        h = nn.Dropout(cfg.DROPOUT_RATE, deterministic=deterministic)(h)
        h = nn.Conv(self.out_ch, (3, 3), padding='SAME')(h)
        if self.in_ch != self.out_ch:
            skip = nn.Conv(self.out_ch, (1, 1))(x)
        else:
            skip = x
        h = h + skip
        if self.use_attention:
            h = SelfAttention(self.out_ch)(h)
        return h


class ConditionalUNet(nn.Module):
    img_channels: int = 1
    base_channels: int = 128
    time_emb_dim: int = 256
    num_classes: int = 3
    class_emb_dim: int = 256

    @nn.compact
    def __call__(self, x, t, c, deterministic: bool = True):
        ch = self.base_channels
        te = self.time_emb_dim
        ce = self.class_emb_dim

        # Time embedding
        t_emb = SinusoidalPosEmbed(te)(t)
        t_emb = nn.Dense(te)(t_emb)
        t_emb = nn.silu(t_emb)
        t_emb = nn.Dense(te)(t_emb)

        # Class embedding
        c_emb = nn.Embed(self.num_classes + 1, ce)(c)

        # ══════════════ ENCODER ══════════════
        x0 = nn.Conv(ch, (3, 3), padding='SAME')(x)
        e1 = ResBlock(ch, ch, te, ce)(x0, t_emb, c_emb, deterministic)
        d1 = nn.Conv(ch, (4, 4), strides=(2, 2), padding='SAME')(e1)

        e2 = ResBlock(ch, ch * 2, te, ce)(d1, t_emb, c_emb, deterministic)
        d2 = nn.Conv(ch * 2, (4, 4), strides=(2, 2), padding='SAME')(e2)

        e3 = ResBlock(ch * 2, ch * 4, te, ce, use_attention=True)(d2, t_emb, c_emb, deterministic)
        d3 = nn.Conv(ch * 4, (4, 4), strides=(2, 2), padding='SAME')(e3)

        # ══════════════ BOTTLENECK ══════════════
        b = ResBlock(ch * 4, ch * 8, te, ce, use_attention=True)(d3, t_emb, c_emb, deterministic)
        b = ResBlock(ch * 8, ch * 8, te, ce, use_attention=True)(b, t_emb, c_emb, deterministic)
        b = ResBlock(ch * 8, ch * 4, te, ce)(b, t_emb, c_emb, deterministic)

        # ══════════════ DECODER ══════════════
        u3 = nn.ConvTranspose(ch * 4, (4, 4), strides=(2, 2), padding='SAME')(b)
        u3 = jnp.concatenate([u3, e3], axis=-1)
        u3 = ResBlock(ch * 8, ch * 2, te, ce, use_attention=True)(u3, t_emb, c_emb, deterministic)

        u2 = nn.ConvTranspose(ch * 2, (4, 4), strides=(2, 2), padding='SAME')(u3)
        u2 = jnp.concatenate([u2, e2], axis=-1)
        u2 = ResBlock(ch * 4, ch, te, ce)(u2, t_emb, c_emb, deterministic)

        u1 = nn.ConvTranspose(ch, (4, 4), strides=(2, 2), padding='SAME')(u2)
        u1 = jnp.concatenate([u1, e1], axis=-1)
        u1 = ResBlock(ch * 2, ch, te, ce)(u1, t_emb, c_emb, deterministic)

        # Output
        out = nn.GroupNorm(num_groups=cfg.NUM_GROUPS)(u1)
        out = nn.silu(out)
        out = nn.Conv(self.img_channels, (3, 3), padding='SAME')(out)
        return out


# ═══════════════════════════════════════════════════════════════════════
# RECTIFIED FLOW HELPERS
# ═══════════════════════════════════════════════════════════════════════

def sample_timesteps(key, batch_size):
    if cfg.TIMESTEP_SAMPLING == "logit_normal":
        z = jrandom.normal(key, (batch_size,)) * cfg.LOGIT_NORMAL_STD + cfg.LOGIT_NORMAL_MEAN
        t = jax.nn.sigmoid(z)
        t = jnp.clip(t, cfg.SIGMA_MIN, 1.0 - cfg.SIGMA_MIN)
    else:
        t = jrandom.uniform(key, (batch_size,))
        t = t * (1.0 - cfg.SIGMA_MIN) + cfg.SIGMA_MIN
    return t


def add_noise(x0, t, noise):
    t_view = t[:, None, None, None]
    x_t = (1.0 - t_view) * x0 + t_view * noise
    velocity = noise - x0
    return x_t, velocity


def euler_sample(apply_fn, params, shape, class_labels, key):
    """Euler ODE sampling with fused CFG (single forward pass per step)."""
    b = shape[0]
    x = jrandom.normal(key, shape)
    timesteps = jnp.linspace(1.0, cfg.SIGMA_MIN, cfg.EULER_STEPS + 1)

    # FIX #10: Fuse cond + uncond into one forward pass
    uncond_labels = jnp.full_like(class_labels, cfg.NUM_CLASSES)

    def body(i, x):
        t_current = timesteps[i]
        dt = timesteps[i] - timesteps[i + 1]
        t_batch = jnp.full((b,), t_current)

        # Fused: concat cond+uncond into single batch → 1 forward pass instead of 2
        x2 = jnp.concatenate([x, x], axis=0)
        t2 = jnp.concatenate([t_batch, t_batch], axis=0)
        labels2 = jnp.concatenate([class_labels, uncond_labels], axis=0)
        v2 = apply_fn(params, x2, t2, labels2, deterministic=True)
        v_cond, v_uncond = jnp.split(v2, 2, axis=0)

        v = v_uncond + cfg.CFG_SCALE * (v_cond - v_uncond)
        return x - dt * v

    x = lax.fori_loop(0, cfg.EULER_STEPS, body, x)
    return jnp.clip(x, -1.0, 1.0)


# FIX #10b: pmap'd generation across all TPU cores
def euler_sample_pmap(apply_fn, params, shape_per_device, class_labels_per_device, key):
    """Per-device euler sample for use with pmap."""
    return euler_sample(apply_fn, params, shape_per_device, class_labels_per_device, key)


# ═══════════════════════════════════════════════════════════════════════
# EMA + TRAIN STATE
# ═══════════════════════════════════════════════════════════════════════

def ema_update(ema_params, params, decay):
    return jax.tree.map(lambda e, p: e * decay + p * (1 - decay), ema_params, params)

class TrainStateEMA(train_state.TrainState):
    ema_params: Any = None


# ═══════════════════════════════════════════════════════════════════════
# WEIGHT DECAY MASK (exclude biases and norms)
# ═══════════════════════════════════════════════════════════════════════

def create_weight_decay_mask(params):
    """Only apply weight decay to parameters with ndim >= 2 (kernels)."""
    return jax.tree.map(lambda x: x.ndim >= 2, params)


# ═══════════════════════════════════════════════════════════════════════
# TRAINING STEP
# ═══════════════════════════════════════════════════════════════════════

def create_train_step(model):
    @functools.partial(jax.pmap, axis_name='devices', donate_argnums=(0,))
    def train_step(state, batch_images, batch_labels, rng):
        rng, t_rng, noise_rng, drop_rng, aug_rng = jrandom.split(rng, 5)
        B = batch_images.shape[0]

        # Online augmentation
        if cfg.ONLINE_FLIP_ENABLED:
            flip_mask = jrandom.uniform(aug_rng, (B, 1, 1, 1)) < cfg.ONLINE_FLIP_P
            batch_images = jnp.where(flip_mask, jnp.flip(batch_images, axis=2), batch_images)

        # CFG label dropout
        drop_mask = jrandom.uniform(drop_rng, (B,)) < cfg.CFG_DROPOUT_RATE
        labels_cfg = jnp.where(drop_mask, cfg.NUM_CLASSES, batch_labels)

        t = sample_timesteps(t_rng, B)
        noise = jrandom.normal(noise_rng, batch_images.shape)
        x_noisy, velocity_target = add_noise(batch_images, t, noise)

        deterministic = not cfg.USE_DROPOUT

        def loss_fn(params):
            # FIX #5: pass dropout RNG so Flax won't crash if dropout is enabled
            v_pred = model.apply(
                params, x_noisy, t, labels_cfg, deterministic=deterministic,
                rngs={'dropout': drop_rng} if not deterministic else {}
            )
            return jnp.mean((v_pred - velocity_target) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        grads = lax.pmean(grads, axis_name='devices')
        loss = lax.pmean(loss, axis_name='devices')

        state = state.apply_gradients(grads=grads)
        new_ema = ema_update(state.ema_params, state.params, cfg.EMA_DECAY)
        state = state.replace(ema_params=new_ema)

        return state, loss, rng

    return train_step


# ═══════════════════════════════════════════════════════════════════════
# SAMPLE GENERATION
# ═══════════════════════════════════════════════════════════════════════

def generate_samples(model, params, epoch):
    npc = cfg.NUM_SAMPLES_PER_CLASS
    key = jrandom.PRNGKey(epoch)
    all_samples = {}

    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None

    wandb_images = {}
    for class_idx, class_name in cfg.IDX_TO_CLASS.items():
        labels = jnp.full((npc,), class_idx, dtype=jnp.int32)
        shape = (npc, cfg.IMG_SIZE, cfg.IMG_SIZE, cfg.IMG_CHANNELS)
        key, sample_key = jrandom.split(key)
        samples = euler_sample(model.apply, params, shape, labels, sample_key)
        samples_np = np.array((jnp.clip(samples, -1, 1) + 1) / 2 * 255).astype(np.uint8)
        all_samples[class_name] = samples_np

        if plt is not None:
            fig, axes = plt.subplots(1, npc, figsize=(npc * 3, 3))
            fig.suptitle(f'{class_name} — Epoch {epoch}', fontsize=14, fontweight='bold')
            for col in range(npc):
                ax = axes[col] if npc > 1 else axes
                ax.imshow(samples_np[col, :, :, 0], cmap='gray')
                ax.axis('off')
            plt.tight_layout()
            grid_path = os.path.join(cfg.OUTPUT_DIR, f"samples_epoch{epoch}_{class_name}.png")
            plt.savefig(grid_path, dpi=150, bbox_inches='tight')
            plt.close()
            if cfg.USE_WANDB and HAS_WANDB:
                wandb_images[f"samples/{class_name}"] = wandb.Image(grid_path)

    # Log ALL classes in one call so they appear at the same W&B step
    if wandb_images and cfg.USE_WANDB and HAS_WANDB:
        wandb.log(wandb_images)

    # Combined inline display
    if plt is not None:
        try:
            n_classes = len(all_samples)
            fig, axes = plt.subplots(n_classes, npc, figsize=(npc * 3, n_classes * 3))
            fig.suptitle(f'Generated Samples — Epoch {epoch}', fontsize=16, fontweight='bold')
            for row, (cname, imgs) in enumerate(all_samples.items()):
                for col in range(npc):
                    ax = axes[row, col] if n_classes > 1 else axes[col]
                    ax.imshow(imgs[col, :, :, 0], cmap='gray')
                    ax.axis('off')
                    if col == 0:
                        ax.set_ylabel(cname, fontsize=12, fontweight='bold',
                                      rotation=0, labelpad=60, va='center')
            plt.tight_layout(rect=[0.05, 0, 1, 0.95])
            combined_path = os.path.join(cfg.OUTPUT_DIR, f"samples_epoch{epoch}_all.png")
            plt.savefig(combined_path, dpi=150, bbox_inches='tight')
            try:
                from IPython.display import display, Image as IPImage
                plt.close()
                display(IPImage(filename=combined_path))
            except ImportError:
                plt.show(); plt.close()
        except Exception as e:
            print(f"  ⚠️ Combined display error: {e}")


def plot_loss(train_losses):
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(train_losses, linewidth=1)
        ax.set(xlabel='Epoch', ylabel='MSE Loss', title='Rectified Flow Loss (JAX/TPU)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(cfg.OUTPUT_DIR, "training_loss.png")
        plt.savefig(path, dpi=150)
        plt.close()
        if cfg.USE_WANDB and HAS_WANDB:
            wandb.log({"loss_curve": wandb.Image(path)})
    except ImportError:
        pass


# ═══════════════════════════════════════════════════════════════════════
# LR SCHEDULE
# ═══════════════════════════════════════════════════════════════════════

def build_lr_schedule(total_steps):
    if cfg.LR_SCHEDULE == "warmup_cosine":
        warmup = cfg.LR_WARMUP_STEPS if cfg.LR_WARMUP_STEPS > 0 else min(500, total_steps // 20)
        return optax.warmup_cosine_decay_schedule(
            init_value=0.0, peak_value=cfg.LEARNING_RATE,
            warmup_steps=warmup, decay_steps=total_steps,
            end_value=cfg.LEARNING_RATE * 0.01)
    elif cfg.LR_SCHEDULE == "constant":
        return cfg.LEARNING_RATE
    else:  # cosine_decay (default, matches GPU)
        # FIX #15: Don't decay all the way to zero
        return optax.warmup_cosine_decay_schedule(
            init_value=cfg.LEARNING_RATE, peak_value=cfg.LEARNING_RATE,
            warmup_steps=1, decay_steps=total_steps,
            end_value=cfg.LEARNING_RATE * 0.01)


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.makedirs(cfg.SYNTHETIC_DIR, exist_ok=True)
    for cls_name in cfg.IDX_TO_CLASS.values():
        os.makedirs(os.path.join(cfg.SYNTHETIC_DIR, cls_name), exist_ok=True)

    # ── Data ──
    all_paths, all_labels = load_dataset_paths(cfg.DATASET_ROOT)
    train_paths, val_paths, test_paths, train_labels, val_labels, test_labels = \
        sequential_split(all_paths, all_labels)
    print(f"Loading {len(train_paths)} training images...")
    train_images, train_labels_arr = load_images_to_memory(train_paths, train_labels)
    print(f"  Loaded: {train_images.shape}")

    # ── Model ──
    model = ConditionalUNet(
        img_channels=cfg.IMG_CHANNELS, base_channels=cfg.BASE_CHANNELS,
        time_emb_dim=cfg.TIME_EMB_DIM, num_classes=cfg.NUM_CLASSES,
        class_emb_dim=cfg.CLASS_EMB_DIM,
    )

    key = jrandom.PRNGKey(cfg.SEED)
    key, init_key = jrandom.split(key)
    dummy_x = jnp.ones((1, cfg.IMG_SIZE, cfg.IMG_SIZE, cfg.IMG_CHANNELS))
    dummy_t = jnp.ones((1,))
    dummy_c = jnp.ones((1,), dtype=jnp.int32)
    params = model.init(init_key, dummy_x, dummy_t, dummy_c, deterministic=True)

    total_params = sum(x.size for x in jax.tree.leaves(params))
    print(f"  Parameters: {total_params:,} ({total_params / 1e6:.2f}M)")

    # ── Optimizer ──
    steps_per_epoch = max(len(train_paths) // cfg.EFFECTIVE_BATCH, 1)
    total_steps = cfg.EPOCHS * steps_per_epoch
    lr_schedule = build_lr_schedule(total_steps)

    # FIX #19: Weight decay mask — don't decay biases and norms
    wd_mask = create_weight_decay_mask(params)
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.GRAD_CLIP_NORM),
        optax.adamw(lr_schedule, weight_decay=cfg.WEIGHT_DECAY, mask=wd_mask),
    )
    print(f"  LR: {cfg.LEARNING_RATE} ({cfg.LR_SCHEDULE}), steps/epoch: {steps_per_epoch}")

    # ── Train State ──
    state = TrainStateEMA.create(
        apply_fn=model.apply, params=params, tx=optimizer, ema_params=params)

    start_epoch = 1
    best_loss = float('inf')
    train_losses = []

    # ── Resume from checkpoint ──
    # RESUME_CHECKPOINT should point to the rf_output/ directory containing
    # checkpoint folders like best_596/, latest_600/, etc.
    # Also handles if user points directly to a checkpoint folder.
    if cfg.RESUME_CHECKPOINT and os.path.exists(cfg.RESUME_CHECKPOINT):
        import shutil
        ckpt_src = cfg.RESUME_CHECKPOINT
        # Detect if user pointed at a subfolder (e.g. best_596) vs parent (rf_output)
        basename = os.path.basename(ckpt_src.rstrip('/'))
        has_manifest = os.path.exists(os.path.join(ckpt_src, 'manifest.ocdbt'))
        if has_manifest:
            # User pointed at the checkpoint folder itself (e.g. .../best_596/)
            # Copy it as a subfolder of OUTPUT_DIR
            dst = os.path.join(cfg.OUTPUT_DIR, basename)
            if not os.path.exists(dst):
                shutil.copytree(ckpt_src, dst)
                print(f"📦 Copied checkpoint folder {basename}/ → {cfg.OUTPUT_DIR}")
        else:
            # User pointed at parent dir containing checkpoint folders
            print(f"📦 Copying checkpoints from {ckpt_src}...")
            for item in os.listdir(ckpt_src):
                src = os.path.join(ckpt_src, item)
                dst = os.path.join(cfg.OUTPUT_DIR, item)
                if os.path.isdir(src) and not os.path.exists(dst):
                    shutil.copytree(src, dst)
                    print(f"  ✅ Copied {item}/")
                elif os.path.isfile(src) and not os.path.exists(dst):
                    shutil.copy2(src, dst)

    # Try to restore: prefer latest_ over best_
    resumed = False
    for prefix in ['latest_', 'best_']:
        try:
            state = checkpoints.restore_checkpoint(cfg.OUTPUT_DIR, state, prefix=prefix)
            if state.step > 0:
                # state.step = optimizer steps (not epochs!)
                # Convert: epoch = optimizer_steps / steps_per_epoch
                resume_epoch = int(state.step) // steps_per_epoch
                start_epoch = resume_epoch + 1
                print(f"\n🔄 Resumed from '{prefix}' checkpoint")
                print(f"   Optimizer step: {state.step}, Epoch: {resume_epoch}, Resuming from: {start_epoch}")
                resumed = True
                break
        except Exception:
            continue

    # FIX #1: Restore best_loss from JSON sidecar
    if resumed:
        saved_best, saved_epoch = load_training_meta()
        if saved_best < float('inf'):
            best_loss = saved_best
            print(f"   Restored best_loss: {best_loss:.4f} (from epoch {saved_epoch})")
        else:
            print(f"   ⚠️ No training_meta.json found — best_loss starts at inf")
    else:
        print(f"\n🆕 No checkpoint found — training from scratch")

    # ── Replicate ──
    state = jax_utils.replicate(state)
    train_step = create_train_step(model)

    # ── W&B ──
    if cfg.USE_WANDB and HAS_WANDB:
        try:
            from kaggle_secrets import UserSecretsClient
            wandb.login(key=UserSecretsClient().get_secret("WANDB_API_KEY"))
        except Exception:
            try: wandb.login()
            except Exception: pass
        wandb.init(
            project=cfg.WANDB_PROJECT, name=cfg.WANDB_RUN_NAME,
            config={
                "method": "Rectified Flow", "framework": "JAX/Flax",
                "accelerator": f"TPU v5e × {NUM_DEVICES}",
                "per_device_batch": cfg.PER_DEVICE_BATCH,
                "effective_batch": cfg.EFFECTIVE_BATCH,
                "lr": cfg.LEARNING_RATE, "lr_schedule": cfg.LR_SCHEDULE,
                "epochs": cfg.EPOCHS, "params": total_params,
                "img_size": cfg.IMG_SIZE, "base_channels": cfg.BASE_CHANNELS,
            },
        )

    # ═══════════════════════════════════════════════════════════════
    # TRAINING LOOP
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"RECTIFIED FLOW — JAX on {NUM_DEVICES}× TPU v5e")
    print(f"{'='*60}")
    print(f"  Batch:  {cfg.PER_DEVICE_BATCH} × {NUM_DEVICES} = {cfg.EFFECTIVE_BATCH}")
    print(f"  LR:     {cfg.LEARNING_RATE:.6f} ({cfg.LR_SCHEDULE})")
    print(f"  Images: {len(train_images)} train, {steps_per_epoch} batches/epoch")
    print(f"  Save:   every {cfg.SAVE_EVERY} ep | Sample: every {cfg.SAMPLE_EVERY} ep")
    print(f"  Guard:  {cfg.MAX_TRAINING_HOURS}h")
    print(f"{'='*60}")
    print(f"\n🚀 Training {start_epoch}→{cfg.EPOCHS}")
    sys.stdout.flush()

    # FIX #2: Use epoch-aware RNG so resume doesn't replay epoch 1
    rng = jrandom.PRNGKey(cfg.SEED + start_epoch)
    rng = jrandom.split(rng, NUM_DEVICES)

    epoch_times = []
    # FIX #2: Advance NumPy RNG past previously-seen epochs
    np_rng = np.random.default_rng(cfg.SEED + start_epoch)
    last_epoch = start_epoch

    for epoch in range(start_epoch, cfg.EPOCHS + 1):
        last_epoch = epoch
        elapsed_h = (time.time() - TRAINING_START_TIME) / 3600
        if (cfg.MAX_TRAINING_HOURS - elapsed_h) < (cfg.GENERATION_BUDGET_MIN / 60):
            print(f"\n⏰ TIME GUARD: {elapsed_h:.1f}h. Stopping.")
            break

        epoch_start = time.time()

        batch_imgs, batch_labs, n_batches = make_batches(
            train_images, train_labels_arr, cfg.PER_DEVICE_BATCH, np_rng, NUM_DEVICES)

        # FIX #8: Accumulate loss on-device; no float() sync per batch
        epoch_loss_device = jnp.zeros((), dtype=jnp.float32)
        for i in range(n_batches):
            b_img = jnp.array(batch_imgs[i])
            b_lab = jnp.array(batch_labs[i])
            state, loss, rng = train_step(state, b_img, b_lab, rng)
            epoch_loss_device = epoch_loss_device + loss[0]

        # Single device→host sync per epoch (instead of per batch)
        avg_loss = float(jax.device_get(epoch_loss_device)) / max(n_batches, 1)
        train_losses.append(avg_loss)
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        elapsed_h = (time.time() - TRAINING_START_TIME) / 3600
        eta_min = (cfg.EPOCHS - epoch) * np.mean(epoch_times[-100:]) / 60

        if epoch % cfg.LOG_EVERY == 0:
            print(f"E{epoch}/{cfg.EPOCHS} | Loss: {avg_loss:.4f} | "
                  f"Best: {best_loss:.4f} | {epoch_time:.1f}s | "
                  f"{elapsed_h:.1f}h | ETA: {eta_min:.0f}m")
            sys.stdout.flush()

        if cfg.USE_WANDB and HAS_WANDB:
            wandb.log({
                "train/loss": avg_loss, "train/best_loss": min(best_loss, avg_loss),
                "train/epoch_time_s": epoch_time, "train/epoch": epoch,
                "train/elapsed_h": elapsed_h,
            })

        # FIX #11: Deduplicate unreplicate — compute once, use everywhere
        need_single = (avg_loss < best_loss) or \
                      (epoch % cfg.SAVE_EVERY == 0) or \
                      (epoch % cfg.SAMPLE_EVERY == 0)

        if need_single:
            single_state = jax_utils.unreplicate(state)

        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoints.save_checkpoint(
                cfg.OUTPUT_DIR, single_state, step=int(single_state.step),
                prefix='best_', overwrite=True)
            # FIX #1: Persist best_loss to JSON
            save_training_meta(best_loss, epoch)
            if epoch % cfg.LOG_EVERY == 0:
                print(f"  ★ Best (loss={best_loss:.4f})")

        if epoch % cfg.SAVE_EVERY == 0:
            checkpoints.save_checkpoint(
                cfg.OUTPUT_DIR, single_state, step=int(single_state.step),
                prefix='latest_', overwrite=True)
            # Also update meta on latest save
            save_training_meta(best_loss, epoch)
            print(f"  💾 Checkpoint epoch {epoch}")

        if epoch % cfg.SAMPLE_EVERY == 0:
            generate_samples(model, single_state.ema_params, epoch)

    # ── Final ──
    single_state = jax_utils.unreplicate(state)
    checkpoints.save_checkpoint(
        cfg.OUTPUT_DIR, single_state, step=int(single_state.step),
        prefix='final_', overwrite=True)
    save_training_meta(best_loss, last_epoch)
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE — {last_epoch} epochs, best={best_loss:.4f}, "
          f"{(time.time() - TRAINING_START_TIME) / 3600:.1f}h")
    print(f"{'='*60}")
    plot_loss(train_losses)

    # ═══════════════════════════════════════════════════════════════
    # GENERATION (pmap'd across all TPU cores)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}\nGENERATION\n{'='*60}")
    gen_params = single_state.ema_params
    # Replicate params for pmap generation
    gen_params_rep = jax_utils.replicate(gen_params)

    # Create pmap'd generation function
    p_euler_sample = jax.pmap(
        lambda params, labels, shape0, key: euler_sample(
            model.apply, params,
            (shape0, cfg.IMG_SIZE, cfg.IMG_SIZE, cfg.IMG_CHANNELS),
            labels, key),
        in_axes=(0, 0, None, 0),
        static_broadcasted_argnums=(2,),
    )

    gen_targets = {
        0: ("Benign", cfg.NUM_SYNTHETIC_BENIGN),
        1: ("Malignant", cfg.NUM_SYNTHETIC_MALIGNANT),
        2: ("Normal", cfg.NUM_SYNTHETIC_NORMAL),
    }
    for class_idx, (class_name, total) in gen_targets.items():
        if total <= 0:
            continue
        class_dir = os.path.join(cfg.SYNTHETIC_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)
        print(f"\n🎨 Generating {total} {class_name}...")

        generated = 0
        gen_key = jrandom.PRNGKey(42 + class_idx)

        # Generate in parallel across all devices
        per_device = cfg.GEN_BATCH_SIZE // NUM_DEVICES
        if per_device < 1:
            per_device = 1

        while generated < total:
            cur_total = min(per_device * NUM_DEVICES, total - generated)
            cur_per_device = cur_total // NUM_DEVICES
            if cur_per_device < 1:
                # Fallback to single-device for small remainders
                labels = jnp.full((cur_total,), class_idx, dtype=jnp.int32)
                shape = (cur_total, cfg.IMG_SIZE, cfg.IMG_SIZE, cfg.IMG_CHANNELS)
                gen_key, sk = jrandom.split(gen_key)
                samples = euler_sample(model.apply, gen_params, shape, labels, sk)
                samples = np.array((jnp.clip(samples, -1, 1) + 1) / 2 * 255).astype(np.uint8)
                for j in range(cur_total):
                    Image.fromarray(samples[j, :, :, 0]).save(
                        os.path.join(class_dir, f"{class_name.lower()}_{generated:04d}.png"))
                    generated += 1
            else:
                # Parallel generation across all TPU cores
                labels_all = jnp.full((NUM_DEVICES, cur_per_device), class_idx, dtype=jnp.int32)
                gen_key, *device_keys = jrandom.split(gen_key, NUM_DEVICES + 1)
                device_keys = jnp.stack(device_keys)
                samples = p_euler_sample(gen_params_rep, labels_all, cur_per_device, device_keys)
                # samples shape: (NUM_DEVICES, cur_per_device, H, W, C)
                samples = samples.reshape(-1, cfg.IMG_SIZE, cfg.IMG_SIZE, cfg.IMG_CHANNELS)
                samples = np.array((jnp.clip(samples, -1, 1) + 1) / 2 * 255).astype(np.uint8)
                for j in range(samples.shape[0]):
                    if generated >= total:
                        break
                    Image.fromarray(samples[j, :, :, 0]).save(
                        os.path.join(class_dir, f"{class_name.lower()}_{generated:04d}.png"))
                    generated += 1

        print(f"  ✅ {generated} saved")

    if cfg.USE_WANDB and HAS_WANDB:
        wandb.finish()
    print("\n🎉 Done!")


print("=" * 60)
print("Rectified Flow — JAX/Flax on TPU v5e")
print("=" * 60)
main()
