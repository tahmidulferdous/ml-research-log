"""
=============================================================================
Class-Conditional Rectified Flow for CT Scan Augmentation — TPU v5e
=============================================================================
Optimized for Kaggle TPU v5e (single-process, BF16).
Uses PyTorch/XLA for native TPU acceleration.

Usage (notebook):  Just paste and run — works in Kaggle TPU notebooks.
Usage (script):    python rectified_flow_tpu.py
=============================================================================
"""

import os, sys, math, copy, time, random, functools
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ── TPU / XLA imports ──
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met

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


# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

class Config:
    # ── Dataset ──
    DATASET_ROOT = "/kaggle/input/datasets/hamdallak/the-iqothnccd-lung-cancer-dataset/The IQ-OTHNCCD lung cancer dataset"
    OUTPUT_DIR = "/kaggle/working/rf_output"
    SYNTHETIC_DIR = "/kaggle/working/synthetic_images_rf"

    # ── Classes ──
    CLASSES = ["Bengin cases", "Malignant cases", "Normal cases"]
    CLASS_TO_IDX = {"Bengin cases": 0, "Malignant cases": 1, "Normal cases": 2}
    IDX_TO_CLASS = {0: "Benign", 1: "Malignant", 2: "Normal"}
    NUM_CLASSES = 3

    # ── Image ──
    IMG_SIZE = 224
    IMG_CHANNELS = 1

    # ── Rectified Flow ──
    SIGMA_MIN = 1e-4
    TIMESTEP_SAMPLING = "logit_normal"
    LOGIT_NORMAL_MEAN = 0.0
    LOGIT_NORMAL_STD = 1.0

    # ── Model ──
    BASE_CHANNELS = 128
    TIME_EMB_DIM = 256
    CLASS_EMB_DIM = 256

    # ── Training ──
    EPOCHS = 3000
    BATCH_SIZE = 24              # 16GB HBM, 224×224 grayscale
    LEARNING_RATE = 1.5e-4
    EMA_DECAY = 0.9999
    SAVE_EVERY = 20
    SAMPLE_EVERY = 20

    # ── W&B ──
    WANDB_PROJECT = "ct-rectified-flow"
    WANDB_RUN_NAME = "rf-TPUv5e"
    USE_WANDB = True

    # ── Kaggle Time Guard ──
    MAX_TRAINING_HOURS = 7.5     # Leave 30min for generation + eval
    GENERATION_BUDGET_MIN = 30

    # ── Resume (auto-detect) ──
    RESUME_CHECKPOINT = None  # Will be auto-detected at runtime

    @staticmethod
    def auto_detect_checkpoint():
        """Find the best checkpoint to resume from."""
        candidates = [
          "/kaggle/input/models/tahmidulkashfi/rectified-flow-75m-1300epoch/pytorch/default/1/best_rf_model75m_1300epoch.pt",
            os.path.join(Config.OUTPUT_DIR, "latest_checkpoint.pt"),
            os.path.join(Config.OUTPUT_DIR, "final_checkpoint.pt"),
            os.path.join(Config.OUTPUT_DIR, "best_rf_model.pt"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    # ── Generation ──
    NUM_SYNTHETIC_BENIGN = 300
    NUM_SYNTHETIC_NORMAL = 100
    EULER_STEPS = 50
    CFG_SCALE = 4.0

    # ── Reproducibility ──
    SEED = 42
    NUM_WORKERS = 4


cfg = Config()


# ═══════════════════════════════════════════════════════════════════════
# RECTIFIED FLOW SCHEDULER
# ═══════════════════════════════════════════════════════════════════════

class RectifiedFlowScheduler:
    def __init__(self, sigma_min=1e-4, device=None):
        self.sigma_min = sigma_min
        self.device = device

    def sample_timesteps(self, batch_size, method="logit_normal", mean=0.0, std=1.0):
        if method == "logit_normal":
            z = torch.randn(batch_size, device=self.device) * std + mean
            t = torch.sigmoid(z)
            t = t.clamp(self.sigma_min, 1.0 - self.sigma_min)
        else:
            t = torch.rand(batch_size, device=self.device)
            t = t * (1.0 - self.sigma_min) + self.sigma_min
        return t

    def add_noise(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        t_view = t.view(-1, 1, 1, 1)
        x_t = (1.0 - t_view) * x0 + t_view * noise
        velocity = noise - x0
        return x_t, noise, velocity

    @torch.no_grad()
    def sample_euler(self, model, shape, class_labels, num_steps=30,
                     cfg_scale=3.0, num_classes=3):
        model.eval()
        b = shape[0]
        device = self.device
        x = torch.randn(shape, device=device)
        timesteps = torch.linspace(1.0, self.sigma_min, num_steps + 1, device=device)

        for i in range(num_steps):
            t_current = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_current - t_next
            t_batch = torch.full((b,), t_current, device=device)

            v_cond = model(x, t_batch, class_labels)
            uncond_labels = torch.full_like(class_labels, num_classes)
            v_uncond = model(x, t_batch, uncond_labels)
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
            x = x - dt * v
            xm.mark_step()  # Prevent XLA graph accumulation

        return torch.clamp(x, -1.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════
# U-NET MODEL
# ═══════════════════════════════════════════════════════════════════════

class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        t_scaled = t.float() * 1000.0
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t_scaled[:, None] * emb[None, :]
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
        q = self.q(h).view(B, C, H * W).permute(0, 2, 1)
        k = self.k(h).view(B, C, H * W).permute(0, 2, 1)
        v = self.v(h).view(B, C, H * W).permute(0, 2, 1)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 2, 1).view(B, C, H, W)
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
            nn.GroupNorm(8, out_ch), nn.SiLU(), nn.Dropout(0.1),
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

        self.out = nn.Sequential(
            nn.GroupNorm(8, ch), nn.SiLU(),
            nn.Conv2d(ch, img_channels, 3, padding=1),
        )

    def forward(self, x, t, c):
        t_emb = self.time_embed(t)
        c_emb = self.class_embed(c)

        x0 = self.enc_conv0(x)
        e1 = self.enc1(x0, t_emb, c_emb)
        e2 = self.enc2(self.down1(e1), t_emb, c_emb)
        e3 = self.enc3(self.down2(e2), t_emb, c_emb)

        b = self.bot1(self.down3(e3), t_emb, c_emb)
        b = self.bot2(b, t_emb, c_emb)
        b = self.bot3(b, t_emb, c_emb)

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1), t_emb, c_emb)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1), t_emb, c_emb)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1), t_emb, c_emb)
        return self.out(d1)


# ═══════════════════════════════════════════════════════════════════════
# EMA
# ═══════════════════════════════════════════════════════════════════════

class EMA:
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
        xm.mark_step()  # Batch the EMA update into one XLA op

    def forward(self, *args, **kwargs):
        return self.shadow(*args, **kwargs)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict):
        self.shadow.load_state_dict(state_dict)


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


class CTDataset(Dataset):
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
                    A.GaussNoise(p=0.2),
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
            img = np.array(Image.fromarray(img).resize((self.img_size, self.img_size)))
        img = img.astype(np.float32) / 255.0
        img = img * 2.0 - 1.0
        img = torch.FloatTensor(img).unsqueeze(0)
        label = torch.LongTensor([self.labels[idx]]).squeeze()
        return img, label


def sequential_split(paths, labels, train_ratio=0.70, val_ratio=0.15):
    class_files = defaultdict(list)
    for p, l in zip(paths, labels):
        class_files[l].append(p)
    train_p, train_l, val_p, val_l, test_p, test_l = [], [], [], [], [], []
    for cls_idx, cls_paths in sorted(class_files.items()):
        n = len(cls_paths)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_p.extend(cls_paths[:n_train]); train_l.extend([cls_idx] * n_train)
        val_p.extend(cls_paths[n_train:n_train + n_val]); val_l.extend([cls_idx] * n_val)
        test_p.extend(cls_paths[n_train + n_val:]); test_l.extend([cls_idx] * (n - n_train - n_val))
    return train_p, val_p, test_p, train_l, val_l, test_l


# ═══════════════════════════════════════════════════════════════════════
# SAMPLE GENERATION (during training)
# ═══════════════════════════════════════════════════════════════════════

def _generate_samples(model, scheduler, epoch, device):
    """Generate and display sample images per class during training."""
    model.eval()
    num_per_class = 4
    all_samples = {}

    for class_idx, class_name in cfg.IDX_TO_CLASS.items():
        labels = torch.full((num_per_class,), class_idx, device=device, dtype=torch.long)
        shape = (num_per_class, cfg.IMG_CHANNELS, cfg.IMG_SIZE, cfg.IMG_SIZE)
        samples = scheduler.sample_euler(
            model, shape, labels, num_steps=cfg.EULER_STEPS,
            cfg_scale=cfg.CFG_SCALE, num_classes=cfg.NUM_CLASSES
        )
        samples = (samples.clamp(-1, 1) + 1) / 2  # [-1,1] → [0,1]
        samples_cpu = samples.cpu()
        all_samples[class_name] = samples_cpu

        # Save grid per class
        import torchvision.utils as vutils
        grid = vutils.make_grid(samples_cpu, nrow=num_per_class, padding=2)
        grid_path = os.path.join(cfg.OUTPUT_DIR, f"samples_epoch{epoch}_{class_name}.png")
        vutils.save_image(grid, grid_path)
        if cfg.USE_WANDB and HAS_WANDB:
            wandb.log({f"samples/{class_name}": wandb.Image(grid_path)})

    # Display inline matplotlib figure (3 rows × 4 cols)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        n_classes = len(all_samples)
        fig, axes = plt.subplots(n_classes, num_per_class,
                                 figsize=(num_per_class * 3, n_classes * 3))
        fig.suptitle(f'Generated Samples — Epoch {epoch}', fontsize=16, fontweight='bold')

        for row, (class_name, imgs) in enumerate(all_samples.items()):
            for col in range(num_per_class):
                ax = axes[row, col] if n_classes > 1 else axes[col]
                img_np = imgs[col, 0].numpy()  # [H, W]
                ax.imshow(img_np, cmap='gray', vmin=0, vmax=1)
                ax.axis('off')
                if col == 0:
                    ax.set_ylabel(class_name, fontsize=12, fontweight='bold', rotation=0,
                                  labelpad=60, va='center')

        plt.tight_layout(rect=[0.05, 0, 1, 0.95])
        combined_path = os.path.join(cfg.OUTPUT_DIR, f"samples_epoch{epoch}_all.png")
        plt.savefig(combined_path, dpi=150, bbox_inches='tight')

        # Show inline in notebook
        try:
            from IPython.display import display, Image as IPImage
            plt.close()
            display(IPImage(filename=combined_path))
        except ImportError:
            plt.show()
            plt.close()

        if cfg.USE_WANDB and HAS_WANDB:
            wandb.log({f"samples/all_classes": wandb.Image(combined_path)})
    except ImportError:
        pass

    model.train()


def _plot_loss(train_losses):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(train_losses, label='Training Loss', linewidth=1)
        ax.set(xlabel='Epoch', ylabel='MSE Loss (velocity)',
               title='Rectified Flow Training Loss (TPU v5e)')
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.OUTPUT_DIR, "training_loss.png"), dpi=150)
        plt.close()
    except ImportError:
        pass


# ═══════════════════════════════════════════════════════════════════════
# FID / EVALUATION
# ═══════════════════════════════════════════════════════════════════════

class InceptionV3Features(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            from torchvision.models import inception_v3, Inception_V3_Weights
            inception = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        except Exception:
            from torchvision.models import inception_v3
            inception = inception_v3(pretrained=True)
        self.blocks = nn.Sequential(
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3, nn.MaxPool2d(3, stride=2),
            inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            nn.MaxPool2d(3, stride=2),
            inception.Mixed_5b, inception.Mixed_5c, inception.Mixed_5d,
            inception.Mixed_6a, inception.Mixed_6b, inception.Mixed_6c,
            inception.Mixed_6d, inception.Mixed_6e,
            inception.Mixed_7a, inception.Mixed_7b, inception.Mixed_7c,
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = (x + 1) / 2
        x = x.repeat(1, 3, 1, 1)
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        x = (x - self.mean) / self.std
        return self.blocks(x).flatten(1)


class ImageFolderDataset(Dataset):
    def __init__(self, folder, img_size=224):
        self.paths = sorted([
            p for p in Path(folder).rglob("*")
            if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        ])
        self.img_size = img_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('L')
        img = img.resize((self.img_size, self.img_size))
        img = np.array(img, dtype=np.float32) / 255.0
        img = img * 2.0 - 1.0
        return torch.FloatTensor(img).unsqueeze(0)


def compute_fid(real_dir, synthetic_dir, device, img_size=224, batch_size=32):
    from scipy.linalg import sqrtm
    real_ds = ImageFolderDataset(real_dir, img_size)
    synth_ds = ImageFolderDataset(synthetic_dir, img_size)
    if len(real_ds) == 0 or len(synth_ds) == 0:
        return float('inf')
    real_loader = DataLoader(real_ds, batch_size=batch_size, shuffle=False)
    synth_loader = DataLoader(synth_ds, batch_size=batch_size, shuffle=False)
    feat_ext = InceptionV3Features().to(device).eval()

    def extract(loader):
        feats = []
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                feats.append(feat_ext(batch.to(device)).cpu().numpy())
        return np.concatenate(feats, axis=0)

    real_f = extract(real_loader)
    synth_f = extract(synth_loader)
    mu_r, sigma_r = np.mean(real_f, 0), np.cov(real_f, rowvar=False)
    mu_s, sigma_s = np.mean(synth_f, 0), np.cov(synth_f, rowvar=False)
    eps = 1e-6
    sigma_r += eps * np.eye(sigma_r.shape[0])
    sigma_s += eps * np.eye(sigma_s.shape[0])
    diff = mu_r - mu_s
    covmean, _ = sqrtm(sigma_r @ sigma_s, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma_r + sigma_s - 2 * covmean))


def nearest_neighbor_analysis(real_dir, synthetic_dir, device, img_size=224,
                               num_samples=50):
    real_ds = ImageFolderDataset(real_dir, img_size)
    synth_ds = ImageFolderDataset(synthetic_dir, img_size)
    if len(real_ds) == 0 or len(synth_ds) == 0:
        return float('inf'), []
    real_images = [real_ds[i] for i in range(min(len(real_ds), 500))]
    real_tensor = torch.stack(real_images).to(device)
    min_distances = []
    for i in range(min(num_samples, len(synth_ds))):
        synth_img = synth_ds[i].unsqueeze(0).to(device)
        dists = torch.norm(real_tensor.flatten(1) - synth_img.flatten(1), dim=1)
        min_distances.append(dists.min().item())
    avg_dist = np.mean(min_distances) if min_distances else float('inf')
    return avg_dist, min_distances


def plot_tsne(real_dir, synthetic_dir, device, img_size=224, batch_size=32,
              save_path=None):
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
    except ImportError:
        print("⚠️ matplotlib/sklearn not available — skipping t-SNE")
        return
    real_ds = ImageFolderDataset(real_dir, img_size)
    synth_ds = ImageFolderDataset(synthetic_dir, img_size)
    real_loader = DataLoader(real_ds, batch_size=batch_size, shuffle=False)
    synth_loader = DataLoader(synth_ds, batch_size=batch_size, shuffle=False)
    feat_ext = InceptionV3Features().to(device).eval()
    def extract(loader):
        feats = []
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)): batch = batch[0]
                feats.append(feat_ext(batch.to(device)).cpu().numpy())
        return np.concatenate(feats, axis=0)
    real_f = extract(real_loader)
    synth_f = extract(synth_loader)
    all_f = np.concatenate([real_f, synth_f])
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_f) - 1))
    emb = tsne.fit_transform(all_f)
    fig, ax = plt.subplots(figsize=(10, 8))
    n_r = len(real_f)
    ax.scatter(emb[:n_r, 0], emb[:n_r, 1], c='blue', alpha=0.6, label='Real', s=20)
    ax.scatter(emb[n_r:, 0], emb[n_r:, 1], c='red', alpha=0.6, label='Synthetic (RF)', s=20)
    ax.legend(fontsize=12)
    ax.set_title('t-SNE: Real vs Rectified Flow Synthetic (InceptionV3)', fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = save_path or os.path.join(cfg.OUTPUT_DIR, "tsne_real_vs_rf.png")
    plt.savefig(save_path, dpi=150); plt.close()
    print(f"  📊 t-SNE saved: {save_path}")


def run_evaluation(synthetic_dir=None, device='cpu'):
    synthetic_dir = synthetic_dir or cfg.SYNTHETIC_DIR
    print(f"\n{'='*60}\nQUALITY EVALUATION\n{'='*60}")
    real_root = cfg.DATASET_ROOT
    results = {}
    for class_name in ["Benign", "Normal"]:
        search_terms = [class_name.lower()]
        if class_name.lower() == "benign":
            search_terms.append("bengin")
        matches = [c for c in cfg.CLASSES if any(s in c.lower() for s in search_terms)]
        if not matches:
            continue
        real_class_dir = os.path.join(real_root, matches[0])
        synth_class_dir = os.path.join(synthetic_dir, class_name)
        if not os.path.exists(synth_class_dir):
            continue
        synth_count = len(list(Path(synth_class_dir).glob("*.png")))
        if synth_count == 0:
            continue
        print(f"\n── {class_name} ({synth_count} images) ──")
        fid = compute_fid(real_class_dir, synth_class_dir, device, cfg.IMG_SIZE)
        quality = "🟢 Excellent" if fid < 20 else "🟢 Good" if fid < 50 else "🟡 OK" if fid < 100 else "🔴 Poor"
        print(f"  FID: {fid:.2f} — {quality}")
        results[f"fid_{class_name.lower()}"] = fid
        avg_nn, _ = nearest_neighbor_analysis(real_class_dir, synth_class_dir, device, cfg.IMG_SIZE)
        print(f"  Avg NN distance: {avg_nn:.4f}")
        results[f"nn_{class_name.lower()}"] = avg_nn
        try:
            tsne_path = os.path.join(cfg.OUTPUT_DIR, f"tsne_{class_name.lower()}.png")
            plot_tsne(real_class_dir, synth_class_dir, device, cfg.IMG_SIZE, save_path=tsne_path)
        except Exception as e:
            print(f"  ⚠️ t-SNE failed: {e}")
    results_path = os.path.join(cfg.OUTPUT_DIR, "quality_scores.txt")
    with open(results_path, 'w') as f:
        for k, v in results.items():
            f.write(f"{k}: {v:.4f}\n")
    print(f"\n✅ Evaluation complete. Results: {results_path}")
    return results


# ═══════════════════════════════════════════════════════════════════════
# TRAINING (single-process, single TPU device)
# ═══════════════════════════════════════════════════════════════════════

def train_and_generate():
    """Main training + generation + evaluation — single TPU core."""
    device = xm.xla_device()
    print(f"  TPU device: {device}")

    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.makedirs(cfg.SYNTHETIC_DIR, exist_ok=True)
    for cls_name in cfg.IDX_TO_CLASS.values():
        os.makedirs(os.path.join(cfg.SYNTHETIC_DIR, cls_name), exist_ok=True)

    # ── Data ──
    all_paths, all_labels = load_dataset_paths(cfg.DATASET_ROOT)
    train_paths, val_paths, test_paths, train_labels, val_labels, test_labels = \
        sequential_split(all_paths, all_labels)

    print(f"\n{'='*60}")
    print(f"RECTIFIED FLOW — TPU v5e Training")
    print(f"{'='*60}")
    print(f"  Precision:     BF16 (TPU native)")
    print(f"  Batch Size:    {cfg.BATCH_SIZE}")
    print(f"  Resolution:    {cfg.IMG_SIZE}×{cfg.IMG_SIZE}")
    print(f"  Images:        {len(all_paths)} total, {len(train_paths)} train")
    print(f"  Time Guard:    {cfg.MAX_TRAINING_HOURS}h")
    print(f"{'='*60}")

    train_dataset = CTDataset(train_paths, train_labels, cfg.IMG_SIZE, augment=True)
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.BATCH_SIZE,
        shuffle=True, num_workers=cfg.NUM_WORKERS,
        drop_last=True, persistent_workers=False,  # Safer for TPU
        pin_memory=False,  # Not needed for TPU
    )

    # ── Model ──
    model = ConditionalUNet(
        img_channels=cfg.IMG_CHANNELS, base_channels=cfg.BASE_CHANNELS,
        time_emb_dim=cfg.TIME_EMB_DIM, num_classes=cfg.NUM_CLASSES,
        class_emb_dim=cfg.CLASS_EMB_DIM,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters:    {total_params:,} ({total_params / 1e6:.2f}M)")

    # ── Scheduler, Optimizer, EMA ──
    scheduler = RectifiedFlowScheduler(sigma_min=cfg.SIGMA_MIN, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)
    ema = EMA(model, decay=cfg.EMA_DECAY)

    # ── W&B ──
    if cfg.USE_WANDB and HAS_WANDB:
        try:
            from kaggle_secrets import UserSecretsClient
            wandb.login(key=UserSecretsClient().get_secret("WANDB_API_KEY"))
        except Exception:
            try:
                wandb.login()
            except Exception:
                pass
        wandb.init(
            project=cfg.WANDB_PROJECT, name=cfg.WANDB_RUN_NAME,
            config={
                "method": "Rectified Flow", "accelerator": "TPU v5e",
                "img_size": cfg.IMG_SIZE, "epochs": cfg.EPOCHS,
                "batch_size": cfg.BATCH_SIZE,
                "lr": cfg.LEARNING_RATE, "precision": "bfloat16",
            },
        )

    # ── Resume ──
    train_losses = []
    best_loss = float('inf')
    stop_reason = "completed"
    start_epoch = 1
    p_uncond = 0.1

    resume_path = cfg.RESUME_CHECKPOINT or cfg.auto_detect_checkpoint()
    if resume_path and os.path.exists(resume_path):
        print(f"\n🔄 Resuming from: {resume_path}")
        ckpt = torch.load(resume_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        if 'ema_state_dict' in ckpt:
            ema.load_state_dict(ckpt['ema_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            # Move optimizer states to TPU device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            for pg in optimizer.param_groups:
                pg['lr'] = cfg.LEARNING_RATE
                pg['initial_lr'] = cfg.LEARNING_RATE
        start_epoch = ckpt.get('epoch', 0) + 1
        best_loss = ckpt.get('best_loss', float('inf'))
        train_losses = ckpt.get('train_losses', [])
        remaining = cfg.EPOCHS - start_epoch + 1
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining)
        print(f"  ✅ Resumed at epoch {start_epoch}, best_loss={best_loss:.4f}")
        del ckpt
    else:
        print("\n📝 No checkpoint found — starting fresh")

    # ═══════════════════════════════════════════════════════════════
    # TRAINING LOOP
    # ═══════════════════════════════════════════════════════════════
    print(f"\n🚀 Training {start_epoch}→{cfg.EPOCHS} | BF16")
    print(f"  ⏳ First epoch will be slow (XLA graph compilation ~3-5 min)...")
    sys.stdout.flush()

    epoch_times = []

    for epoch in range(start_epoch, cfg.EPOCHS + 1):
        elapsed_h = (time.time() - TRAINING_START_TIME) / 3600
        if (cfg.MAX_TRAINING_HOURS - elapsed_h) < (cfg.GENERATION_BUDGET_MIN / 60):
            stop_reason = f"time_guard ({elapsed_h:.1f}h)"
            print(f"\n⏰ TIME GUARD: {elapsed_h:.1f}h. Stopping.")
            break

        model.train()
        epoch_loss = torch.zeros(1, device=device)
        num_batches = 0
        epoch_start = time.time()

        for batch_images, batch_labels in train_loader:
            batch_images = batch_images.to(device, dtype=torch.bfloat16)
            batch_labels = batch_labels.to(device)
            B = batch_images.shape[0]

            mask = torch.rand(B, device=device) < p_uncond
            batch_labels_cfg = torch.where(mask, cfg.NUM_CLASSES, batch_labels)

            t = scheduler.sample_timesteps(
                B, method=cfg.TIMESTEP_SAMPLING,
                mean=cfg.LOGIT_NORMAL_MEAN, std=cfg.LOGIT_NORMAL_STD
            )
            noise = torch.randn_like(batch_images)
            x_noisy, _, velocity_target = scheduler.add_noise(batch_images, t, noise)

            v_pred = model(x_noisy, t, batch_labels_cfg)
            loss = F.mse_loss(v_pred, velocity_target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            xm.optimizer_step(optimizer)  # mark_step materializes graph
            ema.update(model)  # includes mark_step()

            # Accumulate on device — single sync per epoch
            epoch_loss += loss.detach()
            num_batches += 1

        # ── Epoch Summary (single .item() sync per epoch) ──
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        avg_loss = (epoch_loss / max(num_batches, 1)).item()
        train_losses.append(avg_loss)
        lr_scheduler.step()

        # NaN check (once per epoch, not per batch)
        if math.isnan(avg_loss) or math.isinf(avg_loss):
            print(f"  ⚠️ NaN/Inf at epoch {epoch}! Rolling back...")
            rb = os.path.join(cfg.OUTPUT_DIR, 'latest_checkpoint.pt')
            if os.path.exists(rb):
                ck = torch.load(rb, map_location='cpu', weights_only=False)
                model.load_state_dict(ck['model_state_dict'])
                if 'ema_state_dict' in ck:
                    ema.load_state_dict(ck['ema_state_dict'])
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=lr_scheduler.get_last_lr()[0], weight_decay=1e-4)
                del ck
                print(f"  ✅ Rolled back.")
            continue

        elapsed_h = (time.time() - TRAINING_START_TIME) / 3600
        eta_min = (cfg.EPOCHS - epoch) * np.mean(epoch_times) / 60
        print(f"E{epoch}/{cfg.EPOCHS} | Loss: {avg_loss:.4f} | "
              f"Best: {best_loss:.4f} | {epoch_time:.1f}s | "
              f"{elapsed_h:.1f}h | ETA: {eta_min:.0f}m | "
              f"LR: {lr_scheduler.get_last_lr()[0]:.6f}")
        sys.stdout.flush()

        if cfg.USE_WANDB and HAS_WANDB:
            # ── TPU utilization metrics ──
            xla_report = met.metric_data('CompileTime')
            xla_exec = met.metric_data('ExecuteTime')
            tpu_metrics = {}
            if xla_exec:
                tpu_metrics['tpu/execute_time_ms'] = xla_exec[1] / max(xla_exec[0], 1)
            if xla_report:
                tpu_metrics['tpu/compile_time_ms'] = xla_report[1] / max(xla_report[0], 1)
            tpu_metrics['tpu/epoch_time_s'] = epoch_time
            # Utilization estimate
            if epoch_time > 0 and xla_exec:
                compute_frac = (xla_exec[1] / 1e9) / epoch_time
                tpu_metrics['tpu/utilization_pct'] = min(compute_frac * 100, 100.0)

            # ── HBM Memory ──
            try:
                mem_info = xm.get_memory_info(device)
                hbm_used_gb = mem_info.get('bytes_used', 0) / 1e9
                hbm_limit_gb = mem_info.get('bytes_limit', 0) / 1e9
                hbm_pct = (hbm_used_gb / hbm_limit_gb * 100) if hbm_limit_gb > 0 else 0
                tpu_metrics['tpu/hbm_used_gb'] = hbm_used_gb
                tpu_metrics['tpu/hbm_total_gb'] = hbm_limit_gb
                tpu_metrics['tpu/hbm_pct'] = hbm_pct
            except Exception:
                hbm_used_gb = hbm_limit_gb = hbm_pct = 0

            # ── Throughput ──
            imgs_per_sec = (num_batches * cfg.BATCH_SIZE) / max(epoch_time, 1)
            tpu_metrics['tpu/images_per_sec'] = imgs_per_sec

            wandb.log({
                "train/loss": avg_loss, "train/best_loss": min(best_loss, avg_loss),
                "train/lr": lr_scheduler.get_last_lr()[0],
                "train/epoch_time_s": epoch_time,
                "train/elapsed_h": elapsed_h, "train/epoch": epoch,
                **tpu_metrics,
            })

            # Print TPU stats every epoch
            util_pct = tpu_metrics.get('tpu/utilization_pct', 0)
            print(f"  \U0001f4ca TPU: HBM {hbm_used_gb:.1f}/{hbm_limit_gb:.1f}GB "
                  f"({hbm_pct:.0f}%) | Util ~{util_pct:.0f}% | "
                  f"{imgs_per_sec:.1f} img/s")
            sys.stdout.flush()

        if avg_loss < best_loss:
            best_loss = avg_loss
            xm.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'ema_state_dict': ema.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'train_losses': train_losses,
                'config': {
                    'method': 'rectified_flow', 'img_size': cfg.IMG_SIZE,
                    'base_channels': cfg.BASE_CHANNELS,
                    'num_classes': cfg.NUM_CLASSES,
                },
            }, os.path.join(cfg.OUTPUT_DIR, "best_rf_model.pt"))
            print(f"  ★ Best (loss={best_loss:.4f})")

        if epoch % cfg.SAVE_EVERY == 0:
            xm.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'ema_state_dict': ema.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'train_losses': train_losses,
            }, os.path.join(cfg.OUTPUT_DIR, 'latest_checkpoint.pt'))
            print(f"  💾 Checkpoint epoch {epoch}")

        if epoch % cfg.SAMPLE_EVERY == 0:
            _generate_samples(ema.shadow, scheduler, epoch, device)

    # ── Final Save ──
    xm.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'ema_state_dict': ema.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss,
        'train_losses': train_losses,
        'stop_reason': stop_reason,
    }, os.path.join(cfg.OUTPUT_DIR, "final_checkpoint.pt"))

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE — {stop_reason}")
    print(f"  Epochs: {epoch} | Best: {best_loss:.4f} | "
          f"Time: {(time.time() - TRAINING_START_TIME) / 3600:.1f}h")
    print(f"{'='*60}")
    _plot_loss(train_losses)

    # ═══════════════════════════════════════════════════════════════
    # GENERATION
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}\nGENERATION\n{'='*60}")

    ckpt_path = os.path.join(cfg.OUTPUT_DIR, "best_rf_model.pt")
    if os.path.exists(ckpt_path):
        gen_model = ConditionalUNet(
            img_channels=cfg.IMG_CHANNELS, base_channels=cfg.BASE_CHANNELS,
            time_emb_dim=cfg.TIME_EMB_DIM, num_classes=cfg.NUM_CLASSES,
            class_emb_dim=cfg.CLASS_EMB_DIM,
        ).to(device)
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if 'ema_state_dict' in ckpt:
            gen_model.load_state_dict(ckpt['ema_state_dict'])
        else:
            gen_model.load_state_dict(ckpt['model_state_dict'])
        gen_model.eval()
        del ckpt

        gen_sched = RectifiedFlowScheduler(sigma_min=cfg.SIGMA_MIN, device=device)
        gen_plan = {0: ("Benign", cfg.NUM_SYNTHETIC_BENIGN),
                    2: ("Normal", cfg.NUM_SYNTHETIC_NORMAL)}

        for class_idx, (class_name, total) in gen_plan.items():
            if total <= 0:
                continue
            class_dir = os.path.join(cfg.SYNTHETIC_DIR, class_name)
            os.makedirs(class_dir, exist_ok=True)
            print(f"\n🎨 Generating {total} {class_name}...")
            generated = 0
            while generated < total:
                cur_batch = min(16, total - generated)
                labels = torch.full((cur_batch,), class_idx, device=device, dtype=torch.long)
                shape = (cur_batch, cfg.IMG_CHANNELS, cfg.IMG_SIZE, cfg.IMG_SIZE)
                with torch.no_grad():
                    samples = gen_sched.sample_euler(
                        gen_model, shape, labels, num_steps=cfg.EULER_STEPS,
                        cfg_scale=cfg.CFG_SCALE, num_classes=cfg.NUM_CLASSES)
                samples = (samples.clamp(-1, 1) + 1) / 2 * 255
                samples = samples.cpu().numpy().astype(np.uint8)
                for j in range(cur_batch):
                    img_path = os.path.join(class_dir, f"{class_name.lower()}_{generated:04d}.png")
                    Image.fromarray(samples[j, 0]).save(img_path)
                    generated += 1
            print(f"  ✅ {generated} {class_name} saved")
        del gen_model
    else:
        print("⚠️ No best_rf_model.pt — skipping generation")

    # ── Evaluation (CPU for InceptionV3 compat) ──
    try:
        run_evaluation(synthetic_dir=cfg.SYNTHETIC_DIR, device='cpu')
    except Exception as e:
        print(f"⚠️ Evaluation failed: {e}")

    if cfg.USE_WANDB and HAS_WANDB:
        wandb.finish()
    print("\n🎉 All done!")


# ═══════════════════════════════════════════════════════════════════════
# ENTRY POINT — works in both notebooks and scripts
# ═══════════════════════════════════════════════════════════════════════

print("=" * 60)
print("Rectified Flow — TPU v5e (PyTorch/XLA)")
print("=" * 60)
train_and_generate()
