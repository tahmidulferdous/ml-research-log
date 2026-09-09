
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 1 — IMPORTS · SEED · CONFIG · AUGMENTATION · DATASET · LOADERS   ║
# ║  Dataset : IQ-OTH/NCCD Lung Cancer CT                                      ║
# ║  Run this cell once — then Section 2 is ready to train.                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


# ═══════════════════════════════════════════════════════════════════════════════
# 1.1  IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════

import os
import random
import warnings
from collections import Counter
warnings.filterwarnings('ignore')

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import albumentations as A
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

print("✅ All libraries imported successfully!")


# ═══════════════════════════════════════════════════════════════════════════════
# 1.2  SEED & DEVICE
# ═══════════════════════════════════════════════════════════════════════════════

SEED = 42

def seed_everything(seed: int = SEED) -> None:
    """Ensure full reproducibility across all random sources."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = './saved_models'
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"🖥️  Device : {DEVICE}")
if torch.cuda.is_available():
    print(f"🔥 GPU    : {torch.cuda.get_device_name(0)}")
    print(f"✅ FP16   : Supported")
else:
    print("⚠️  No GPU detected — training will be slow on CPU.")


# ═══════════════════════════════════════════════════════════════════════════════
# 1.3  DATASET CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

class Config:
    # ── Paths ─────────────────────────────────────────────────────────────────
    DATASET_ROOT = (
        '/kaggle/input/datasets/hamdallak/'
        'the-iqothnccd-lung-cancer-dataset/'
        'The IQ-OTHNCCD lung cancer dataset'
    )

    # NOTE: "Bengin" is a typo in the original dataset folder name — keep as-is.
    CLASSES      = ["Bengin cases", "Malignant cases", "Normal cases"]
    CLASS_TO_IDX = {"Bengin cases": 0, "Malignant cases": 1, "Normal cases": 2}
    IDX_TO_CLASS = {0: "Benign",    1: "Malignant",           2: "Normal"}
    NUM_CLASSES  = 3

    # ── Image / loader ────────────────────────────────────────────────────────
    IMG_SIZE   = 224
    BATCH_SIZE = 64
    SEED       = 42

    # ── Augmentation — Geometric (TWEAK THESE) ────────────────────────────────
    HFLIP_PROB     = 0.5    # horizontal flip probability
    ROTATE_LIMIT   = 15     # ± degrees
    ROTATE_PROB    = 0.5
    ELASTIC_ALPHA  = 120    # deformation intensity
    ELASTIC_SIGMA  = 6      # smoothness
    ELASTIC_PROB   = 0.3

    # ── Augmentation — Photometric (TWEAK THESE) ──────────────────────────────
    CLAHE_CLIP        = 2.0     # contrast limit
    CLAHE_PROB        = 0.3
    BRIGHTNESS_LIMIT  = 0.1     # ± 10 %
    CONTRAST_LIMIT    = 0.1     # ± 10 %
    BC_PROB           = 0.3
    NOISE_VAR         = (10, 50)
    NOISE_PROB        = 0.2

cfg = Config()
print("✅ Config ready")


# ═══════════════════════════════════════════════════════════════════════════════
# 1.4  AUGMENTATION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def get_train_transform(config: Config = cfg) -> A.Compose:
    """Medical-imaging augmentation pipeline for the training set."""
    return A.Compose([
        A.Resize(config.IMG_SIZE, config.IMG_SIZE),
        # Geometric
        A.HorizontalFlip(p=config.HFLIP_PROB),
        A.Rotate(limit=config.ROTATE_LIMIT, p=config.ROTATE_PROB, border_mode=0),
        A.ElasticTransform(
            alpha=config.ELASTIC_ALPHA, sigma=config.ELASTIC_SIGMA,
            p=config.ELASTIC_PROB, border_mode=0),
        # Photometric
        A.CLAHE(clip_limit=config.CLAHE_CLIP, tile_grid_size=(8, 8), p=config.CLAHE_PROB),
        A.RandomBrightnessContrast(
            brightness_limit=config.BRIGHTNESS_LIMIT,
            contrast_limit=config.CONTRAST_LIMIT,
            p=config.BC_PROB),
        A.GaussNoise(var_limit=config.NOISE_VAR, p=config.NOISE_PROB),
    ])


def get_val_transform(config: Config = cfg) -> A.Compose:
    """Validation / test  — resize only, no augmentation."""
    return A.Compose([A.Resize(config.IMG_SIZE, config.IMG_SIZE)])


train_transform = get_train_transform()
val_transform   = get_val_transform()
print("✅ Augmentation transforms ready")


# ═══════════════════════════════════════════════════════════════════════════════
# 1.5  VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def visualize_augmentation_grid(image_path, num_augmentations: int = 8):
    """
    3 × 3 grid: original (top-left) + up to 8 random augmentations.
    Accepts a file path string or a numpy array.
    Returns the matplotlib Figure.
    """
    img = (np.array(Image.open(image_path).convert('L'))
           if isinstance(image_path, str) else image_path)

    imgs = [val_transform(image=img)['image']]          # original (resized)
    for _ in range(num_augmentations):
        imgs.append(train_transform(image=img)['image'])

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle('Original + Random Augmentations', fontsize=16, y=0.98, weight='bold')

    for i, aug_img in enumerate(imgs[:9]):
        row, col = divmod(i, 3)
        axes[row, col].imshow(aug_img, cmap='gray', vmin=0, vmax=255)
        axes[row, col].set_title(
            'Original' if i == 0 else f'Aug {i}',
            fontsize=12 if i == 0 else 10,
            color='green' if i == 0 else 'black',
            weight='bold' if i == 0 else 'normal',
        )
        axes[row, col].axis('off')

    plt.tight_layout()
    return fig


def visualize_class_augmentations(image_paths: list):
    """
    3 rows (one per class) × 4 cols (original + 3 augmentations).
    Args:
        image_paths: list of 3 file paths — [benign, malignant, normal]
    Returns the matplotlib Figure.
    """
    class_names = ['Benign', 'Malignant', 'Normal']
    fig, axes   = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('Augmentation Examples per Class', fontsize=16, weight='bold')

    for row, (img_path, class_name) in enumerate(zip(image_paths, class_names)):
        img         = np.array(Image.open(img_path).convert('L'))
        img_resized = val_transform(image=img)['image']

        axes[row, 0].imshow(img_resized, cmap='gray', vmin=0, vmax=255)
        axes[row, 0].set_title(f'{class_name} — Original', fontsize=11, weight='bold')
        axes[row, 0].axis('off')

        for col in range(1, 4):
            aug = train_transform(image=img)['image']
            axes[row, col].imshow(aug, cmap='gray', vmin=0, vmax=255)
            axes[row, col].set_title(f'Aug {col}', fontsize=10)
            axes[row, col].axis('off')

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 1.6  DATASET CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class CTDataset(Dataset):
    """
    CT-scan dataset with online (in-memory) augmentation.
    No extra disk space needed — every __getitem__ call augments on the fly.
    Output tensor: float32, shape (1, H, W), values in [-1, 1].
    """

    def __init__(self, image_paths: list, labels: list, transform: A.Compose):
        self.image_paths = image_paths
        self.labels      = labels
        self.transform   = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img   = np.array(Image.open(self.image_paths[idx]).convert('L'))
        img   = self.transform(image=img)['image']
        img   = (img.astype(np.float32) / 255.0) * 2.0 - 1.0   # [0,255] → [-1,1]
        img   = torch.FloatTensor(img).unsqueeze(0)              # (1, H, W)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label


# ═══════════════════════════════════════════════════════════════════════════════
# 1.7  DATA LOADING & DATALOADER BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def load_paths(root_dir: str):
    """Walk dataset root and return (paths, labels) lists."""
    _EXT = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    paths, labels = [], []
    for class_name in cfg.CLASSES:
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"⚠️  WARNING: {class_dir} not found — skipping.")
            continue
        for fname in sorted(os.listdir(class_dir)):
            if os.path.splitext(fname)[1].lower() in _EXT:
                paths.append(os.path.join(class_dir, fname))
                labels.append(cfg.CLASS_TO_IDX[class_name])
    return paths, labels


def build_dataloaders():
    """
    Stratified 70 / 15 / 15 split → DataLoaders with:
      • WeightedRandomSampler  (balanced training batches)
      • Online augmentation    (no extra disk usage)
      • class_weights tensor   (for CrossEntropyLoss)

    Returns:
        train_loader, val_loader, test_loader,
        class_weights (Tensor),
        (train_paths, train_labels)
    """
    print("\n" + "=" * 60)
    print("📊 BUILDING DATALOADERS")
    print("=" * 60)

    all_paths, all_labels = load_paths(cfg.DATASET_ROOT)
    print(f"\nTotal images : {len(all_paths)}")
    for idx in sorted(set(all_labels)):
        print(f"  {cfg.IDX_TO_CLASS[idx]:<12}: {all_labels.count(idx)} images")

    # ── Stratified split ──────────────────────────────────────────────────────
    train_p, temp_p, train_l, temp_l = train_test_split(
        all_paths, all_labels, test_size=0.30,
        stratify=all_labels, random_state=cfg.SEED)
    val_p, test_p, val_l, test_l = train_test_split(
        temp_p, temp_l, test_size=0.50,
        stratify=temp_l, random_state=cfg.SEED)

    print(f"\n✂️  Split  → Train: {len(train_p)} | Val: {len(val_p)} | Test: {len(test_p)}")
    print(f"Train dist : {dict(Counter(train_l))}")

    # ── WeightedRandomSampler ─────────────────────────────────────────────────
    counts           = Counter(train_l)
    w_per_class      = {c: 1.0 / n for c, n in counts.items()}
    sample_weights   = [w_per_class[l] for l in train_l]
    sampler          = WeightedRandomSampler(
                           sample_weights, num_samples=len(train_l), replacement=True)

    # ── Class-weighted loss tensor ────────────────────────────────────────────
    total         = sum(counts.values())
    class_weights = torch.ones(cfg.NUM_CLASSES)
    for i in range(cfg.NUM_CLASSES):
        if counts.get(i, 0) > 0:
            class_weights[i] = total / (cfg.NUM_CLASSES * counts[i])

    print(f"\n🎯 Loss weights:")
    for i in range(cfg.NUM_CLASSES):
        tag = "" if i in counts else " ⚠️ MISSING"
        print(f"  {cfg.IDX_TO_CLASS[i]:<12}: {class_weights[i]:.3f}×{tag}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = CTDataset(train_p, train_l, train_transform)
    val_ds   = CTDataset(val_p,   val_l,   val_transform)
    test_ds  = CTDataset(test_p,  test_l,  val_transform)

    # ── Loaders ───────────────────────────────────────────────────────────────
    _loader_kwargs = dict(num_workers=2, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE,
                              sampler=sampler, drop_last=True, **_loader_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE,
                              shuffle=False, **_loader_kwargs)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.BATCH_SIZE,
                              shuffle=False, **_loader_kwargs)

    return train_loader, val_loader, test_loader, class_weights, (train_p, train_l)


# ═══════════════════════════════════════════════════════════════════════════════
# 1.8  RUN — BUILD LOADERS + VISUALIZE + VERIFY
# ═══════════════════════════════════════════════════════════════════════════════

train_loader, val_loader, test_loader, class_weights, (train_paths, train_labels) = (
    build_dataloaders()
)

# ── Visualize augmentations ───────────────────────────────────────────────────
sample_imgs = []
for cls_idx in sorted(set(train_labels)):
    cls_paths = [p for p, l in zip(train_paths, train_labels) if l == cls_idx]
    if cls_paths:
        sample_imgs.append(cls_paths[0])

if len(sample_imgs) >= 3:
    print("\n🎨 Class augmentation grid (Original + 3 per class)...")
    visualize_class_augmentations(sample_imgs)
    plt.show()

if sample_imgs:
    print("🎨 Detailed augmentation grid (Original + 8 variations)...")
    visualize_augmentation_grid(sample_imgs[0])
    plt.show()

# ── Batch sanity check ────────────────────────────────────────────────────────
batch_imgs, batch_lbls = next(iter(train_loader))
print("\n" + "=" * 60)
print("✅ BATCH SANITY CHECK")
print("=" * 60)
print(f"  Shape      : {tuple(batch_imgs.shape)}")
print(f"  Pixel range: [{batch_imgs.min():.2f}, {batch_imgs.max():.2f}]")
print(f"  Class dist : {dict(Counter(batch_lbls.numpy().tolist()))}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("🎉 SECTION 1 COMPLETE — READY FOR SECTION 2")
print("=" * 60)
print("  ✅ train_loader  — balanced, augmented")
print("  ✅ val_loader    — clean, resize-only")
print("  ✅ test_loader   — clean, resize-only")
print("  ✅ class_weights — for CrossEntropyLoss")
print(f"\n  ➡️  Next: pass these to Section 2  (DEVICE={DEVICE})")
