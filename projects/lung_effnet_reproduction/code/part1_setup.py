# %% [markdown]
# # Lung-EffNet: Lung Cancer Classification Using EfficientNet from CT-Scan Images
# **Reproduction Notebook**
#
# | Field | Detail |
# |-------|--------|
# | **Paper** | Raza, R. et al. (2023) |
# | **Journal** | Engineering Applications of Artificial Intelligence, 126, 106902 |
# | **DOI** | [10.1016/j.engappai.2023.106902](https://doi.org/10.1016/j.engappai.2023.106902) |
# | **Dataset** | [IQ-OTH/NCCD Lung Cancer Dataset](https://www.kaggle.com/datasets/hamdallak/the-iqothnccd-lung-cancer-dataset) |
# | **License** | CC BY 4.0 |
# | **Date** | April 2026 (reproduction) |
# | **Python** | 3.10+ |
# | **GPU** | NVIDIA T4 / CUDA 11.8 |
#
# ## Abstract / Objective
# This notebook reproduces and validates the **Lung-EffNet** framework which uses
# transfer learning with **EfficientNet B0–B4** for 3-class lung cancer classification
# (Benign / Malignant / Normal) from CT-scan slices.
#
# **Key claims to validate:**
# 1. EfficientNetB1 achieves **99.10% test accuracy** with data augmentation
# 2. EfficientNetB1 outperforms B0, B2–B4 and other architectures (ResNet50, MobileNet)
# 3. Data augmentation improves generalization on imbalanced medical data
# 4. Transfer learning vastly outperforms training from scratch

# %% [markdown]
# ## 1. Environment & Reproducibility Setup

# %%
import os, sys, json, time, random, warnings, platform, gc
from pathlib import Path
from collections import Counter, defaultdict, OrderedDict
from datetime import datetime

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as tv_models

from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize

try:
    import cv2
except ImportError:
    os.system("pip install opencv-python-headless")
    import cv2

warnings.filterwarnings("ignore")

# ── Reproducibility ──────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print(" ENVIRONMENT INFO")
print("=" * 70)
print(f"  Python     : {sys.version.split()[0]}")
print(f"  PyTorch    : {torch.__version__}")
print(f"  CUDA avail : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU        : {torch.cuda.get_device_name(0)}")
    print(f"  GPU Memory : {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
print(f"  Device     : {DEVICE}")
print(f"  OS         : {platform.system()} {platform.release()}")
print(f"  Seed       : {SEED}")
print(f"  Timestamp  : {datetime.now().isoformat()}")
print("=" * 70)

# %% [markdown]
# ### Hyperparameter Configuration (Table 4 from paper)

# %%
CONFIG = {
    # ── Data ──
    "dataset_name": "IQ-OTH/NCCD Lung Cancer Dataset",
    "num_classes": 3,
    "class_names": ["Benign", "Malignant", "Normal"],
    "img_size": 240,                  # Paper: 240×240×3
    "train_ratio": 0.80,              # Paper: 80:20 split
    "val_from_train": 0.10,           # Paper: 10% of train for validation

    # ── Training ──
    "epochs": 50,                     # Paper: Table 4
    "batch_size": 32,                 # Paper: Table 4
    "optimizer": "Adam",              # Paper: Table 4
    "learning_rate": 1e-3,            # Paper: Table 4
    "lr_decay_factor": 0.3,           # Paper: Table 4
    "lr_patience": 5,                 # Paper: Table 4
    "drop_connect_rate": 0.2,         # Paper: Table 4
    "dropout_rate": 0.5,              # Paper: Section 3.4.2
    "loss_function": "categorical_crossentropy",
    "output_activation": "softmax",   # Paper: Table 4

    # ── Augmentation factors (Paper: Section 3.3) ──
    "aug_factor_benign": 13,
    "aug_factor_malignant": 2,
    "aug_factor_normal": 3,
}

# ── Paths ──
# Adjust these for your environment (local / Kaggle / Colab)
DATASET_ROOT = Path("/kaggle/input/the-iqothnccd-lung-cancer-dataset/The IQ-OTHNCCD lung cancer dataset")
if not DATASET_ROOT.exists():
    DATASET_ROOT = Path("/home/metal/Experiment/data/The IQ-OTHNCCD lung cancer dataset")
if not DATASET_ROOT.exists():
    print("⚠️  Dataset not found. Please set DATASET_ROOT to the correct path.")

RESULTS_DIR = Path("./lung_effnet_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_FOLDERS = ["Bengin cases", "Malignant cases", "Normal cases"]
CLASS_MAP = {"Bengin cases": 0, "Malignant cases": 1, "Normal cases": 2}
LABEL_NAMES = {0: "Benign", 1: "Malignant", 2: "Normal"}

print("\n📋 Hyperparameters (matching Table 4):")
for k, v in CONFIG.items():
    print(f"  {k:30s}: {v}")

# %% [markdown]
# ## 2. Data Loading & Preprocessing
#
# **Dataset**: IQ-OTH/NCCD — 1097 CT-scan slices from 110 patients
#
# | Class | Patients | Samples |
# |-------|----------|---------|
# | Benign | 15 | 120 |
# | Malignant | 40 | 561 |
# | Normal | 55 | 416 |
# | **Total** | **110** | **1097** |
#
# **Preprocessing (Paper Section 3.2):**
# 1. Shuffle images per class
# 2. 80:20 train-test split
# 3. Crop lung contour (remove background)
# 4. Data augmentation (train only)
# 5. Resize to 240×240×3
# 6. Label encode: Benign=0, Malignant=1, Normal=2

# %%
def load_dataset(root):
    """Load all image paths with labels."""
    records = []
    for folder in CLASS_FOLDERS:
        d = root / folder
        if not d.exists():
            print(f"⚠️  Missing: {d}")
            continue
        label = CLASS_MAP[folder]
        files = sorted([f for f in d.iterdir()
                       if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}])
        for f in files:
            records.append({
                'path': str(f),
                'label': label,
                'class_name': folder,
                'filename': f.name
            })
    return records

def split_dataset(records, train_ratio=0.80, seed=SEED):
    """Paper: shuffle + 80:20 split per class (slice-level, as in paper)."""
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for r in records:
        by_class[r['label']].append(r)

    train_recs, test_recs = [], []
    for label, recs in sorted(by_class.items()):
        recs = list(recs)
        rng.shuffle(recs)
        n_train = int(len(recs) * train_ratio)
        for r in recs[:n_train]:
            r['split'] = 'train'
            train_recs.append(r)
        for r in recs[n_train:]:
            r['split'] = 'test'
            test_recs.append(r)

    # 10% of train → validation (Paper: Section 4.3)
    val_recs = []
    final_train = []
    by_class_train = defaultdict(list)
    for r in train_recs:
        by_class_train[r['label']].append(r)
    for label, recs in sorted(by_class_train.items()):
        rng.shuffle(recs)
        n_val = max(1, int(len(recs) * CONFIG['val_from_train']))
        for r in recs[:n_val]:
            r['split'] = 'val'
            val_recs.append(r)
        final_train.extend(recs[n_val:])

    return final_train + val_recs + test_recs


def split_dataset_sequential(records, train_ratio=0.70, val_ratio=0.20):
    """Sequential split: first 70% train, next 20% val, last 10% test.
    NO shuffling — images stay in sorted(os.listdir()) order per class.
    This is the 'honest' split that avoids data leakage."""
    by_class = defaultdict(list)
    for r in records:
        by_class[r['label']].append(r)

    train_recs, val_recs, test_recs = [], [], []
    for label, recs in sorted(by_class.items()):
        # Keep original sorted order — NO shuffle
        n = len(recs)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        for r in recs[:train_end]:
            r['split'] = 'train'
            train_recs.append(r)
        for r in recs[train_end:val_end]:
            r['split'] = 'val'
            val_recs.append(r)
        for r in recs[val_end:]:
            r['split'] = 'test'
            test_recs.append(r)

    return train_recs + val_recs + test_recs

records = load_dataset(DATASET_ROOT)
all_records = split_dataset(records)

print("\n📊 Dataset Split Summary (Paper Table 2):")
print(f"{'Split':<8} {'Benign':>8} {'Malignant':>10} {'Normal':>8} {'Total':>8}")
print("-" * 44)
for sp in ['train', 'val', 'test']:
    sub = [r for r in all_records if r['split'] == sp]
    counts = Counter(r['label'] for r in sub)
    total = sum(counts.values())
    print(f"{sp:<8} {counts.get(0,0):>8} {counts.get(1,0):>10} {counts.get(2,0):>8} {total:>8}")

# %% [markdown]
# ### 2.1 Exploratory Data Analysis (EDA)

# %%
def plot_class_distribution(records):
    """Visualize class imbalance."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Overall distribution
    labels = [LABEL_NAMES[r['label']] for r in records]
    counts = Counter(labels)
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    axes[0].bar(counts.keys(), counts.values(), color=colors, edgecolor='black', alpha=0.85)
    axes[0].set_title("Overall Class Distribution (N=1097)", fontsize=14, fontweight='bold')
    axes[0].set_ylabel("Count", fontsize=12)
    for i, (k, v) in enumerate(counts.items()):
        axes[0].text(i, v + 5, str(v), ha='center', fontweight='bold', fontsize=12)

    # Per-split distribution
    splits = ['train', 'val', 'test']
    x = np.arange(3)
    width = 0.25
    for i, sp in enumerate(splits):
        sub = [r for r in records if r['split'] == sp]
        vals = [sum(1 for r in sub if r['label'] == l) for l in range(3)]
        axes[1].bar(x + i * width, vals, width, label=sp.capitalize(), alpha=0.85)
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(CONFIG['class_names'])
    axes[1].set_title("Per-Split Class Distribution", fontsize=14, fontweight='bold')
    axes[1].set_ylabel("Count", fontsize=12)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "eda_class_distribution.png", dpi=150, bbox_inches='tight')
    plt.savefig(RESULTS_DIR / "eda_class_distribution.svg", bbox_inches='tight')
    plt.show()

plot_class_distribution(all_records)

# %%
def show_sample_images(records, n_per_class=4):
    """Visualize sample images per class (Paper Fig 2)."""
    fig, axes = plt.subplots(3, n_per_class, figsize=(3.5 * n_per_class, 10))
    for row, label in enumerate(range(3)):
        samples = [r for r in records if r['label'] == label][:n_per_class]
        for col, r in enumerate(samples):
            img = Image.open(r['path'])
            axes[row, col].imshow(img, cmap='gray')
            axes[row, col].axis('off')
            if col == 0:
                axes[row, col].set_ylabel(LABEL_NAMES[label], fontsize=14, fontweight='bold')
    fig.suptitle("Sample CT-Scan Images per Class (cf. Paper Fig. 2)", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "eda_sample_images.png", dpi=150, bbox_inches='tight')
    plt.show()

show_sample_images(all_records)

# %%
def analyze_image_properties(records, max_samples=200):
    """Check image sizes, channels, pixel statistics."""
    sizes, means, stds = [], [], []
    for r in random.sample(records, min(max_samples, len(records))):
        img = np.array(Image.open(r['path']))
        sizes.append(img.shape[:2])
        means.append(img.mean())
        stds.append(img.std())

    print("📐 Image Property Statistics:")
    heights, widths = zip(*sizes)
    print(f"  Height : min={min(heights)}, max={max(heights)}, mean={np.mean(heights):.0f}")
    print(f"  Width  : min={min(widths)}, max={max(widths)}, mean={np.mean(widths):.0f}")
    print(f"  Mean px: {np.mean(means):.1f} ± {np.std(means):.1f}")
    print(f"  Std px : {np.mean(stds):.1f} ± {np.std(stds):.1f}")

analyze_image_properties(all_records)

# %% [markdown]
# ### 2.2 Preprocessing Pipeline
#
# **Paper Section 3.2:**
# - Crop extreme points of biggest lung contour (remove background)
# - Resize to 240×240×3
# - Pixel values 0–255 (EfficientNet has built-in normalization layer)
#
# **Data Augmentation (Paper Section 3.3, Table 3):**
# - Horizontal flip, Rotation, Brightness, Zoom, Width/Height shift
# - Benign ×13, Malignant ×2, Normal ×3 → ~balanced training set

# %%
def crop_lung_contour(image_np):
    """Crop unwanted regions from CT scan (Paper Fig 3).
    Finds largest contour and crops to bounding box."""
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np.copy()

    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        pad = 5
        x, y = max(0, x - pad), max(0, y - pad)
        w, h = min(image_np.shape[1] - x, w + 2*pad), min(image_np.shape[0] - y, h + 2*pad)
        return image_np[y:y+h, x:x+w]
    return image_np


class LungCropTransform:
    """Crop lung contour as preprocessing step."""
    def __call__(self, image):
        img_np = np.array(image)
        cropped = crop_lung_contour(img_np)
        return Image.fromarray(cropped)


def get_train_transforms(img_size):
    """Paper augmentation: flip, rotation, brightness, zoom, shifts."""
    return T.Compose([
        LungCropTransform(),
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.2),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        T.ToTensor(),
        # EfficientNet has built-in normalization — input 0-255 range
        # But PyTorch EfficientNet expects ImageNet norm
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_eval_transforms(img_size):
    return T.Compose([
        LungCropTransform(),
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# %%  Show preprocessing pipeline
def visualize_preprocessing(records):
    """Show raw → cropped → augmented pipeline (Paper Fig 3, 4)."""
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    cols = ['Original', 'Cropped', 'Augmented 1', 'Augmented 2']
    for i, label in enumerate(range(3)):
        r = [r for r in records if r['label'] == label][0]
        img = Image.open(r['path']).convert('RGB')
        img_np = np.array(img)

        # Original
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(cols[0] if i == 0 else '', fontsize=12)
        axes[i, 0].set_ylabel(LABEL_NAMES[label], fontsize=13, fontweight='bold')

        # Cropped
        cropped = crop_lung_contour(img_np)
        axes[i, 1].imshow(cropped)
        axes[i, 1].set_title(cols[1] if i == 0 else '', fontsize=12)

        # Augmented
        aug_tf = get_train_transforms(CONFIG['img_size'])
        for j in range(2):
            aug = aug_tf(img)
            # Denormalize for display
            aug_np = aug.permute(1, 2, 0).numpy()
            aug_np = aug_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            aug_np = np.clip(aug_np, 0, 1)
            axes[i, 2 + j].imshow(aug_np)
            axes[i, 2 + j].set_title(cols[2 + j] if i == 0 else '', fontsize=12)

        for ax in axes[i]:
            ax.axis('off')

    fig.suptitle("Preprocessing Pipeline (cf. Paper Figs 3 & 4)", fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "preprocessing_pipeline.png", dpi=150, bbox_inches='tight')
    plt.show()

visualize_preprocessing(all_records)

# %% [markdown]
# ### 2.3 Dataset & DataLoader with Augmentation Balancing

# %%
class LungCTDataset(Dataset):
    """CT scan dataset with optional oversampling for class balance."""
    def __init__(self, records, transform, oversample_factors=None):
        self.transform = transform
        if oversample_factors:
            self.records = []
            for r in records:
                factor = oversample_factors.get(r['label'], 1)
                self.records.extend([r] * factor)
            print(f"  Oversampled: {len(records)} → {len(self.records)}")
        else:
            self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        img = Image.open(r['path']).convert('RGB')
        img = self.transform(img)
        return img, r['label']


def make_dataloaders(records, img_size, batch_size, augment=True):
    """Create train/val/test dataloaders."""
    train_recs = [r for r in records if r['split'] == 'train']
    val_recs = [r for r in records if r['split'] == 'val']
    test_recs = [r for r in records if r['split'] == 'test']

    train_tf = get_train_transforms(img_size) if augment else get_eval_transforms(img_size)
    eval_tf = get_eval_transforms(img_size)

    # Paper augmentation factors for class balancing
    aug_factors = {0: CONFIG['aug_factor_benign'],
                   1: CONFIG['aug_factor_malignant'],
                   2: CONFIG['aug_factor_normal']} if augment else None

    train_ds = LungCTDataset(train_recs, train_tf, oversample_factors=aug_factors)
    val_ds = LungCTDataset(val_recs, eval_tf)
    test_ds = LungCTDataset(test_recs, eval_tf)

    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=2, pin_memory=True, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)
    test_ld = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                         num_workers=2, pin_memory=True)

    print(f"\n📦 DataLoader sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    # Show post-augmentation class balance
    if augment:
        counts = Counter(r['label'] for r in train_ds.records)
        print(f"  After augmentation: {dict(counts)} (cf. Paper Table 3)")
    return train_ld, val_ld, test_ld

print("✅ Part 1 (Setup + Data) complete.")
