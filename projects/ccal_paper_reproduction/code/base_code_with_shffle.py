import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler  # kept for reference, not used
import torchvision.transforms as transforms
from pathlib import Path
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
import cv2
import seaborn as sns
from sklearn.model_selection import train_test_split
import wandb
import random
import time

# Enable Flash Attention (PyTorch 2.0+)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)

# ===================== PAPER CONFIG =====================
# Paper: lr=0.001, Adam, batch=32, 50 epochs, dropout=0.5
# Paper: 80/20 image-level random split (data leakage)
# Paper: CLAHE + resize 224x224 + /255 normalization
# Paper: offline augmentation to ~10,000 images
# ========================================================

config = {
    'data_dir': '/kaggle/input/datasets/hamdallak/the-iqothnccd-lung-cancer-dataset/The IQ-OTHNCCD lung cancer dataset',
    'batch_size': 32,
    'learning_rate': 0.0001,       # Paper: 0.001 (NOT 0.00001)
    'epochs': 50,
    'num_heads': 4,
    'key_dim': 16,
    'dropout': 0.5,
    'target_total_images': 10000, # Paper: augment to ~10k
    'test_size': 0.20,            # Paper: 80/20 split
    'image_size': 224,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 42,
}


# ===================== REPRODUCIBILITY =====================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(config['seed'])


# ===================== CLAHE TRANSFORM =====================
class CLAHETransform:
    """Paper: CLAHE with 4 tiles, clip limit applied."""
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            image = clahe.apply(image)
        return Image.fromarray(image)


# ===================== DATASET =====================
class LungDataset(Dataset):
    def __init__(self, samples, transform=None, cache_in_ram=True):
        self.samples = samples
        self.transform = transform
        self.cache = {}
        # Pre-load all images into RAM (dataset is small ~1097 images)
        if cache_in_ram:
            print(f"  Caching {len(samples)} images in RAM...")
            for i, (img_path, _) in enumerate(samples):
                if img_path not in self.cache:
                    self.cache[img_path] = Image.open(img_path).convert('RGB')
            print(f"  Cached {len(self.cache)} unique images.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        if img_path in self.cache:
            image = self.cache[img_path].copy()  # copy so transforms don't mutate cache
        else:
            image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_class_counts(self):
        counts = {}
        for _, label in self.samples:
            counts[label] = counts.get(label, 0) + 1
        return counts


# ===================== PAPER PREPROCESSING =====================
def collect_all_images(data_dir):
    """Collect all images from the IQ-OTH/NCCD dataset directory."""
    classes = {'Bengin cases': 0, 'Malignant cases': 1, 'Normal cases': 2}
    samples = []

    # Handle both flat and pre-split directory structures
    for class_name, label in classes.items():
        # Try flat structure first (original dataset)
        class_path = os.path.join(data_dir, class_name)
        if os.path.exists(class_path):
            for img_name in sorted(os.listdir(class_path)):
                if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                    samples.append((os.path.join(class_path, img_name), label))
        else:
            # Try pre-split structure (train/test subdirs)
            for split in ['train', 'test', 'Train', 'Test']:
                split_class = os.path.join(data_dir, split, class_name)
                if os.path.exists(split_class):
                    for img_name in sorted(os.listdir(split_class)):
                        if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                            samples.append((os.path.join(split_class, img_name), label))

    return samples, classes


def offline_augment(train_samples, target_total, image_size=224):
    """
    Paper method: augment training set to ~10,000 images maintaining
    class proportions. Augmentations: rotation, width/height shift,
    shear, zoom, horizontal flip, fill mode.
    """
    from torchvision.transforms import functional as TF

    augment_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(20),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),   # width/height shift
            scale=(0.9, 1.1),       # zoom
            shear=10,               # shear
            fill=0
        ),
        transforms.RandomHorizontalFlip(p=0.5),
    ])

    # Count per class
    class_counts = {}
    class_samples = {}
    for path, label in train_samples:
        class_counts[label] = class_counts.get(label, 0) + 1
        class_samples.setdefault(label, []).append((path, label))

    total_orig = len(train_samples)
    augmented = list(train_samples)  # keep originals

    # Calculate how many augmented images per class (proportional)
    for label, samples in class_samples.items():
        proportion = len(samples) / total_orig
        target_for_class = int(target_total * proportion)
        num_to_generate = max(0, target_for_class - len(samples))

        for i in range(num_to_generate):
            src_path, src_label = samples[i % len(samples)]
            # We store the original path + a flag; augmentation applied in Dataset
            augmented.append((src_path, src_label))

    random.shuffle(augmented)
    return augmented


def get_paper_transforms(image_size=224):
    """
    Paper preprocessing:
    1. CLAHE
    2. Resize to 224x224
    3. Convert to tensor (/255 normalization, scale 0-1)
    Paper does NOT use ImageNet normalization.
    """
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(20),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=10,
            fill=0
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8)),
        transforms.ToTensor(),  # scales to [0, 1] i.e. /255
    ])

    test_transform = transforms.Compose([
        CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8)),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),  # scales to [0, 1] i.e. /255
    ])

    return train_transform, test_transform


def prepare_paper_data(data_dir, test_size=0.20, target_total=10000, batch_size=32, image_size=224):
    """
    Paper method: 
    1. Collect ALL images
    2. 80/20 IMAGE-LEVEL random split (this is the paper's leaky method)
    3. Augment training to ~10k images
    4. Apply CLAHE + resize + /255
    """
    all_samples, classes = collect_all_images(data_dir)
    labels = [s[1] for s in all_samples]

    print(f"Total images found: {len(all_samples)}")
    for cls_name, cls_id in classes.items():
        cnt = sum(1 for _, l in all_samples if l == cls_id)
        print(f"  {cls_name}: {cnt}")

    # Paper: 80/20 image-level random split (data leakage - as paper describes)
    train_samples, test_samples = train_test_split(
        all_samples, test_size=test_size, random_state=config['seed'],
        stratify=labels
    )

    print(f"\nAfter 80/20 split:")
    print(f"  Train: {len(train_samples)}")
    print(f"  Test:  {len(test_samples)}")

    # Paper: augment training to ~10,000 images
    train_augmented = offline_augment(train_samples, target_total, image_size)
    print(f"  Train after augmentation: {len(train_augmented)}")

    train_transform, test_transform = get_paper_transforms(image_size)

    train_dataset = LungDataset(train_augmented, train_transform)
    test_dataset = LungDataset(test_samples, test_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=4
    )

    return train_loader, test_loader


# ======================== Model Architecture ========================

class CNNModel1(nn.Module):
    """
    Deeper CNN branch (4 conv blocks).
    Paper: Conv(32)->Pool -> Conv(64)->Pool -> Conv(128)->Conv(128)->Pool -> Conv(256) -> GlobalMaxPool
    """
    def __init__(self, in_channels=3):
        super(CNNModel1, self).__init__()
        self.features = nn.Sequential(
            # Block 1: 224 -> 112
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 2: 112 -> 56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 3: 56 -> 28 (additional Conv2D with 128 filters)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 4: final conv
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        x = self.features(x)               # (B, 256, 28, 28)
        x = self.global_max_pool(x)         # (B, 256, 1, 1)
        return x.view(x.size(0), -1)        # (B, 256)


class CNNModel2(nn.Module):
    """
    Shallower CNN branch (2 conv blocks).
    Paper: Conv(16)->Pool -> Conv(32)->Pool -> GlobalMaxPool
    """
    def __init__(self, in_channels=3):
        super(CNNModel2, self).__init__()
        self.features = nn.Sequential(
            # Block 1: 224 -> 112
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 2: 112 -> 56
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        x = self.features(x)               # (B, 32, 56, 56)
        x = self.global_max_pool(x)         # (B, 32, 1, 1)
        return x.view(x.size(0), -1)        # (B, 32)


class CCALModel(nn.Module):
    """
    Concatenated CNN Attention Lung (CCAL) Model.

    Pipeline:
        CNN_Model1 ──┐
                      ├── Concatenate -> Reshape -> MHA_1 -> MHA_2 -> MLP Block -> Classification
        CNN_Model2 ──┘

    MHA: num_heads=4, key_dim=16, value_dim=64
    MLP: GlobalAvgPool1D -> BN -> Dense(256)+GELU -> BN -> Dense(3)+Softmax
    """
    def __init__(self, num_classes=3, num_heads=4, key_dim=16, value_dim=64):
        super(CCALModel, self).__init__()

        # ---- Two CNN Branches ----
        self.cnn1 = CNNModel1()    # output: (B, 256)
        self.cnn2 = CNNModel2()    # output: (B, 32)

        concat_dim = 256 + 32      # 288

        # ---- Reshape & Project for MHA ----
        self.embed_dim = num_heads * key_dim  # 64
        self.num_heads = num_heads
        self.num_tokens = num_heads  # 4 tokens

        # Project concatenated features to (num_tokens * embed_dim) = 4 * 64 = 256
        self.feature_projection = nn.Linear(concat_dim, self.num_tokens * self.embed_dim)

        # ---- Two MHA Layers ----
        self.mha1 = nn.MultiheadAttention(
            embed_dim=self.embed_dim, num_heads=num_heads,
            dropout=0.1, batch_first=True
        )
        self.mha2 = nn.MultiheadAttention(
            embed_dim=self.embed_dim, num_heads=num_heads,
            dropout=0.1, batch_first=True
        )

        # ---- MLP Block ----
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.bn1 = nn.BatchNorm1d(self.embed_dim)
        self.dense1 = nn.Linear(self.embed_dim, 256)
        self.gelu = nn.GELU()
        self.bn2 = nn.BatchNorm1d(256)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        # ---- CNN Feature Extraction ----
        feat1 = self.cnn1(x)                    # (B, 256)
        feat2 = self.cnn2(x)                    # (B, 32)

        # ---- Concatenation ----
        concat = torch.cat([feat1, feat2], dim=1)  # (B, 288)

        # ---- Reshape for MHA ----
        proj = self.feature_projection(concat)     # (B, 256)
        tokens = proj.view(
            proj.size(0), self.num_tokens, self.embed_dim
        )                                          # (B, 4, 64)

        # ---- Multi-Head Attention ----
        attn_out1, attn_weights1 = self.mha1(tokens, tokens, tokens)
        attn_out2, attn_weights2 = self.mha2(attn_out1, attn_out1, attn_out1)

        # ---- MLP Block ----
        x = attn_out2.transpose(1, 2)             # (B, 64, 4)
        x = self.global_avg_pool(x).squeeze(-1)   # (B, 64)
        x = self.bn1(x)
        x = self.dense1(x)                        # (B, 256)
        x = self.gelu(x)
        x = self.bn2(x)
        logits = self.classifier(x)               # (B, 3)

        return logits, (attn_weights1, attn_weights2)


# ================= Training & Testing (Paper Method + Enhanced W&B) ==================

def train_model(model, train_loader, test_loader, num_epochs=50, learning_rate=0.001, device='cuda'):
    """Paper: Adam optimizer, lr=0.001, CrossEntropyLoss, 50 epochs."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    model.to(device)

    best_val_acc = 0.0
    best_epoch = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # === Training ===
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for data, targets in train_loader:
            data, targets = data.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            outputs, _ = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100. * correct / total

        # === Validation/Test ===
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                outputs, _ = model(data)
                loss = criterion(outputs, targets)
                probs = F.softmax(outputs, dim=1)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        avg_val_loss = val_loss / len(test_loader)
        val_acc = 100. * val_correct / val_total
        epoch_time = time.time() - epoch_start

        # Per-epoch metrics
        val_precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
        val_recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
        val_f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)

        # Store history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': avg_val_loss,
            }, 'best_model.pth')
            print(f'  ✓ New best model! Val Acc: {val_acc:.2f}%')

        # Step scheduler based on val loss
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < old_lr:
            print(f'  ↓ LR reduced: {old_lr:.6f} → {current_lr:.6f}')

        marker = ' ✓' if val_acc >= best_val_acc else ''
        print(f'Epoch {epoch+1:2d}/{num_epochs}: TrL={avg_train_loss:.4f} TrA={train_acc:.2f}% | VL={avg_val_loss:.4f} VA={val_acc:.2f}% | LR={current_lr:.6f} | {epoch_time:.1f}s{marker}')

        # === W&B Logging (enhanced) ===
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": avg_train_loss,
            "train/accuracy": train_acc,
            "val/loss": avg_val_loss,
            "val/accuracy": val_acc,
            "val/precision": val_precision * 100,
            "val/recall": val_recall * 100,
            "val/f1": val_f1 * 100,
            "best_val_acc": best_val_acc,
            "epoch_time_sec": epoch_time,
            "learning_rate": current_lr,
        })

    print(f'\nTraining complete! Best: epoch {best_epoch}, val acc {best_val_acc:.2f}%')
    return history


def load_best_model(model, device='cuda'):
    checkpoint = torch.load('best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Loaded best model from epoch {checkpoint["epoch"]}, val acc: {checkpoint["val_acc"]:.2f}%')
    return model


def evaluate_model(model, test_loader, device='cuda'):
    model.eval()
    all_preds, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            outputs, _ = model(data)
            probs = F.softmax(outputs, dim=1)

            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='weighted')
    recall = recall_score(all_targets, all_preds, average='weighted')
    f1 = f1_score(all_targets, all_preds, average='weighted')
    auc_score = roc_auc_score(all_targets, np.array(all_probs), multi_class='ovr', average='weighted')

    # Per-class metrics
    class_names = ['Benign', 'Malignant', 'Normal']
    per_class_precision = precision_score(all_targets, all_preds, average=None)
    per_class_recall = recall_score(all_targets, all_preds, average=None)
    per_class_f1 = f1_score(all_targets, all_preds, average=None)

    print(f'\n{"="*50}')
    print(f'TEST RESULTS (Paper Method - Image-Level Split)')
    print(f'{"="*50}')
    print(f'Accuracy:  {accuracy*100:.2f}%')
    print(f'Precision: {precision*100:.2f}%')
    print(f'Recall:    {recall*100:.2f}%')
    print(f'F1-Score:  {f1*100:.2f}%')
    print(f'AUC:       {auc_score*100:.2f}%')

    # Classification Report
    print(f'\n{classification_report(all_targets, all_preds, target_names=class_names)}')

    # W&B: log final test metrics
    wandb.log({
        "test/accuracy": accuracy * 100,
        "test/precision": precision * 100,
        "test/recall": recall * 100,
        "test/f1": f1 * 100,
        "test/auc": auc_score * 100,
    })

    # W&B: per-class metrics
    for i, name in enumerate(class_names):
        wandb.log({
            f"test/precision_{name}": per_class_precision[i] * 100,
            f"test/recall_{name}": per_class_recall[i] * 100,
            f"test/f1_{name}": per_class_f1[i] * 100,
        })

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    plt.tight_layout()
    wandb.log({"test/confusion_matrix": wandb.Image(fig)})
    plt.show()
    plt.close()

    # W&B summary table
    summary_table = wandb.Table(
        columns=["Metric", "Value"],
        data=[
            ["Accuracy", f"{accuracy*100:.2f}%"],
            ["Precision", f"{precision*100:.2f}%"],
            ["Recall", f"{recall*100:.2f}%"],
            ["F1-Score", f"{f1*100:.2f}%"],
            ["AUC", f"{auc_score*100:.2f}%"],
        ]
    )
    wandb.log({"test/results_table": summary_table})

    return accuracy, precision, recall, f1, auc_score


def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_title('Loss Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    best_idx = np.argmax(history['val_acc'])
    ax2.scatter(best_idx + 1, history['val_acc'][best_idx],
                color='red', s=100, marker='*', zorder=5,
                label=f'Best: {history["val_acc"][best_idx]:.2f}%')
    ax2.set_title('Accuracy Curves')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    wandb.log({"training/curves": wandb.Image(fig)})
    plt.show()
    plt.close()


# ===================== XAI: Grad-CAM + Attention Visualization =====================
def get_gradcam(model, image_tensor, target_class, device='cuda'):
    """Grad-CAM on CNN_Model1's last conv layer (Paper XAI approach)."""
    model.eval()
    image_tensor = image_tensor.unsqueeze(0).to(device).requires_grad_(False)

    # Hook to capture activations and gradients from CNN1's last conv
    activations = {}
    gradients = {}

    def forward_hook(module, inp, out):
        activations['value'] = out.detach()

    def backward_hook(module, grad_in, grad_out):
        gradients['value'] = grad_out[0].detach()

    # Hook to last Conv2d in CNN1 (features[-2] is the Conv2d before final ReLU)
    target_layer = model.cnn1.features[-2]
    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_full_backward_hook(backward_hook)

    # Forward + backward
    input_tensor = image_tensor.unsqueeze(0).to(device) if image_tensor.dim() == 3 else image_tensor.to(device)
    input_tensor = input_tensor.detach().requires_grad_(True)
    outputs, (attn_w1, attn_w2) = model(input_tensor)
    model.zero_grad()
    outputs[0, target_class].backward()

    # Compute Grad-CAM
    weights = gradients['value'].mean(dim=[2, 3], keepdim=True)
    cam = (weights * activations['value']).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
    cam = cam.squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    handle_fwd.remove()
    handle_bwd.remove()

    return cam, attn_w1.detach().cpu(), attn_w2.detach().cpu()


def visualize_xai(model, test_loader, device='cuda'):
    """Visualize Grad-CAM heatmaps + attention weights for sample test images."""
    model.eval()
    class_names = ['Benign', 'Malignant', 'Normal']

    # Collect one sample per class
    samples = {}
    for images, labels in test_loader:
        for i in range(images.size(0)):
            lbl = labels[i].item()
            if lbl not in samples:
                samples[lbl] = images[i]
        if len(samples) >= 3:
            break

    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    col_titles = ['Original', 'Grad-CAM Heatmap', 'Grad-CAM Overlay']

    last_attn_w1, last_attn_w2 = None, None
    for row, (lbl, img_tensor) in enumerate(sorted(samples.items())):
        with torch.enable_grad():
            cam, attn_w1, attn_w2 = get_gradcam(model, img_tensor, lbl, device)
            last_attn_w1, last_attn_w2 = attn_w1, attn_w2

        # Original image
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np, 0, 1)
        axes[row, 0].imshow(img_np)
        axes[row, 0].set_title(f'{col_titles[0]}\nTrue: {class_names[lbl]}', fontsize=11)
        axes[row, 0].axis('off')

        # Grad-CAM heatmap
        axes[row, 1].imshow(cam, cmap='jet')
        axes[row, 1].set_title(col_titles[1], fontsize=11)
        axes[row, 1].axis('off')

        # Overlay
        heatmap = plt.cm.jet(cam)[:, :, :3]
        overlay = 0.5 * img_np + 0.5 * heatmap
        overlay = np.clip(overlay, 0, 1)
        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title(col_titles[2], fontsize=11)
        axes[row, 2].axis('off')

    fig.suptitle('XAI: Grad-CAM Visualization (CNN1 Last Conv)', fontsize=15, fontweight='bold')
    plt.tight_layout()
    wandb.log({"xai/gradcam": wandb.Image(fig)})
    plt.show()
    plt.close()

    # Attention weight visualization
    if last_attn_w1 is not None:
        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
        for idx, (attn, title) in enumerate([(last_attn_w1, 'MHA Layer 1'), (last_attn_w2, 'MHA Layer 2')]):
            # attn shape: (1, num_tokens, num_tokens) — already head-averaged by nn.MHA
            attn_map = attn.squeeze(0).numpy()  # (tokens, tokens)
            if attn_map.ndim == 1:
                attn_map = attn_map.reshape(1, -1)  # safety fallback
            sns.heatmap(attn_map, annot=True, fmt='.3f', cmap='viridis', ax=axes2[idx],
                        xticklabels=[f'T{i}' for i in range(attn_map.shape[1])],
                        yticklabels=[f'T{i}' for i in range(attn_map.shape[0])])
            axes2[idx].set_title(f'{title} — Attention Weights', fontsize=12)
        fig2.suptitle('XAI: Self-Attention Maps', fontsize=15, fontweight='bold')
        plt.tight_layout()
        wandb.log({"xai/attention_weights": wandb.Image(fig2)})
        plt.show()
        plt.close()

    print("\n✅ XAI visualizations generated.")


# ===================== MAIN EXECUTION =====================
if __name__ == '__main__':
    # W&B login via Kaggle secrets
    from kaggle_secrets import UserSecretsClient
    secret = UserSecretsClient()
    wandb_key = secret.get_secret("wandb_api_key")
    wandb.login(key=wandb_key)

    wandb.init(
        project="ccal_paper_reproduction",
        name="ccal_paper_method_leaky_split",
        config=config,
        tags=["paper-method", "image-level-split", "leaky", "80-20", "fp16"],
        notes="Exact paper reproduction: image-level 80/20 split, lr=0.001, /255 norm, augment to 10k, FP16+FlashAttn"
    )

    # Prepare data (paper method)
    print("="*60)
    print("CCAL Paper Reproduction - Image-Level 80/20 Split")
    print("="*60)
    train_loader, test_loader = prepare_paper_data(
        config['data_dir'],
        test_size=config['test_size'],
        target_total=config['target_total_images'],
        batch_size=config['batch_size'],
        image_size=config['image_size'],
    )

    # Log dataset info
    wandb.log({
        "dataset/train_samples": len(train_loader.dataset),
        "dataset/test_samples": len(test_loader.dataset),
        "dataset/split_method": "image-level random (paper method)",
    })

    # Create model
    print("\nCreating CCAL model...")
    model = CCALModel(
        num_classes=3,
        num_heads=config['num_heads'],
        key_dim=config['key_dim'],
    )

    # torch.compile removed — overhead hurts small models (703K params)
    # model = torch.compile(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    wandb.log({
        "model/total_parameters": total_params,
        "model/trainable_parameters": trainable_params,
    })
    wandb.watch(model, log="all", log_freq=100)

    # Train
    print("\nStarting training...")
    history = train_model(
        model, train_loader, test_loader,
        num_epochs=config['epochs'],
        learning_rate=config['learning_rate'],
        device=config['device']
    )

    # Plot
    plot_training_history(history)

    # Load best & evaluate
    print("\nLoading best model for evaluation...")
    model = load_best_model(model, device=config['device'])
    accuracy, precision, recall, f1, auc_val = evaluate_model(model, test_loader, device=config['device'])

    # XAI: Grad-CAM + Attention visualization
    print("\nGenerating XAI visualizations...")
    visualize_xai(model, test_loader, device=config['device'])

    # Save final model
    torch.save(model.state_dict(), 'ccal_paper_method.pth')
    artifact = wandb.Artifact('ccal_paper_method', type='model')
    artifact.add_file('ccal_paper_method.pth')
    wandb.log_artifact(artifact)

    wandb.finish()
    print("Done!")
