# %% [markdown]
# ## 3. Model Architecture Definition
#
# **Paper Section 3.4.2 — Fine-tuned EfficientNet:**
# 1. Load EfficientNet B0–B4 pretrained on ImageNet
# 2. Remove original 1000-unit classification head
# 3. Add: Global Average Pooling → Dropout(0.5) → Dense(3, softmax)
# 4. Retrain entire network end-to-end
#
# $$\text{Compound Scaling: } d = \alpha^\phi,\; w = \beta^\phi,\; r = \gamma^\phi$$
# $$\text{s.t. } \alpha \cdot \beta^2 \cdot \gamma^2 \approx 2,\; \alpha \ge 1, \beta \ge 1, \gamma \ge 1$$

# %%
import torch
import torch.nn as nn
import torchvision.models as tv_models


def build_lung_effnet(variant='b1', num_classes=3, dropout=0.5, drop_connect=0.2,
                      pretrained=True):
    """
    Build Lung-EffNet: EfficientNet backbone + custom classification head.
    Paper Section 3.4.2, Fig 9.

    Architecture:
        EfficientNet backbone (frozen or fine-tuned)
        → Global Average Pooling
        → Dropout(0.5)
        → Dense(num_classes, softmax)
    """
    weights_map = {
        'b0': (tv_models.efficientnet_b0, tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1),
        'b1': (tv_models.efficientnet_b1, tv_models.EfficientNet_B1_Weights.IMAGENET1K_V1),
        'b2': (tv_models.efficientnet_b2, tv_models.EfficientNet_B2_Weights.IMAGENET1K_V1),
        'b3': (tv_models.efficientnet_b3, tv_models.EfficientNet_B3_Weights.IMAGENET1K_V1),
        'b4': (tv_models.efficientnet_b4, tv_models.EfficientNet_B4_Weights.IMAGENET1K_V1),
    }

    model_fn, weights = weights_map[variant]
    model = model_fn(weights=weights if pretrained else None)

    # Replace classifier head (Paper: GAP → Dropout(0.5) → Dense(3))
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )

    # Parameter summary (Paper Table 8)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🏗  Lung-EffNet ({variant.upper()}):")
    print(f"   Total params     : {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"   Trainable params : {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"   Dropout          : {dropout}")
    print(f"   Output units     : {num_classes}")

    return model


def build_comparison_model(arch_name, num_classes=3, pretrained=True):
    """Build comparison architectures (Paper Table 8): ResNet50, MobileNet, etc."""
    if arch_name == 'resnet50':
        model = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch_name == 'mobilenet_v2':
        model = tv_models.mobilenet_v2(weights=tv_models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif arch_name == 'mobilenet_v3_small':
        model = tv_models.mobilenet_v3_small(weights=tv_models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    else:
        raise ValueError(f"Unknown arch: {arch_name}")

    total = sum(p.numel() for p in model.parameters())
    print(f"  {arch_name}: {total/1e6:.2f}M params")
    return model


# %%  Build all EfficientNet variants and log params
print("=" * 60)
print(" Parameter Count Comparison (Paper Table 8)")
print("=" * 60)
paper_params = {'b0': 4.01, 'b1': 6.51, 'b2': 7.71, 'b3': 10.70, 'b4': 17.55}
for var in ['b0', 'b1', 'b2', 'b3', 'b4']:
    m = build_lung_effnet(var)
    actual = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"   Paper: {paper_params[var]:.2f}M  |  Ours: {actual:.2f}M")
    del m; gc.collect()

# %% [markdown]
# ### 3.1 Architecture Diagram
#
# ```
# ┌─────────────────────────────────────────────────────┐
# │                  INPUT (240×240×3)                   │
# ├─────────────────────────────────────────────────────┤
# │              EfficientNet Backbone                   │
# │  ┌──────────────────────────────────────────────┐   │
# │  │  Stem (Conv 3×3, BN, Swish)                  │   │
# │  │  Block 1: MBConv1, k3×3                      │   │
# │  │  Block 2: MBConv6, k3×3                      │   │
# │  │  Block 3: MBConv6, k5×5                      │   │
# │  │  Block 4: MBConv6, k3×3                      │   │
# │  │  Block 5: MBConv6, k5×5                      │   │
# │  │  Block 6: MBConv6, k5×5                      │   │
# │  │  Block 7: MBConv6, k3×3                      │   │
# │  │  Final (Conv 1×1, BN, Swish)                 │   │
# │  └──────────────────────────────────────────────┘   │
# ├─────────────────────────────────────────────────────┤
# │          Global Average Pooling (GAP)                │
# ├─────────────────────────────────────────────────────┤
# │              Dropout (p=0.5)                         │
# ├─────────────────────────────────────────────────────┤
# │          Dense (3 units, Softmax)                    │
# │        [Benign | Malignant | Normal]                 │
# └─────────────────────────────────────────────────────┘
# ```

# %% [markdown]
# ## 4. Training Pipeline
#
# **Loss:** Categorical Cross-Entropy (Paper Table 4)
# $$\mathcal{L}_{CE} = -\sum_{c=1}^{C} y_c \log(\hat{y}_c)$$
#
# **Optimizer:** Adam with initial LR = 0.001
# **LR Schedule:** ReduceLROnPlateau with factor=0.3, patience=5
#
# **Evaluation Metrics (Paper Section 4.1):**
# - Accuracy: $\frac{TP + TN}{TP + TN + FP + FN}$ (Eq. 2)
# - Precision: $\frac{TP}{TP + FP}$ (Eq. 3)
# - Recall: $\frac{TP}{TP + FN}$ (Eq. 4)
# - F1-Score: $\frac{2 \cdot P \cdot R}{P + R}$ (Eq. 5)
# - ROC/AUC (Eqs. 6, 7)

# %%
import time
import copy
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = torch.tensor(labels, device=device) if not isinstance(labels, torch.Tensor) else labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate_model(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = torch.tensor(labels, device=device) if not isinstance(labels, torch.Tensor) else labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += imgs.size(0)
        all_preds.extend(outputs.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(F.softmax(outputs, dim=1).cpu().tolist())
    return running_loss / total, correct / total, all_preds, all_labels, all_probs


def train_full(model, train_ld, val_ld, test_ld, model_name, device,
               epochs=50, lr=1e-3, save_dir=None):
    """Full training loop matching paper hyperparameters."""
    save_dir = Path(save_dir or RESULTS_DIR / model_name)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=CONFIG['lr_decay_factor'],
                                  patience=CONFIG['lr_patience'])

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [],
               'val_f1': [], 'lr': []}
    best_val_acc = 0
    best_model_wts = None
    train_start = time.time()

    print(f"\n{'='*60}")
    print(f" Training: {model_name} | {epochs} epochs | LR={lr}")
    print(f"{'='*60}")

    for epoch in range(epochs):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_ld, optimizer, criterion, device)
        vl_loss, vl_acc, vl_preds, vl_labels, vl_probs = evaluate_model(model, val_ld, criterion, device)
        vl_f1 = f1_score(vl_labels, vl_preds, average='macro', zero_division=0)

        scheduler.step(vl_acc)
        cur_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)
        history['val_f1'].append(vl_f1)
        history['lr'].append(cur_lr)

        marker = ''
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, save_dir / f'{model_name}_best.pth')
            marker = ' ★'

        if (epoch + 1) % 5 == 0 or epoch == 0 or marker:
            print(f"  [{epoch+1:3d}/{epochs}] loss={tr_loss:.4f}/{vl_loss:.4f} "
                  f"acc={tr_acc:.4f}/{vl_acc:.4f} f1={vl_f1:.4f} "
                  f"lr={cur_lr:.1e} ({time.time()-t0:.1f}s){marker}")

    total_time = time.time() - train_start
    print(f"\n⏱  Total training time: {total_time:.0f}s ({total_time/60:.1f}min)")

    # Load best weights and evaluate on test
    if best_model_wts:
        model.load_state_dict(best_model_wts)

    _, test_acc, test_preds, test_labels, test_probs = evaluate_model(
        model, test_ld, criterion, device)

    results = {
        'model_name': model_name,
        'test_accuracy': test_acc,
        'test_preds': test_preds,
        'test_labels': test_labels,
        'test_probs': test_probs,
        'history': history,
        'training_time_s': total_time,
        'total_params': sum(p.numel() for p in model.parameters()),
    }

    return model, results


# %% [markdown]
# ## 5. Evaluation & Visualization Utilities

# %%
def compute_full_metrics(preds, labels, probs, class_names):
    """Compute all metrics from Paper Section 4.1."""
    report = classification_report(labels, preds, target_names=class_names,
                                    output_dict=True, zero_division=0)
    try:
        auc_score = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
    except:
        auc_score = float('nan')

    return {
        'accuracy': report['accuracy'],
        'precision_macro': report['macro avg']['precision'],
        'recall_macro': report['macro avg']['recall'],
        'f1_macro': report['macro avg']['f1-score'],
        'auc_macro': auc_score,
        'per_class': {n: report[n] for n in class_names},
        'report_str': classification_report(labels, preds, target_names=class_names, zero_division=0)
    }


def plot_training_curves(history, title, save_path):
    """Paper Fig 10, 12: Training/validation accuracy and loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title(f'{title} — Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title(f'{title} — Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(preds, labels, class_names, title, save_path):
    """Paper Fig 11, 13: Confusion matrix."""
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names,
                yticklabels=class_names, ax=ax, annot_kws={'size': 16},
                linewidths=0.5, linecolor='gray')
    ax.set_xlabel('Predicted', fontsize=13)
    ax.set_ylabel('True', fontsize=13)
    ax.set_title(title, fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_roc_curves(labels, probs, class_names, title, save_path):
    """Paper Fig 14: ROC curves."""
    y_bin = label_binarize(labels, classes=list(range(len(class_names))))
    probs_arr = np.array(probs)

    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ['#2ecc71', '#e74c3c', '#3498db']
    for i, (name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs_arr[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=2.5,
                label=f'{name} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_precision_recall(labels, probs, class_names, title, save_path):
    """Precision-Recall curves (supplementary)."""
    y_bin = label_binarize(labels, classes=list(range(len(class_names))))
    probs_arr = np.array(probs)

    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ['#2ecc71', '#e74c3c', '#3498db']
    for i, (name, color) in enumerate(zip(class_names, colors)):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], probs_arr[:, i])
        ap = average_precision_score(y_bin[:, i], probs_arr[:, i])
        ax.plot(rec, prec, color=color, linewidth=2.5,
                label=f'{name} (AP = {ap:.3f})')

    ax.set_xlabel('Recall', fontsize=13)
    ax.set_ylabel('Precision', fontsize=13)
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def full_evaluation(results, class_names, save_dir):
    """Run all evaluation plots and print metrics."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    name = results['model_name']

    metrics = compute_full_metrics(
        results['test_preds'], results['test_labels'],
        results['test_probs'], class_names)

    print(f"\n{'='*60}")
    print(f" TEST RESULTS — {name}")
    print(f"{'='*60}")
    print(metrics['report_str'])
    print(f"  AUC (macro): {metrics['auc_macro']:.4f}")
    print(f"  Training time: {results['training_time_s']:.0f}s")
    print(f"  Parameters: {results['total_params']/1e6:.2f}M")

    plot_training_curves(results['history'], name, save_dir / f'{name}_curves.png')
    plot_confusion_matrix(results['test_preds'], results['test_labels'],
                         class_names, f'{name} — Confusion Matrix',
                         save_dir / f'{name}_confusion.png')
    plot_roc_curves(results['test_labels'], results['test_probs'],
                   class_names, f'{name} — ROC Curves',
                   save_dir / f'{name}_roc.png')
    plot_precision_recall(results['test_labels'], results['test_probs'],
                         class_names, f'{name} — Precision-Recall',
                         save_dir / f'{name}_pr.png')

    # Save metrics
    save_metrics = {k: v for k, v in metrics.items() if k != 'report_str'}
    save_metrics['training_time_s'] = results['training_time_s']
    save_metrics['total_params'] = results['total_params']
    with open(save_dir / f'{name}_metrics.json', 'w') as f:
        json.dump(save_metrics, f, indent=2)

    return metrics

print("✅ Part 2 (Model + Training + Eval) complete.")
