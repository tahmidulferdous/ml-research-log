"""
=============================================================================
 Rectified Flow — Quality Evaluation (FID + Visual Analysis)
 
 Usage:
   python rectified_flow/evaluate.py --checkpoint /path/to/best_rf_model.pt
   
 Computes FID score between real and synthetic images,
 generates t-SNE visualization, and runs nearest-neighbor analysis.
=============================================================================
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from collections import Counter

from torch.utils.data import Dataset, DataLoader

# Local imports
from config import cfg, gpu_cfg


# ═══════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION (for FID)
# ═══════════════════════════════════════════════════════════════════════

class SimpleFeatureExtractor(nn.Module):
    """
    Lightweight CNN feature extractor for FID computation.
    
    Note: For medical images, we use a simple learned extractor rather than
    ImageNet-pretrained InceptionV3, as Inception features are designed for
    natural RGB images and don't transfer well to grayscale CT scans.
    """

    def __init__(self, img_size=224):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )

    def forward(self, x):
        return self.features(x).flatten(1)  # (B, 256*4*4) = (B, 4096)


class ImageFolderDataset(Dataset):
    """Load images from a folder."""

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


# ═══════════════════════════════════════════════════════════════════════
# FID COMPUTATION
# ═══════════════════════════════════════════════════════════════════════

def extract_features(dataloader, feature_extractor, device):
    """Extract features from all images in a dataloader."""
    all_features = []
    feature_extractor.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            features = feature_extractor(batch.to(device))
            all_features.append(features.cpu().numpy())
    return np.concatenate(all_features, axis=0)


def calculate_fid(mu1, sigma1, mu2, sigma2):
    """
    Calculate Fréchet Inception Distance.
    
    FID = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ - 2(Σ₁Σ₂)^½)
    
    Lower = Better. FID < 50 is generally good for medical images.
    """
    from scipy.linalg import sqrtm

    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


def compute_fid(real_dir, synthetic_dir, device, img_size=224, batch_size=32):
    """
    Compute FID between real and synthetic image directories.
    
    Returns:
        fid_score: float (lower is better)
    """
    # Create datasets
    real_dataset = ImageFolderDataset(real_dir, img_size)
    synth_dataset = ImageFolderDataset(synthetic_dir, img_size)

    if len(real_dataset) == 0 or len(synth_dataset) == 0:
        print(f"⚠️ Empty dataset: real={len(real_dataset)}, synthetic={len(synth_dataset)}")
        return float('inf')

    real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False)
    synth_loader = DataLoader(synth_dataset, batch_size=batch_size, shuffle=False)

    # Feature extractor
    feat_ext = SimpleFeatureExtractor(img_size).to(device)

    # Extract features
    real_features = extract_features(real_loader, feat_ext, device)
    synth_features = extract_features(synth_loader, feat_ext, device)

    # Compute statistics
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    mu_synth = np.mean(synth_features, axis=0)
    sigma_synth = np.cov(synth_features, rowvar=False)

    # Compute FID
    fid = calculate_fid(mu_real, sigma_real, mu_synth, sigma_synth)
    return fid


# ═══════════════════════════════════════════════════════════════════════
# NEAREST NEIGHBOR ANALYSIS (Memorization Check)
# ═══════════════════════════════════════════════════════════════════════

def nearest_neighbor_analysis(real_dir, synthetic_dir, device, img_size=224,
                               num_samples=50, batch_size=32):
    """
    Check if synthetic images are just copies of training data.
    
    Computes L2 distance between each synthetic image and its nearest
    real image neighbor. If distances are very small, the model memorized.
    
    Returns:
        avg_min_distance: average nearest-neighbor distance
        min_distances: list of all nearest-neighbor distances
    """
    real_dataset = ImageFolderDataset(real_dir, img_size)
    synth_dataset = ImageFolderDataset(synthetic_dir, img_size)

    if len(real_dataset) == 0 or len(synth_dataset) == 0:
        return float('inf'), []

    # Load all real images into memory
    real_images = []
    for i in range(min(len(real_dataset), 500)):
        real_images.append(real_dataset[i])
    real_tensor = torch.stack(real_images).to(device)  # (N_real, 1, H, W)

    # For each synthetic image, find nearest real neighbor
    min_distances = []
    num_check = min(num_samples, len(synth_dataset))

    for i in tqdm(range(num_check), desc="Nearest neighbor check"):
        synth_img = synth_dataset[i].unsqueeze(0).to(device)  # (1, 1, H, W)
        # L2 distance to all real images
        dists = torch.norm(
            real_tensor.flatten(1) - synth_img.flatten(1), dim=1
        )
        min_distances.append(dists.min().item())

    avg_dist = np.mean(min_distances) if min_distances else float('inf')
    return avg_dist, min_distances


# ═══════════════════════════════════════════════════════════════════════
# T-SNE VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════

def plot_tsne(real_dir, synthetic_dir, device, img_size=224, batch_size=32,
              save_path=None):
    """
    Generate t-SNE plot comparing real vs synthetic image distributions.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
    except ImportError:
        print("⚠️ matplotlib or sklearn not available — skipping t-SNE")
        return

    real_dataset = ImageFolderDataset(real_dir, img_size)
    synth_dataset = ImageFolderDataset(synthetic_dir, img_size)

    real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False)
    synth_loader = DataLoader(synth_dataset, batch_size=batch_size, shuffle=False)

    feat_ext = SimpleFeatureExtractor(img_size).to(device)

    real_features = extract_features(real_loader, feat_ext, device)
    synth_features = extract_features(synth_loader, feat_ext, device)

    # Combine and run t-SNE
    all_features = np.concatenate([real_features, synth_features])
    labels = ['Real'] * len(real_features) + ['Synthetic'] * len(synth_features)

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_features) - 1))
    embeddings = tsne.fit_transform(all_features)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    n_real = len(real_features)
    ax.scatter(embeddings[:n_real, 0], embeddings[:n_real, 1],
               c='blue', alpha=0.6, label='Real', s=20)
    ax.scatter(embeddings[n_real:, 0], embeddings[n_real:, 1],
               c='red', alpha=0.6, label='Synthetic (RF)', s=20)
    ax.legend(fontsize=12)
    ax.set_title('t-SNE: Real vs Rectified Flow Synthetic', fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = save_path or os.path.join(cfg.OUTPUT_DIR, "tsne_real_vs_rf.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  📊 t-SNE saved: {save_path}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN EVALUATION
# ═══════════════════════════════════════════════════════════════════════

def run_evaluation(synthetic_dir=None):
    """Run complete quality evaluation suite."""
    device = cfg.DEVICE
    synthetic_dir = synthetic_dir or cfg.SYNTHETIC_DIR

    print(f"\n{'='*60}")
    print(f"QUALITY EVALUATION")
    print(f"{'='*60}")

    # Find real data directories
    real_root = cfg.DATASET_ROOT

    results = {}

    for class_name in ["Benign", "Normal"]:
        real_class_dir = os.path.join(real_root,
                                       [c for c in cfg.CLASSES if class_name.lower() in c.lower()][0])
        synth_class_dir = os.path.join(synthetic_dir, class_name)

        if not os.path.exists(synth_class_dir):
            print(f"  ⚠️ No synthetic {class_name} images found")
            continue

        synth_count = len(list(Path(synth_class_dir).glob("*.png")))
        if synth_count == 0:
            continue

        print(f"\n── {class_name} ──")
        print(f"  Synthetic images: {synth_count}")

        # FID
        fid = compute_fid(real_class_dir, synth_class_dir, device,
                          img_size=cfg.IMG_SIZE)
        print(f"  FID: {fid:.2f}")
        results[f"fid_{class_name.lower()}"] = fid

        # FID interpretation
        if fid < 20:
            quality = "🟢 Excellent"
        elif fid < 50:
            quality = "🟢 Good"
        elif fid < 100:
            quality = "🟡 Acceptable"
        elif fid < 200:
            quality = "🟠 Mediocre"
        else:
            quality = "🔴 Poor"
        print(f"  Quality: {quality}")

        # Nearest neighbor (memorization check)
        avg_nn, _ = nearest_neighbor_analysis(
            real_class_dir, synth_class_dir, device, img_size=cfg.IMG_SIZE
        )
        print(f"  Avg NN distance: {avg_nn:.4f}")
        if avg_nn < 0.1:
            print(f"  ⚠️ WARNING: Very low NN distance — possible memorization!")
        results[f"nn_dist_{class_name.lower()}"] = avg_nn

        # t-SNE
        tsne_path = os.path.join(cfg.OUTPUT_DIR, f"tsne_{class_name.lower()}.png")
        plot_tsne(real_class_dir, synth_class_dir, device,
                  img_size=cfg.IMG_SIZE, save_path=tsne_path)

    # Save results
    results_path = os.path.join(cfg.OUTPUT_DIR, "quality_scores.txt")
    with open(results_path, 'w') as f:
        f.write("Rectified Flow — Quality Evaluation Results\n")
        f.write("=" * 50 + "\n")
        for k, v in results.items():
            f.write(f"{k}: {v:.4f}\n")
    print(f"\n📄 Results saved: {results_path}")

    print(f"\n{'='*60}")
    print(f"✅ Evaluation complete")
    print(f"{'='*60}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RF synthetic images")
    parser.add_argument("--synthetic_dir", type=str, default=None)
    args = parser.parse_args()
    run_evaluation(synthetic_dir=args.synthetic_dir)
