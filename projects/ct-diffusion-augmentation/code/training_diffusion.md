# Class-Conditional DDPM Training Guide
> Complete reference for `conditional_diffusion_ct_augmentation.py`
> Dataset: IQ-OTH/NCCD Lung Cancer CT Scans | Model: 19M param U-Net | Framework: PyTorch DDP

---

## 📋 Full Configuration Dump

### Dataset & Paths
| Parameter | Value | Notes |
|-----------|-------|-------|
| `DATASET_ROOT` | `/kaggle/input/.../The IQ-OTHNCCD lung cancer dataset` | Kaggle dataset path |
| `OUTPUT_DIR` | `/kaggle/working/diffusion_output` | Checkpoints, plots, samples |
| `SYNTHETIC_DIR` | `/kaggle/working/synthetic_images` | Generated images organized by class |
| `CLASSES` | `["Benign cases", "Malignant cases", "Normal cases"]` | 3 classes |
| `NUM_CLASSES` | 3 | Class-conditional generation |

### Image Settings
| Parameter | Value | Notes |
|-----------|-------|-------|
| `IMG_SIZE` | 128 | 128×128 pixels |
| `IMG_CHANNELS` | 1 | Grayscale CT scans |

### Diffusion Process
| Parameter | Value | Notes |
|-----------|-------|-------|
| `TIMESTEPS` | 1000 | Standard DDPM (T=1000) |
| `BETA_START` | 0.0001 | β₁ — noise at step 1 |
| `BETA_END` | 0.02 | β_T — noise at step T |
| Schedule | **Linear** | `β_t = β₁ + (β_T - β₁) × t/T` |

### U-Net Architecture
| Parameter | Value | Notes |
|-----------|-------|-------|
| `BASE_CHANNELS` | 64 | Channel progression: 64→128→256→512 |
| `TIME_EMB_DIM` | 256 | Sinusoidal timestep embedding |
| `CLASS_EMB_DIM` | 256 | Learned class embedding |
| Total params | **18,986,689** (19M) | Model size: 72.4 MB (fp32) |
| Architecture | U-Net with ResBlocks + Attention | Attention at 16×16 and 8×8 resolution |

### Training Hyperparameters
| Parameter | Value | Notes |
|-----------|-------|-------|
| `EPOCHS` | **1000** | ~62 min on 2×T4 |
| `BATCH_SIZE` | 16 per GPU | Auto-detected for T4 (14.6GB VRAM) |
| Effective batch | **32** | 16 × 2 GPUs |
| `LEARNING_RATE` | 0.0002 | 1e-4 base × 2 (linear DDP scaling) |
| LR Schedule | **CosineAnnealingLR** | Decays from 2e-4 → 0 over 1000 epochs |
| `EMA_DECAY` | 0.9999 | EMA weights used for generation |
| Optimizer | **AdamW** | β1=0.9, β2=0.999 |
| `SAVE_EVERY` | 25 | Checkpoint every 25 epochs |
| `SAMPLE_EVERY` | 10 | Generate samples every 10 epochs |
| `CFG_SCALE` | 3.0 | Classifier-free guidance scale |
| `p_uncond` | 0.1 | 10% label dropout during training |

### Mixed Precision & Compilation
| Parameter | Value | Notes |
|-----------|-------|-------|
| `USE_AMP` | True | Automatic Mixed Precision |
| `AMP_DTYPE` | **torch.float16** | FP16 on T4 (SM 7.5, no BF16) |
| GradScaler | **Enabled** | Required for FP16 to prevent NaN |
| `USE_COMPILE` | True | `torch.compile()` for kernel fusion |

### Distributed Training (DDP)
| Parameter | Value | Notes |
|-----------|-------|-------|
| `WORLD_SIZE` | 2 | 2× Tesla T4 |
| Backend | NCCL | GPU-optimized collective ops |
| Launch | `torchrun --nproc_per_node=2` | Automatic rank assignment |

### W&B Configuration
| Parameter | Value | Notes |
|-----------|-------|-------|
| `WANDB_PROJECT` | `ct-diffusion-augmentation` | W&B project name |
| `WANDB_RUN_NAME` | `ddpm-2xT4-ddp` | Auto-named based on GPU count |
| Logged metrics | loss, best_loss, LR, epoch_time, ETA, elapsed/remaining hours, GPU memory |
| Logged media | Sample images every 10 epochs, final loss plot |

### Time Guard (Kaggle)
| Parameter | Value | Notes |
|-----------|-------|-------|
| `MAX_TRAINING_HOURS` | 10.0 | Kaggle limit = 11h, reserve 1h buffer |
| `GENERATION_BUDGET_MIN` | 30 | Reserve 30 min for generation + evaluation |

### Generation Settings
| Parameter | Value | Notes |
|-----------|-------|-------|
| `NUM_SYNTHETIC_BENIGN` | 300 | Primary target (most underrepresented) |
| `NUM_SYNTHETIC_NORMAL` | 100 | Secondary augmentation |
| `DDIM_STEPS` | 50 | DDIM sampler (50 steps vs 1000 for DDPM) |
| `SEED` | 42 | Reproducibility (offset per rank for diversity) |

---

## 🖥️ GPU Memory Explained

### Why is one GPU at 8GB and the other at 0.4GB?

**Short answer: Both GPUs are working correctly. The W&B chart is misleading.**

The W&B dashboard shows two metrics that mean different things:

```
gpu/memory_reserved_gb  = 8.92 GB  ← CUDA memory POOL (pre-allocated cache)
gpu/memory_allocated_gb = 0.40 GB  ← Actual TENSORS alive at logging time
```

**What's happening:**

```
During forward/backward pass:
┌───────────────────────────────────┐
│  Model weights:        72 MB      │
│  Optimizer states:    144 MB      │  → These persist
│  EMA weights:          72 MB      │
│  ──────────────────────────────   │
│  Activations:       ~7.5 GB      │  → FREED after backward!
│  Gradients:         ~0.5 GB      │  → FREED after optimizer.step()!
└───────────────────────────────────┘

At W&B logging time (after backward + step):
┌───────────────────────────────────┐
│  Model + Optimizer + EMA: ~0.4 GB │  ← memory_allocated = ONLY this
│  (empty cache):         ~8.5 GB   │  ← memory_reserved = whole pool
└───────────────────────────────────┘
```

**Why `reserved` stays at 8.9GB:** PyTorch's CUDA allocator pre-reserves GPU memory and reuses it. Even after tensors are freed, the memory pool stays reserved to avoid slow `cudaMalloc` calls. This is **normal and efficient**.

**Why `allocated` is only 0.4GB:** W&B logs this metric AFTER the backward pass and optimizer step. At that point, all activation tensors and gradients have been freed — only the persistent model weights, optimizer states, and EMA remain.

**Are both GPUs actually training?** Yes! DDP gives each GPU an identical model copy. The training output shows:
- `Batch Size: 16 per GPU × 2 GPU(s) = 32 effective`
- Both rank 0 and rank 1 print gradient warnings
- `24 batches/epoch` = 768 images / 32 per batch ≈ 24 ✓

**To verify both GPUs are active**, add this cell in Kaggle:
```python
!nvidia-smi
```

---

## 📊 Training Loss Interpretation

### Loss Curve Stages

```
Epoch 1-10:     Loss 0.30 → 0.09   "Learning basic noise prediction"
Epoch 10-50:    Loss 0.09 → 0.05   "Learning image structure"
Epoch 50-200:   Loss 0.05 → 0.04   "Plateau — learning fine details"
Epoch 200-500:  Loss ~0.04          "Samples start forming clear anatomy"
Epoch 500-1000: Loss ~0.03-0.04    "Refinement — sharpness and detail"
```

### Why Loss ≠ Visual Quality

The diffusion loss is `MSE(predicted_noise, actual_noise)` **averaged over ALL timesteps**:
- Getting t=1 right (nearly clean image) is easy
- Getting t=500 right (half noise) is hard — this is where structure forms
- Getting t=999 right (pure noise) is easy — just predict the mean

A loss of 0.04 means the model is **good on average** across all timesteps, but visual quality depends on getting the **hard middle timesteps** right — which improves slowly.

**Rule:** Don't judge diffusion quality by loss alone. Look at the sample images in W&B.

---

## 🏗️ Architecture Details

### U-Net Structure

```
Input: [B, 1, 128, 128] + timestep t + class c
                │
    ┌───────────┴───────────┐
    │   Timestep Embedding  │  sinusoidal → MLP → 256-dim
    │   Class Embedding     │  learned → 256-dim
    └───────────┬───────────┘
                │
    ┌───────────┴───────────┐
    │   ENCODER              │
    │   128×128 ─── 64 ch   │  ← ResBlock + GroupNorm + SiLU
    │    64×64 ── 128 ch    │  ← ResBlock × 2
    │    32×32 ── 256 ch    │  ← ResBlock × 2 + Self-Attention
    │    16×16 ── 512 ch    │  ← ResBlock × 2 + Self-Attention
    └───────────┬───────────┘
                │
    ┌───────────┴───────────┐
    │   BOTTLENECK           │
    │    8×8 ─── 512 ch     │  ← ResBlock + Attention + ResBlock
    └───────────┬───────────┘
                │
    ┌───────────┴───────────┐
    │   DECODER (+ skip connections) │
    │    16×16 ── 256 ch    │  ← ResBlock × 2 + Attention
    │    32×32 ── 128 ch    │  ← ResBlock × 2 + Attention
    │    64×64 ─── 64 ch    │  ← ResBlock × 2
    │   128×128 ── 64 ch    │  ← ResBlock × 2
    └───────────┬───────────┘
                │
    Output: [B, 1, 128, 128]  ← Predicted noise ε
```

### Key Design Choices
- **Self-Attention** at 32×32 and 16×16 (not 128×128 — too expensive)
- **GroupNorm** (groups=32) instead of BatchNorm — works better with small batches
- **SiLU activation** (Swish) — smoother gradients than ReLU
- **Sinusoidal timestep embedding** — same as Transformer positional encoding

---

## 🎯 Classifier-Free Guidance (CFG)

### During Training
```python
# 10% of the time, DROP the class label → model learns "unconditional"
if random() < 0.1:
    class_label = NULL_CLASS  # No class info
else:
    class_label = actual_class  # Benign / Malignant / Normal
```

### During Generation
```python
# Generate with amplified class signal
noise_uncond = model(x_t, t, class=NULL)      # What does any image look like?
noise_cond   = model(x_t, t, class="Benign")  # What does Benign look like?

# Amplify the difference (CFG scale = 3.0)
noise_final = noise_uncond + 3.0 × (noise_cond - noise_uncond)
```

| CFG Scale | Effect | Use Case |
|-----------|--------|----------|
| 1.0 | No guidance — diverse but generic | Exploration |
| **3.0** | **Balanced — class-faithful + diverse** | **Your setting** |
| 5.0 | Strong class adherence, less diversity | When classes look too similar |
| 10.0+ | Oversaturated, artifacts | Avoid |

---

## 📁 Output Files

After training completes, find everything at `/kaggle/working/`:

```
/kaggle/working/
├── diffusion_output/
│   ├── best_diffusion_model.pt          ← Best by loss (EMA weights)
│   ├── final_checkpoint.pt              ← Last epoch checkpoint
│   ├── checkpoint_epoch_25.pt           ← Every 25 epochs
│   ├── checkpoint_epoch_50.pt
│   ├── ...
│   ├── samples_epoch10_Benign.png       ← Generated samples every 10 epochs
│   ├── samples_epoch10_Malignant.png
│   ├── samples_epoch10_Normal.png
│   ├── training_loss.png                ← Loss curve plot
│   ├── tsne_real_vs_synthetic.png       ← Distribution comparison
│   └── fid_scores.txt                   ← FID per class
│
├── synthetic_images/
│   ├── Benign/                          ← 300 generated benign CTs
│   │   ├── benign_0000.png
│   │   ├── benign_0001.png
│   │   └── ...
│   ├── Malignant/                       ← (if configured)
│   └── Normal/                          ← 100 generated normal CTs
│       ├── normal_0000.png
│       └── ...
│
└── wandb/                               ← W&B logs (auto-synced)
```

---

## 📊 Quality Metrics

### FID (Fréchet Inception Distance) — Lower = Better

| FID Score | Quality | Interpretation |
|-----------|---------|----------------|
| < 20 | 🟢 Excellent | Near indistinguishable from real |
| 20-50 | 🟢 Good | High quality, publishable |
| 50-100 | 🟡 Acceptable | Useful for augmentation |
| 100-200 | 🟠 Mediocre | May need more training |
| > 200 | 🔴 Poor | Restart with different config |

### Why No Inception Score (IS)?
IS uses an ImageNet classifier to judge diversity and quality. For grayscale medical CT scans:
- ImageNet classes are irrelevant (dogs, cats, cars ≠ CT anatomy)
- Single-channel images break IS assumptions
- **FID + downstream task accuracy** are the standard metrics for medical synthesis

**Thesis defense answer:** *"We use FID and downstream classifier accuracy rather than Inception Score, as IS is designed for natural RGB images and is not applicable to single-channel medical CTs (Borji, 2019)."*

---

## 🔬 How Generation Works (Thesis-Ready)

### Forward Process (Training)
Add Gaussian noise gradually to real images:
```
x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε,    ε ~ N(0, I)

where ᾱ_t = ∏(1 - β_i) from i=1 to t
```

### Reverse Process (Generation)
Start from pure noise, denoise step by step:
```
x_{t-1} = (1/√α_t) · (x_t - (β_t/√(1-ᾱ_t)) · ε_θ(x_t, t, c)) + σ_t · z
```

### DDIM Acceleration
Skip steps deterministically (1000 → 50 steps):
```
x_{t-1} = √ᾱ_{t-1} · predicted_x0 + √(1-ᾱ_{t-1}) · predicted_direction
```
No random noise added → deterministic, faster, same quality.

---

## 🛡️ Training Safeguards in Your Script

| Safeguard | What It Does | Status |
|-----------|-------------|--------|
| **Time Guard** | Stops training at 10h to save results | ✅ Active |
| **GradScaler** | Prevents FP16 NaN/overflow | ✅ Active |
| **EMA** | Stabilizes generation quality | ✅ Active (decay=0.9999) |
| **Seed per rank** | Different data order per GPU | ✅ Active |
| **Rank-0 only saves** | Prevents file conflicts | ✅ Active |
| **DDP barrier** | Syncs GPUs after saves | ✅ Active |
| Early stopping | Patience-based stopping | ❌ Removed (full 1000 epochs) |

---

## 🎓 Thesis Defense Q&A

**Q: Why diffusion over GANs?**
> Diffusion models avoid mode collapse, produce higher diversity, and have stable training (no discriminator games). Critical for medical data where we need varied, realistic samples.

**Q: Why class-conditional?**
> Enables targeted generation of specific pathology types. We can generate exactly 300 Benign CTs to balance the dataset without generating unnecessary Malignant samples.

**Q: Why 128×128 resolution?**
> Balances quality vs compute. 256×256 would need 4× more memory and 4× more training time. For augmentation (not diagnosis), 128×128 preserves sufficient anatomical detail.

**Q: How do you ensure generated images aren't memorized copies?**
> 1. Data augmentation during training prevents overfitting
> 2. DDIM sampling from random noise ensures novel images
> 3. t-SNE visualization confirms synthetic distribution overlaps but doesn't collapse onto real points
> 4. We can verify by nearest-neighbor search between synthetic and training images

**Q: What's the computational cost?**
> ~62 minutes on 2× Tesla T4 (30GB total VRAM), 1000 epochs, 19M parameters. Generation: ~2 seconds per image with DDIM-50.

---

## ⚠️ Common Pitfalls & Solutions

| Problem | Symptom | Solution |
|---------|---------|----------|
| Grey blobs | All samples look the same | Train longer, increase CFG scale |
| NaN loss | Training crashes | Reduce LR, add gradient clipping |
| Mode collapse | Benign = Malignant | Check class conditioning, increase p_uncond |
| Memorization | FID < 5 | Add more augmentation, reduce epochs |
| Noisy samples | Static-like output | Train significantly longer (500+ epochs) |
| Wrong contrast | Washed out images | Check normalization (must be [0,1]) |
| DDP hang | Training freezes | Check barrier placement, same code path all ranks |
