# %% [markdown]
# ## 12. Error Analysis & GradCAM Visualization
#
# Paper Fig 15: GradCAM visualization of class-wise results

# %%
def compute_gradcam(model, img_tensor, target_class, target_layer):
    """Compute GradCAM heatmap for a given image and class."""
    model.eval()
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    def forward_hook(module, input, output):
        activations.append(output)

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)

    output = model(img_tensor.unsqueeze(0).to(DEVICE))
    model.zero_grad()
    output[0, target_class].backward()

    handle_f.remove()
    handle_b.remove()

    grads = gradients[0].cpu().data.numpy()[0]
    acts = activations[0].cpu().data.numpy()[0]
    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (CONFIG['img_size'], CONFIG['img_size']))
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam


def visualize_gradcam(model, records, class_names, save_dir, n_per_class=3):
    """GradCAM visualization (Paper Fig 15)."""
    save_dir = Path(save_dir)
    model.eval()
    # Get last conv layer
    target_layer = model.features[-1]

    fig, axes = plt.subplots(len(class_names), n_per_class * 2,
                             figsize=(4 * n_per_class * 2, 4 * len(class_names)))
    eval_tf = get_eval_transforms(CONFIG['img_size'])

    for row, label in enumerate(range(len(class_names))):
        samples = [r for r in records if r['label'] == label and r['split'] == 'test'][:n_per_class]
        for col, r in enumerate(samples):
            img = Image.open(r['path']).convert('RGB')
            img_tensor = eval_tf(img)

            # Original image
            img_show = img.resize((CONFIG['img_size'], CONFIG['img_size']))
            axes[row, col * 2].imshow(img_show)
            axes[row, col * 2].set_title(f'{class_names[label]}', fontsize=11)
            axes[row, col * 2].axis('off')

            # GradCAM overlay
            try:
                cam = compute_gradcam(model, img_tensor, label, target_layer)
                img_np = np.array(img_show).astype(float) / 255.0
                heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(float) / 255.0
                overlay = 0.5 * img_np + 0.5 * heatmap
                axes[row, col * 2 + 1].imshow(np.clip(overlay, 0, 1))
            except Exception as e:
                axes[row, col * 2 + 1].text(0.5, 0.5, 'Error', ha='center', va='center')
            axes[row, col * 2 + 1].set_title('GradCAM', fontsize=11)
            axes[row, col * 2 + 1].axis('off')

    fig.suptitle("GradCAM Visualization (cf. Paper Fig. 15)", fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_dir / 'gradcam_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()


# Run GradCAM on best model (B1)
best_model = build_lung_effnet('b1', pretrained=True)
best_ckpt = RESULTS_DIR / 'effnet_b1' / 'EfficientNet_B1_best.pth'
if best_ckpt.exists():
    best_model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))
    best_model = best_model.to(DEVICE)
    visualize_gradcam(best_model, all_records, CONFIG['class_names'], RESULTS_DIR / 'effnet_b1')
else:
    print("⚠️  Best model checkpoint not found, skipping GradCAM")

# %% [markdown]
# ### 12.1 Per-Class Error Analysis

# %%
def error_analysis(results, class_names, save_dir):
    """Detailed per-class performance breakdown."""
    preds = np.array(results['test_preds'])
    labels = np.array(results['test_labels'])

    print(f"\n{'='*60}")
    print(f" Error Analysis — {results['model_name']}")
    print(f"{'='*60}")

    for i, name in enumerate(class_names):
        mask = labels == i
        class_preds = preds[mask]
        class_acc = (class_preds == i).mean()
        misclassified = class_preds[class_preds != i]
        print(f"\n  {name} ({mask.sum()} samples):")
        print(f"    Accuracy: {class_acc*100:.2f}%")
        if len(misclassified) > 0:
            misc_counts = Counter(misclassified)
            for wrong_label, count in misc_counts.items():
                print(f"    Misclassified as {class_names[wrong_label]}: {count}")
        else:
            print(f"    ✅ No misclassifications!")

    # Confidence analysis
    probs = np.array(results['test_probs'])
    correct_mask = preds == labels
    correct_conf = probs[np.arange(len(labels)), labels][correct_mask].mean()
    if (~correct_mask).any():
        wrong_conf = probs[np.arange(len(labels)), preds][~correct_mask].mean()
    else:
        wrong_conf = 0
    print(f"\n  Avg confidence (correct):   {correct_conf:.4f}")
    print(f"  Avg confidence (incorrect): {wrong_conf:.4f}")

# Run on best model results
if 'EffNet-B1 (aug)' in all_experiment_results:
    error_analysis(all_experiment_results['EffNet-B1 (aug)'], CONFIG['class_names'], RESULTS_DIR)

# %% [markdown]
# ## 13. Computational Cost Reporting (Paper Table 8)
#
# | Model | FLOPs (G) | Params | Train Time | Test Time | Size (MB) | Acc (%) |
# |-------|-----------|--------|------------|-----------|-----------|---------|
# | B0    | 0.95      | 4.01M  | 00:03:20   | 00:00:02  | 47.2      | 93.67   |
# | B1    | 1.42      | 6.51M  | 00:46:40   | 00:00:01  | 76.4      | 99.10   |
# | B2    | 1.64      | 7.71M  | 00:50:50   | 00:00:02  | 90.0      | 97.74   |
# | B3    | 2.38      | 10.70M | 00:65:20   | 00:00:02  | 124.6     | 98.19   |
# | B4    | 3.71      | 17.55M | 01:05:50   | 00:00:04  | 203.6     | 97.29   |

# %%
def measure_inference_latency(model, img_size, device, n_warmup=10, n_test=50):
    """Measure per-sample inference latency."""
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size).to(device)

    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Measure
    times = []
    for _ in range(n_test):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            _ = model(dummy)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.time() - t0)

    return np.mean(times) * 1000, np.std(times) * 1000  # ms


def compute_flops(model, img_size):
    """Estimate FLOPs using thop if available, else return None."""
    try:
        from thop import profile
        dummy = torch.randn(1, 3, img_size, img_size).to(DEVICE)
        flops, params = profile(model.to(DEVICE), inputs=(dummy,), verbose=False)
        return flops / 1e9  # GFLOPs
    except ImportError:
        return None


def computational_cost_report(save_dir):
    """Generate computational cost table (Paper Table 8)."""
    print(f"\n{'='*80}")
    print(f" COMPUTATIONAL COST REPORT (cf. Paper Table 8)")
    print(f"{'='*80}")

    rows = []
    for variant in ['b0', 'b1', 'b2', 'b3', 'b4']:
        model = build_lung_effnet(variant, pretrained=True)
        model = model.to(DEVICE)

        total_params = sum(p.numel() for p in model.parameters())
        model_size_mb = sum(p.nelement() * p.element_size() for p in model.parameters()) / (1024**2)
        lat_mean, lat_std = measure_inference_latency(model, CONFIG['img_size'], DEVICE)
        flops = compute_flops(model, CONFIG['img_size'])

        # Get training time from results if available
        key = f'EffNet-{variant.upper()} (aug)'
        train_time = all_experiment_results.get(key, {}).get('training_time_s', 0)
        test_acc = all_experiment_results.get(key, {}).get('accuracy', 0)

        rows.append({
            'Model': f'EfficientNet-{variant.upper()}',
            'Params (M)': round(total_params / 1e6, 2),
            'FLOPs (G)': round(flops, 2) if flops else 'N/A',
            'Size (MB)': round(model_size_mb, 1),
            'Inference (ms)': f'{lat_mean:.1f}±{lat_std:.1f}',
            'Train Time (s)': round(train_time, 1),
            'Test Acc (%)': round(test_acc * 100, 2),
        })
        del model; torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(save_dir / 'computational_cost.csv', index=False)

    # GPU memory
    if torch.cuda.is_available():
        print(f"\n  Peak GPU Memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

computational_cost_report(RESULTS_DIR)

# %% [markdown]
# ## 14. Final Summary & Discussion
#
# ### Key Findings:
# 1. **EfficientNetB1** achieves the best balance of accuracy and efficiency
# 2. Data augmentation consistently improves performance across all variants
# 3. Transfer learning provides massive gains over training from scratch
# 4. The 80:20 split provides optimal results
#
# ### Limitations:
# - Small dataset (1097 images) → risk of overfitting
# - Slice-level splitting may cause **data leakage** (slices from same patient in train/test)
# - No patient-level split validation
# - Limited to 3-class classification
# - Results may not generalize to other CT-scan datasets
#
# ### Future Work (Paper Section 6):
# - Synthetic data generation (GANs) for augmentation
# - Larger and more diverse datasets
# - Patient-level splitting for honest evaluation
# - Additional architectures (DenseNet, Vision Transformers)

# %%
def final_summary(all_results, save_dir):
    """Publication-quality summary of all experiments."""
    print("\n" + "=" * 90)
    print(" FINAL EXPERIMENT SUMMARY — Lung-EffNet Reproduction")
    print("=" * 90)

    all_dfs = {}
    for exp_name, res_dict in [
        ("With Augmentation", all_experiment_results),
        ("Without Augmentation", noaug_results),
        ("From Scratch", scratch_results),
        ("Comparison Architectures", comparison_results),
    ]:
        if res_dict:
            df = build_results_table(res_dict, CONFIG['class_names'])
            all_dfs[exp_name] = df
            print(f"\n{'─'*60}")
            print(f"  {exp_name}")
            print(f"{'─'*60}")
            print(df.to_string(index=False))

    # Save all results
    with open(save_dir / 'full_results.json', 'w') as f:
        summary = {}
        for name, res in {**all_experiment_results, **noaug_results,
                          **scratch_results, **comparison_results}.items():
            summary[name] = {
                'accuracy': res.get('accuracy', 0),
                'f1_macro': res.get('f1_macro', 0),
                'auc_macro': res.get('auc_macro', 0),
                'params_M': round(res.get('total_params', 0) / 1e6, 2),
                'training_time_s': round(res.get('training_time_s', 0), 1),
            }
        json.dump(summary, f, indent=2)

    print(f"\n✅ All results saved to {save_dir}")

final_summary(all_experiment_results, RESULTS_DIR)

# %% [markdown]
# ## 15. References
#
# ```bibtex
# @article{raza2023lungeffnet,
#   title={Lung-EffNet: Lung cancer classification using EfficientNet from CT-scan images},
#   author={Raza, Rehan and Zulfiqar, Fatima and Khan, Muhammad Owais and Arif, Muhammad
#           and Alvi, Atif and Iftikhar, Muhammad Aksam and Alam, Tanvir},
#   journal={Engineering Applications of Artificial Intelligence},
#   volume={126},
#   pages={106902},
#   year={2023},
#   publisher={Elsevier},
#   doi={10.1016/j.engappai.2023.106902}
# }
#
# @article{tan2019efficientnet,
#   title={EfficientNet: Rethinking model scaling for convolutional neural networks},
#   author={Tan, Mingxing and Le, Quoc},
#   journal={ICML},
#   pages={6105--6114},
#   year={2019}
# }
#
# @misc{iqothnccd2020,
#   title={The IQ-OTH/NCCD Lung Cancer Dataset},
#   author={Alyasriy, H. and Muayed, A.},
#   year={2020},
#   url={https://www.kaggle.com/datasets/hamdallak/the-iqothnccd-lung-cancer-dataset}
# }
# ```
#
# ## Appendix: Full Configuration Dump

# %%
print("\n📋 Full Configuration Dump:")
print(json.dumps(CONFIG, indent=2))
print(f"\n  PyTorch       : {torch.__version__}")
print(f"  CUDA          : {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
print(f"  cuDNN         : {torch.backends.cudnn.version() if torch.cuda.is_available() else 'N/A'}")
print(f"  Seed          : {SEED}")
print(f"  Deterministic : {torch.backends.cudnn.deterministic}")
print(f"  Benchmark     : {torch.backends.cudnn.benchmark}")

print("\n🎉 Notebook complete!")
