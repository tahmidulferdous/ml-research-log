# %% [markdown]
# ## Sequential Split Experiment — EfficientNet B0–B4
#
# **Purpose**: Demonstrate data leakage impact by comparing:
# - Paper's method: shuffle + 80/20 image-level split (leaky)
# - Honest method: sequential 70/20/10 split (no shuffle)
#
# **Everything else identical**: same architecture, same augmentation,
# same hyperparameters, same evaluation pipeline.

# %%
# ── Requires part1_setup.py and part2_model_train.py to be run first ──
# If running as standalone, uncomment:
# exec(open('part1_setup.py').read())
# exec(open('part2_model_train.py').read())

import gc
import pandas as pd

# %% [markdown]
# ### Create Sequential Split DataLoaders

# %%
# Reload fresh records (split_dataset modifies in-place)
seq_records = load_dataset(DATASET_ROOT)
seq_all_records = split_dataset_sequential(seq_records, train_ratio=0.70, val_ratio=0.20)

print("\n📊 Sequential Split Summary (70/20/10 — No Shuffle):")
print(f"{'Split':<8} {'Benign':>8} {'Malignant':>10} {'Normal':>8} {'Total':>8}")
print("-" * 44)
for sp in ['train', 'val', 'test']:
    sub = [r for r in seq_all_records if r['split'] == sp]
    counts = Counter(r['label'] for r in sub)
    total = sum(counts.values())
    print(f"{sp:<8} {counts.get(0,0):>8} {counts.get(1,0):>10} {counts.get(2,0):>8} {total:>8}")

# Create dataloaders WITH augmentation (same as paper experiments)
seq_train_ld, seq_val_ld, seq_test_ld = make_dataloaders(
    seq_all_records, CONFIG['img_size'], CONFIG['batch_size'], augment=True)

# %% [markdown]
# ### Run EfficientNet B0–B4 with Sequential Split

# %%
seq_results = {}

for variant in ['b0', 'b1', 'b2', 'b3', 'b4']:
    print(f"\n{'#'*70}")
    print(f" EfficientNet-{variant.upper()} | SEQUENTIAL SPLIT (No Shuffle)")
    print(f"{'#'*70}")
    model = build_lung_effnet(variant, num_classes=CONFIG['num_classes'],
                              dropout=CONFIG['dropout_rate'], pretrained=True)
    _, results = train_full(model, seq_train_ld, seq_val_ld, seq_test_ld,
                           f'EffNet_{variant.upper()}_sequential', DEVICE,
                           epochs=CONFIG['epochs'], lr=CONFIG['learning_rate'],
                           save_dir=RESULTS_DIR / f'effnet_{variant}_sequential')
    metrics = full_evaluation(results, CONFIG['class_names'],
                             RESULTS_DIR / f'effnet_{variant}_sequential')
    seq_results[f'EffNet-{variant.upper()} (seq)'] = {**metrics, **results}
    del model; torch.cuda.empty_cache(); gc.collect()

# %% [markdown]
# ### Results: Sequential Split

# %%
seq_df = build_results_table(seq_results, CONFIG['class_names'])
print("\n" + "=" * 90)
print(" RESULTS: EfficientNet B0–B4 | SEQUENTIAL SPLIT (No Shuffle)")
print("=" * 90)
print(seq_df.to_string(index=False))
seq_df.to_csv(RESULTS_DIR / 'results_sequential_split.csv', index=False)

# %% [markdown]
# ### Side-by-Side Comparison: Leaky vs Honest

# %%
# Paper's reported values (with shuffle + augmentation)
paper_results = {
    'EffNet-B0': {'acc': 93.67, 'prec': 87.20, 'rec': 89.48, 'f1': 88.18},
    'EffNet-B1': {'acc': 99.10, 'prec': 99.22, 'rec': 97.22, 'f1': 98.16},
    'EffNet-B2': {'acc': 99.10, 'prec': 99.22, 'rec': 97.22, 'f1': 98.16},
    'EffNet-B3': {'acc': 98.19, 'prec': 98.48, 'rec': 94.44, 'f1': 96.19},
    'EffNet-B4': {'acc': 97.29, 'prec': 96.44, 'rec': 92.66, 'f1': 94.29},
}

print("\n" + "=" * 90)
print(" COMPARISON: Paper (Leaky) vs Sequential (Honest)")
print("=" * 90)
print(f"{'Model':<15} {'Paper Acc':>10} {'Seq Acc':>10} {'Δ Acc':>8} {'Paper F1':>10} {'Seq F1':>10} {'Δ F1':>8}")
print("-" * 75)

for variant in ['B0', 'B1', 'B2', 'B3', 'B4']:
    paper = paper_results[f'EffNet-{variant}']
    seq_key = f'EffNet-{variant} (seq)'
    if seq_key in seq_results:
        seq = seq_results[seq_key]
        seq_acc = seq['accuracy'] * 100
        seq_f1 = seq['f1_macro'] * 100
        d_acc = seq_acc - paper['acc']
        d_f1 = seq_f1 - paper['f1']
        print(f"EffNet-{variant:<8} {paper['acc']:>9.2f}% {seq_acc:>9.2f}% {d_acc:>+7.2f}% {paper['f1']:>9.2f}% {seq_f1:>9.2f}% {d_f1:>+7.2f}%")

# %%
# Save comparison
comparison_rows = []
for variant in ['B0', 'B1', 'B2', 'B3', 'B4']:
    paper = paper_results[f'EffNet-{variant}']
    seq_key = f'EffNet-{variant} (seq)'
    if seq_key in seq_results:
        seq = seq_results[seq_key]
        comparison_rows.append({
            'Model': f'EffNet-{variant}',
            'Paper_Acc': paper['acc'],
            'Sequential_Acc': round(seq['accuracy'] * 100, 2),
            'Delta_Acc': round(seq['accuracy'] * 100 - paper['acc'], 2),
            'Paper_F1': paper['f1'],
            'Sequential_F1': round(seq['f1_macro'] * 100, 2),
            'Delta_F1': round(seq['f1_macro'] * 100 - paper['f1'], 2),
        })

comp_df = pd.DataFrame(comparison_rows)
comp_df.to_csv(RESULTS_DIR / 'leaky_vs_sequential_comparison.csv', index=False)
print(f"\n✅ Comparison saved to {RESULTS_DIR / 'leaky_vs_sequential_comparison.csv'}")

print("\n✅ Part 5 (Sequential Split Experiment) complete.")
