# %% [markdown]
# ## 6. Main Experiments — EfficientNet B0–B4

# %%
# ── Run all 5 EfficientNet variants (Paper Table 9) ──
all_experiment_results = {}

# Create dataloaders WITH augmentation (Paper's primary experiments)
train_ld, val_ld, test_ld = make_dataloaders(
    all_records, CONFIG['img_size'], CONFIG['batch_size'], augment=True)

for variant in ['b0', 'b1', 'b2', 'b3', 'b4']:
    print(f"\n{'#'*70}")
    print(f" EfficientNet-{variant.upper()} WITH Data Augmentation")
    print(f"{'#'*70}")
    model = build_lung_effnet(variant, num_classes=CONFIG['num_classes'],
                              dropout=CONFIG['dropout_rate'], pretrained=True)
    _, results = train_full(model, train_ld, val_ld, test_ld,
                           f'EfficientNet_{variant.upper()}', DEVICE,
                           epochs=CONFIG['epochs'], lr=CONFIG['learning_rate'],
                           save_dir=RESULTS_DIR / f'effnet_{variant}')
    metrics = full_evaluation(results, CONFIG['class_names'],
                             RESULTS_DIR / f'effnet_{variant}')
    all_experiment_results[f'EffNet-{variant.upper()} (aug)'] = {**metrics, **results}
    del model; torch.cuda.empty_cache(); gc.collect()

# %% [markdown]
# ## 7. Results Comparison (Paper Tables 5, 6, 9)

# %%
import pandas as pd

def build_results_table(results_dict, class_names):
    """Build comparison table like Paper Tables 5/6/9."""
    rows = []
    for name, r in results_dict.items():
        row = {
            'Model': name,
            'Accuracy (%)': round(r['accuracy'] * 100, 2),
            'Precision (%)': round(r['precision_macro'] * 100, 2),
            'Recall (%)': round(r['recall_macro'] * 100, 2),
            'F1-Score (%)': round(r['f1_macro'] * 100, 2),
            'AUC': round(r['auc_macro'], 4),
            'Params (M)': round(r['total_params'] / 1e6, 2),
            'Train Time (s)': round(r['training_time_s'], 1),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    return df

results_df = build_results_table(all_experiment_results, CONFIG['class_names'])
print("\n" + "=" * 90)
print(" RESULTS: EfficientNet B0–B4 WITH Augmentation (cf. Paper Table 9)")
print("=" * 90)
print(results_df.to_string(index=False))
results_df.to_csv(RESULTS_DIR / 'results_with_augmentation.csv', index=False)

# ── Paper comparison values (Table 9) ──
paper_results = {
    'EffNet-B0': {'acc': 93.67, 'prec': 87.20, 'rec': 89.48, 'f1': 88.18},
    'EffNet-B1': {'acc': 99.10, 'prec': 99.22, 'rec': 97.22, 'f1': 98.16},
    'EffNet-B2': {'acc': 99.10, 'prec': 99.22, 'rec': 97.22, 'f1': 98.16},
    'EffNet-B3': {'acc': 98.19, 'prec': 98.48, 'rec': 94.44, 'f1': 96.19},
    'EffNet-B4': {'acc': 97.29, 'prec': 96.44, 'rec': 92.66, 'f1': 94.29},
}
print("\n📊 Paper's reported values (Table 9, with augmentation):")
for k, v in paper_results.items():
    print(f"  {k}: Acc={v['acc']}%, Prec={v['prec']}%, Rec={v['rec']}%, F1={v['f1']}%")

# %% [markdown]
# ## 8. Ablation Study 1: With vs Without Augmentation (Paper Table 9)

# %%
# Create dataloaders WITHOUT augmentation
train_ld_noaug, val_ld_noaug, test_ld_noaug = make_dataloaders(
    all_records, CONFIG['img_size'], CONFIG['batch_size'], augment=False)

noaug_results = {}
for variant in ['b0', 'b1', 'b2', 'b3', 'b4']:
    print(f"\n{'#'*70}")
    print(f" EfficientNet-{variant.upper()} WITHOUT Augmentation")
    print(f"{'#'*70}")
    model = build_lung_effnet(variant, pretrained=True)
    _, results = train_full(model, train_ld_noaug, val_ld_noaug, test_ld_noaug,
                           f'EffNet_{variant.upper()}_noaug', DEVICE,
                           epochs=CONFIG['epochs'], lr=CONFIG['learning_rate'],
                           save_dir=RESULTS_DIR / f'effnet_{variant}_noaug')
    metrics = full_evaluation(results, CONFIG['class_names'],
                             RESULTS_DIR / f'effnet_{variant}_noaug')
    noaug_results[f'EffNet-{variant.upper()} (no aug)'] = {**metrics, **results}
    del model; torch.cuda.empty_cache(); gc.collect()

noaug_df = build_results_table(noaug_results, CONFIG['class_names'])
print("\n" + "=" * 90)
print(" ABLATION: Without Augmentation (cf. Paper Table 9, left columns)")
print("=" * 90)
print(noaug_df.to_string(index=False))

# %% [markdown]
# ## 9. Ablation Study 2: With vs Without Transfer Learning (Paper Table 11)

# %%
scratch_results = {}
for variant in ['b0', 'b1']:  # B0, B1 as representative (full set takes long)
    print(f"\n{'#'*70}")
    print(f" EfficientNet-{variant.upper()} FROM SCRATCH (no pretrained)")
    print(f"{'#'*70}")
    model = build_lung_effnet(variant, pretrained=False)
    _, results = train_full(model, train_ld, val_ld, test_ld,
                           f'EffNet_{variant.upper()}_scratch', DEVICE,
                           epochs=CONFIG['epochs'], lr=CONFIG['learning_rate'],
                           save_dir=RESULTS_DIR / f'effnet_{variant}_scratch')
    metrics = full_evaluation(results, CONFIG['class_names'],
                             RESULTS_DIR / f'effnet_{variant}_scratch')
    scratch_results[f'EffNet-{variant.upper()} (scratch)'] = {**metrics, **results}
    del model; torch.cuda.empty_cache(); gc.collect()

print("\n📊 Transfer Learning vs From Scratch (Paper Table 11):")
scratch_df = build_results_table(scratch_results, CONFIG['class_names'])
print(scratch_df.to_string(index=False))

# %% [markdown]
# ## 10. Ablation Study 3: Different Data Splits (Paper Table 10)

# %%
split_results = {}
for ratio_name, train_r in [('70:30', 0.70), ('80:20', 0.80), ('90:10', 0.90)]:
    print(f"\n{'#'*70}")
    print(f" EfficientNet-B1 | Split {ratio_name}")
    print(f"{'#'*70}")
    cfg_copy = dict(CONFIG)
    cfg_copy['train_ratio'] = train_r
    temp_records = split_dataset(records, train_ratio=train_r)
    t_ld, v_ld, te_ld = make_dataloaders(temp_records, CONFIG['img_size'],
                                          CONFIG['batch_size'], augment=True)
    model = build_lung_effnet('b1', pretrained=True)
    _, results = train_full(model, t_ld, v_ld, te_ld,
                           f'EffNet_B1_split_{ratio_name.replace(":","")}', DEVICE,
                           epochs=CONFIG['epochs'], lr=CONFIG['learning_rate'],
                           save_dir=RESULTS_DIR / f'effnet_b1_split_{ratio_name.replace(":","_")}')
    metrics = full_evaluation(results, CONFIG['class_names'],
                             RESULTS_DIR / f'effnet_b1_split_{ratio_name.replace(":","_")}')
    split_results[f'B1 ({ratio_name})'] = {**metrics, **results}
    del model; torch.cuda.empty_cache(); gc.collect()

print("\n📊 Data Split Ablation (Paper Table 10):")
split_df = build_results_table(split_results, CONFIG['class_names'])
print(split_df.to_string(index=False))

# %% [markdown]
# ## 11. Comparison with Other Architectures (Paper Table 8)

# %%
comparison_results = {}
archs = [('resnet50', 'ResNet50'), ('mobilenet_v2', 'MobileNetV2'),
         ('mobilenet_v3_small', 'MobileNetV3Small')]
for arch_key, arch_name in archs:
    print(f"\n{'#'*70}")
    print(f" {arch_name}")
    print(f"{'#'*70}")
    model = build_comparison_model(arch_key, num_classes=CONFIG['num_classes'])
    _, results = train_full(model, train_ld, val_ld, test_ld,
                           arch_name, DEVICE,
                           epochs=CONFIG['epochs'], lr=CONFIG['learning_rate'],
                           save_dir=RESULTS_DIR / arch_key)
    metrics = full_evaluation(results, CONFIG['class_names'], RESULTS_DIR / arch_key)
    comparison_results[arch_name] = {**metrics, **results}
    del model; torch.cuda.empty_cache(); gc.collect()

comp_df = build_results_table(comparison_results, CONFIG['class_names'])
print("\n📊 Comparison Architectures (Paper Table 8):")
print(comp_df.to_string(index=False))
