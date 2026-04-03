# Experiments Record

Results on ETTh1 unless noted. All runs: `seq_len=336`, `batch_size=128`, `lr=0.0001`, `train_epochs=20`.

## Results Table

| # | Date | Model | Data | pred_len | d_model | d_ff | e_layers | n_heads | patch_len | stride | feature_extractor | contrastive_type | λ_c / λ_t / λ_a | extra | MSE | MAE | Notes |
|---|------|-------|------|----------|---------|------|----------|---------|-----------|--------|-------------------|------------------|-----------------|-------|-----|-----|-------|
| 1 | | PatchTST | ETTh1 | 96 | 128 | 256 | 3 | 16 | 16 | 8 | - | - | - | - | | | baseline |
| 2 | | PatchTST | ETTh1 | 192 | 128 | 256 | 3 | 16 | 16 | 8 | - | - | - | - | | | |
| 3 | | PatchTST | ETTh1 | 336 | 128 | 256 | 3 | 16 | 16 | 8 | - | - | - | - | | | |
| 4 | | PatchTST | ETTh1 | 720 | 128 | 256 | 3 | 16 | 16 | 8 | - | - | - | - | | | |
| 5 | | PatchTST_REPA | ETTh1 | 96 | 128 | 256 | 3 | 16 | 16 | 16 | chronos | patch_wise_cos | 0.1 / - / - | - | | | stride=16 → patch_num=21 |
| 6 | | PatchTST_REPA_Fusion | ETTh1 | 96 | 128 | 256 | 3 | 16 | 16 | 8 | chronos | patch_wise_cos | 0.1 / - / - | split_MLP | | | |
| 7 | | PatchTST_REPA_Fusion | ETTh1 | 96 | 128 | 256 | 3 | 16 | - | - | chronos | patch_wise_cos | 0.1 / - / - | none | | | patch_len auto |
| 8 | | PatchTST_future_align | ETTh1 | 96 | 128 | 256 | 3 | 16 | - | - | - | - | - / 0.5 / 0.1 | - | | | patch auto |
| 9 | | Chronos2_head | ETTh1 | 96 | 128 | - | - | - | 16 | - | - | - | - | future+proj_down | | | teacher-forcing |

---

## Column Reference

| Column | Description |
|--------|-------------|
| `feature_extractor` | `tivit` / `mantis` / `chronos` / `-` |
| `contrastive_type` | `mean_pool` / `patch_wise_cos` / `patch_wise_mse` / `-` |
| `λ_c / λ_t / λ_a` | contrastive weight / teacher loss weight / alignment loss weight |
| `extra` | `patch_fusion_type` (fusion_MLP / split_MLP / none), `proj_down`, etc. |

## Key Configs

```
# patch_len=16, stride=16 → patch_num=21 (matches Chronos2 past tokens for patch_wise_cos)
# patch_fusion_type=none  → patch_len auto = seq_len // (pred_len // 16)
# future_align            → patch_len auto, output_patch_num = pred_len // 16
```
