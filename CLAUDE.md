# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

Extended implementation of PatchTST with feature alignment using contrastive learning. Supports four feature extractors: **TiViT**, **Mantis**, **Chronos2**, and **Chronos2_head** (frozen encoder + prediction head).

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Download Data
Place CSV files in `./dataset/` (ETTm1, ETTm2, ETTh1, ETTh2, electricity, traffic, weather, etc.)

### Training Commands

```bash
# Original PatchTST (baseline)
python -u run_longExp.py --is_training 1 --model PatchTST --data custom \
  --root_path ./dataset/ --data_path weather.csv \
  --features M --seq_len 336 --pred_len 96 \
  --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 \
  --patch_len 16 --stride 8 --batch_size 128 --learning_rate 0.0001

# PatchTST_REPA (MLP Projector + Contrastive Loss)
python -u run_longExp.py --is_training 1 --model PatchTST_REPA --data custom \
  --root_path ./dataset/ --data_path weather.csv \
  --features M --seq_len 336 --pred_len 96 \
  --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 \
  --patch_len 16 --stride 8 --batch_size 128 --learning_rate 0.0001 \
  --feature_extractor chronos --lambda_contrastive 0.1 \
  --contrastive_type mean_pool  # patch counts differ (41 vs 6), must use mean_pool

# PatchTST_REPA_Fusion (Patch Fusion + Contrastive Loss)
# Recommended: split_MLP + patch_wise + lambda=0.1
python -u run_longExp.py --is_training 1 --model PatchTST_REPA_Fusion --data custom \
  --root_path ./dataset/ --data_path weather.csv \
  --features M --seq_len 336 --pred_len 96 \
  --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 \
  --patch_len 16 --stride 8 --batch_size 128 --learning_rate 0.0001 \
  --feature_extractor chronos --lambda_contrastive 0.1 \
  --patch_fusion_type split_MLP --contrastive_type patch_wise

# Chronos2_head (frozen Chronos2 encoder + prediction head)
# use_future_patch=0: past tokens + Flatten_Head (~1.5M trainable params)
# use_future_patch=1: future tokens only + PatchwiseHead (~450K trainable params)
python -u run_longExp.py --is_training 1 --model Chronos2_head --data custom \
  --root_path ./dataset/ --data_path weather.csv \
  --features M --seq_len 336 --pred_len 96 \
  --patch_len 16 --batch_size 128 --learning_rate 0.0001 \
  --use_future_patch 0
```

Or use shell scripts:
```bash
sh ./scripts/PatchTST.sh              # Baseline
sh ./scripts/mantis.sh               # PatchTST_REPA + Mantis
sh ./scripts/Chronos2.sh             # PatchTST_REPA + Chronos (patch_wise)
sh ./scripts/Chronos2_featureHead.sh # Chronos2_head (frozen encoder + head)
sh ./scripts/Chronos2_zeroshot.sh    # Chronos2 direct inference (no training)
```

## Architecture

### Four Models
1. `PatchTST` - Original PatchTST (baseline)
2. `PatchTST_REPA` - PatchTST + MLP Projector + contrastive loss
3. `PatchTST_REPA_Fusion` - PatchTST + Patch Fusion branch + contrastive loss
4. `Chronos2_head` - Chronos2 (frozen) + prediction head

### Chronos2_head Architecture

Chronos2_head uses a frozen Chronos2 encoder to extract features, then a trainable prediction head. **All outputs are denormalized back to original scale** using InstanceNorm inverse (same as Chronos2's native forward pass).

| Mode | Features | Head | Trainable Params |
|------|----------|------|------------------|
| `use_future_patch=0` | Past tokens (21 patches) | Flatten_Head | ~1.5M |
| `use_future_patch=1` | Future tokens only (6 patches) | PatchwiseHead | ~450K |

**Flow (use_future_patch=0)**:
```
Input x: (bs, seq_len, nvars)
  ↓ Chronos2.embed(x) - frozen
Feature: (bs, nvars, 21, 768)  [21 = seq_len/patch_len]
  ↓ permute(0,1,3,2): (bs, nvars, 768, 21)
  ↓ Flatten_Head (individual=True)
  ↓ flatten: (bs*nvars, 768*21)
  ↓ linear: (bs*nvars, pred_len)
  ↓ InstanceNorm.inverse (loc, scale from embed)
Output: (bs, pred_len, nvars) - denormalized
```

**Flow (use_future_patch=1)** - Like Chronos2's native prediction:
```
Input x: (bs, seq_len, nvars)
  ↓ Chronos2.model.encode(x, num_output_patches) - frozen
Feature: (bs, nvars, 28, 768)  [21 + 1 (REG) + 6 (future)]
  ↓ extract ONLY future tokens: hidden_states[:, -6:, :]
Feature: (bs, nvars, 6, 768)
  ↓ PatchwiseHead (per-patch prediction)
  ↓ ResidualBlock per patch: d_model -> d_ff -> output_patch_size
  ↓ InstanceNorm.inverse (loc, scale from encode)
Output: (bs, pred_len, nvars) - denormalized
```

**Note on InstanceNorm**: Chronos2 applies instance normalization to input before encoding. The normalization parameters (loc, scale) are captured during encoding and used to denormalize the output. Formula: `denormalized = normalized * scale + loc`.

### Chronos2 Feature Extraction in REPA Models

Both `PatchTST_REPA` and `PatchTST_REPA_Fusion` use `Chronos2.model.encode(batch_x, num_output_patches)` to extract features from the **input** (batch_x), then take the last `num_output_patches` future tokens as `zs_tilde`.

```
batch_x: (bs, seq_len, nvars)
  → reshape: (bs*nvars, seq_len)
  → encode(num_output_patches=pred_len//16)
  → last_hidden_state: (bs*nvars, 21+1+6, 768)
  → future tokens [-6:]: (bs, nvars, 6, 768)  ← zs_tilde
```

This aligns with `PatchTST_REPA_Fusion`'s Patch Fusion output `(bs, nvars, 6, 768)` for patch-wise contrastive loss.
`PatchTST_REPA` (no Fusion) has patch_num=41 vs Chronos2's 6, so must use `mean_pool` mode.

### Key Components
- **`build_mlp(hidden_size, z_dim, projected_dim=512)`**: 统一的对齐 MLP，结构为 Linear→SiLU→Linear→SiLU→Linear，用于所有 `alignment_mlp`
- **Patch_Fusion_MLP**: 联合投影 `d_model*patch_num → d_model*output_patch_num`（`fusion_MLP` 模式）
- **`nn.Linear(patch_num, output_patch_num)`**: `split_MLP` 模式直接内联，仅投影时间维度，保留 `d_model` 不变，参数极少（~258）
- **TransformerDecoder**: 在 patch fusion 后对 `(bs*nvars, output_patch_num, d_model)` 做自注意力
- **alignment_mlp**: `build_mlp(d_model, d_extractor)`，将 patch fusion 输出投影到特征提取器空间用于对比损失
- **PatchwiseHead**: Lightweight head using shared ResidualBlock per patch

### alignment_mlp 统一规范
两种模式均使用 `build_mlp(d_model, d_extractor, projected_dim=512)`：
- `PatchTST_REPA`（无 fusion）：输入 `(bs*nvars*patch_num, d_model)` → 输出 `(bs*nvars*patch_num, d_extractor)`
- `PatchTST_REPA_Fusion`：输入 `(bs*nvars*output_patch_num, d_model)` → 输出 `(bs*nvars*output_patch_num, d_extractor)`

`d_extractor` 由特征提取器决定（Mantis=256，Chronos2/TiViT=768），`projector_dim` 概念已废弃。

### Head Types
- `flatten`: Flatten_Head (standard)
- `patch_wise`: PatchwiseHead
- `quantile`: Quantile_Head for probabilistic forecasting

### Output Shapes
- PatchTST: `(batch, pred_len, nvars)`
- PatchTST_REPA_Fusion: `(batch, pred_len, nvars)` + `(batch, nvars, output_patch_num, d_extractor)` for contrastive loss

## Key Hyperparameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model` | PatchTST / PatchTST_REPA / PatchTST_REPA_Fusion / Chronos2_head | - |
| `patch_fusion_type` | fusion_MLP (joint) or split_MLP (separable) | fusion_MLP |
| `contrastive` | Enable contrastive learning loss (1/0) | auto |
| `feature_extractor` | tivit / mantis / chronos | mantis |
| `head_type` | flatten / patch_wise / quantile | flatten |
| `lambda_contrastive` | Contrastive loss weight | 0.5 (推荐 0.1) |
| `contrastive_type` | mean_pool / patch_wise | mean_pool |
| `use_future_patch` | Chronos2_head: 0=past tokens+Flatten, 1=future tokens+Patchwise | 0 |

## Parameter Comparison

### PatchTST_REPA_Fusion (split_MLP, d_model=128, seq_len=336, pred_len=96, nvars=21)

| Module | Params |
|--------|-------:|
| backbone (encoder) | 272,514 |
| patch_fusion_mlp (`nn.Linear(42,6)`) | 258 |
| transformer_decoder | 99,585 |
| alignment_mlp (`build_mlp(128→512→512→768)`) | 722,688 |
| head (Flatten_Head) | 73,824 |
| revin_layer | 42 |
| **TOTAL** | **1,168,911** |

### 各模型规模对比 (d_model=128, seq_len=336, pred_len=96)

| Model | patch_fusion_mlp | TOTAL |
|-------|-----------------|-------|
| PatchTST | - | ~276K |
| PatchTST_REPA | - | ~1.1M |
| PatchTST_REPA_Fusion (fusion_MLP) | ~670K | ~1.8M |
| PatchTST_REPA_Fusion (split_MLP) | 258 | ~1.2M |
| Chronos2_head (use_future_patch=0) | - | ~1.5M (Flatten_Head) |
| Chronos2_head (use_future_patch=1) | - | ~450K (PatchwiseHead) |

`split_MLP` 的 `patch_fusion_mlp` 仅 258 参数（vs `fusion_MLP` 的 ~670K），主要参数消耗在 `alignment_mlp`（build_mlp 三层）。

## Directory Structure

```
PatchTST_REPA/
├── run_longExp.py              # Main entry point
├── layers/
│   ├── PatchTST_backbone.py   # Core model (build_mlp, Patch_Fusion_MLP, TransformerDecoder, Flatten_Head, PatchwiseHead)
│   ├── PatchTST_layers.py
│   ├── RevIN.py
│   └── Tivit.py
├── models/
│   ├── PatchTST.py            # PatchTST / PatchTST_REPA / PatchTST_REPA_Fusion
│   ├── Chronos2_head.py       # Chronos2 (frozen) + Flatten/Patchwise head
│   └── Chronos2_zeroshot.py   # Chronos2 direct inference test (no training)
├── exp/
│   └── exp_main.py            # Training & evaluation
├── scripts/                    # Training scripts
└── dataset/                    # Data files (not tracked in git)
```
