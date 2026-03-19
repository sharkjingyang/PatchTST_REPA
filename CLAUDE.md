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
  --feature_extractor chronos --lambda_contrastive 0.5

# PatchTST_REPA_Fusion (Patch Fusion + Contrastive Loss)
python -u run_longExp.py --is_training 1 --model PatchTST_REPA_Fusion --data custom \
  --root_path ./dataset/ --data_path weather.csv \
  --features M --seq_len 336 --pred_len 96 \
  --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 \
  --patch_len 16 --stride 8 --batch_size 128 --learning_rate 0.0001 \
  --feature_extractor chronos --lambda_contrastive 0.5 \
  --patch_fusion_type split_MLP  # Use separable projection (fewer params)

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
sh ./scripts/PatchTST.sh        # Baseline
sh ./scripts/mantis.sh          # PatchTST_REPA + Mantis
sh ./scripts/Chronos2.sh        # PatchTST_REPA + Chronos (patch_wise)
sh ./scripts/Chronos2_head.sh   # Chronos2_head (frozen encoder + head)
sh ./scripts/Chronos_original.sh # Chronos2 direct inference (no training)
```

## Architecture

### Four Models
1. `PatchTST` - Original PatchTST (baseline)
2. `PatchTST_REPA` - PatchTST + MLP Projector + contrastive loss
3. `PatchTST_REPA_Fusion` - PatchTST + Patch Fusion branch (d_channel=128) + contrastive loss
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

### Key Components
- **Patch_Fusion_MLP**: Projects encoder features to output patch space
  - `fusion_MLP`: Joint projection (d_model × patch_num → d_channel × output_patch_num)
  - `split_MLP`: Separable projection (d_model → d_channel, then patch_num → output_patch_num)
- **Patch_Split_MLP**: Separable version with fewer parameters
- **PatchwiseHead**: Lightweight head using shared ResidualBlock per patch

### Head Types
- `flatten`: Flatten_Head (standard)
- `patch_wise`: PatchwiseHead (~11K params with d_channel=128)
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
| `lambda_contrastive` | Contrastive loss weight | 0.5 |
| `use_future_patch` | Chronos2_head: 0=past tokens+Flatten, 1=future tokens+Patchwise | 0 |

## Parameter Comparison (d_model=16, seq_len=336, pred_len=96)

| Model | PatchFusionMLP | Total (excl. Chronos) |
|-------|----------------|----------------------|
| PatchTST | - | ~24K |
| PatchTST_REPA | - | ~326K |
| PatchTST_REPA_Fusion (fusion_MLP) | 504K | ~735K |
| PatchTST_REPA_Fusion (split_MLP) | 2.4K | ~233K |
| Chronos2_head (use_future_patch=0) | - | ~1.5M (Flatten_Head) |
| Chronos2_head (use_future_patch=1) | - | ~450K (PatchwiseHead) |

Using `split_MLP` reduces PatchFusionMLP parameters by ~99.5%.

## Directory Structure

```
PatchTST_REPA/
├── run_longExp.py              # Main entry point
├── test_Chronos2_direct.py      # Chronos2 direct inference test (no training)
├── layers/
│   ├── PatchTST_backbone.py   # Core model (Patch_Fusion_MLP, Patch_Split_MLP, Flatten_Head, PatchwiseHead)
│   ├── PatchTST_layers.py
│   ├── RevIN.py
│   └── Tivit.py
├── models/
│   ├── PatchTST.py            # PatchTST / PatchTST_REPA / PatchTST_REPA_Fusion
│   └── Chronos2_head.py       # Chronos2 (frozen) + Flatten/Patchwise head
├── exp/
│   └── exp_main.py            # Training & evaluation
├── scripts/                    # Training scripts
└── dataset/                    # Data files (not tracked in git)
```
