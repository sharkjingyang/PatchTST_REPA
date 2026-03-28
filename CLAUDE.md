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

# PatchTST_REPA_Fusion (none mode: 无 fusion MLP，patch_len 自动推导)
# patch_len/stride/padding_patch 参数会被忽略，自动计算：patch_len = seq_len // output_patch_num
python -u run_longExp.py --is_training 1 --model PatchTST_REPA_Fusion --data custom \
  --root_path ./dataset/ --data_path weather.csv \
  --features M --seq_len 336 --pred_len 96 \
  --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 \
  --batch_size 128 --learning_rate 0.0001 \
  --feature_extractor chronos --lambda_contrastive 0.1 \
  --patch_fusion_type none --contrastive_type patch_wise

# Chronos2_head (frozen Chronos2 encoder + prediction head)
# use_future_patch=0: past tokens + Flatten_Head (pred_len=96: ~1.55M, pred_len=720: ~11.6M, scales linearly)
# use_future_patch=1: future tokens only + PatchwiseHead (~314K trainable params, fixed regardless of pred_len)
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
sh ./scripts/PatchTST_FM_zeroshot.sh # PatchTST-FM-R1 zero-shot inference (no training)
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
| `use_future_patch=0` | Past tokens (21 patches) | Flatten_Head | ~1.55M (pred_len=96) / ~11.6M (pred_len=720), linear in pred_len |
| `use_future_patch=1` | Future tokens only (6 patches) | PatchwiseHead | ~314K (fixed, independent of pred_len) |

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
- **`none` 模式**: 无 fusion MLP，`patch_len` 自动推导为 `seq_len // output_patch_num`，使 patch_num 天然等于 output_patch_num；`patch_len/stride/padding_patch` 参数被忽略
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

### PatchTST-FM-R1 (Zero-shot Baseline)

IBM Research 发布的时序预测基础模型（arXiv:2602.06909，2026），不需要任何训练，直接零样本推理。

| 属性 | 值 |
|------|-----|
| 参数量 | ~260M |
| 上下文长度 | 8192 |
| Patch 大小 | 16 |
| d_model | 384（代码默认值，实际以 `model.config` 为准） |
| 输出 | 99 个分位数（取 `quantile_levels=[0.5]` 即中位数） |

**安装**：
```bash
pip install git+https://github.com/ibm-granite/granite-tsfm.git@patchtst-fm
pip install torch==你原来的版本   # 安装时 torch 会被降级，忽略 torch<2.9 冲突警告
```

**架构**：
```
输入 (B, T)
  ↓ RevIN + asinh 归一化
  ↓ Patching: T/16 个 patch，每 patch 拼接 inv_mask → (32,)
  ↓ in_layer ResidualBlock: 32 → d_model
  ↓ Learned Positional Embedding
  ↓ TransformerBlock × n_layer          ← encoder hidden state 在此
  ↓ out_layer ResidualBlock: d_model → 16×100（预测头）
  ↓ softplus + cumsum → 99 个单调分位数
  ↓ RevIN 逆变换
输出 (B, 99, pred_len)
```

**关键特性**：
- **预测即重建**：mask 预测期后重建，非自回归
- **变长输入**：支持不同长度序列，内部 RevIN + asinh 归一化，无需手动标准化
- **单调分位数**：99 个输出 + 1 个锚点，softplus + cumsum 保证 q10 ≤ q50 ≤ q90
- **Channel-independent**：`models/PatchTST_FM_zeroshot.py` 中将 `(bs, seq_len, nvars)` 展平为 `bs*nvars` 条 1D 序列分别推理
- **类**：`tsfm_public.models.patchtst_fm.PatchTSTFMForPrediction`（非 HuggingFace transformers 标准类）
- **本地路径**：`./Patchtst-Fm-R1`（下载后放在该目录，HuggingFace ID: `ibm-research/patchtst-fm-r1`）

**推理接口**：
```python
output = model(inputs=list_of_1d_tensors, prediction_length=96, quantile_levels=[0.5])
# output.quantile_predictions: (batch_size, num_quantiles, prediction_length)
```

**提取 Hidden State**：
```python
# output.hidden_states 是预测头前的 q_raw（非 encoder 表示）
output = model(inputs=inputs, prediction_length=96, quantile_levels=[0.5], output_hidden_states=True)
# shape: (B, n_patch * d_patch, num_quantile+1)，即 (B, seq_len, 100)

# 提取真正的 encoder hidden state（TransformerBlock 最后一层输出）需用 hook：
hidden = {}
handle = model.model.blocks[-1].register_forward_hook(lambda m, i, o: hidden.update({"h": o}))
output = model(inputs=inputs, prediction_length=96, quantile_levels=[0.5])
handle.remove()
# hidden["h"] shape: (B, n_patch, d_model)，seq_len=336 时为 (B, 21, 384)
```

### Output Shapes
- PatchTST: `(batch, pred_len, nvars)`
- PatchTST_REPA_Fusion: `(batch, pred_len, nvars)` + `(batch, nvars, output_patch_num, d_extractor)` for contrastive loss

## Key Hyperparameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model` | PatchTST / PatchTST_REPA / PatchTST_REPA_Fusion / Chronos2_head | - |
| `patch_fusion_type` | fusion_MLP (joint) / split_MLP (separable) / none (auto patch_len) | fusion_MLP |
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
| PatchTST_REPA_Fusion (none) | 0 | ~1.2M（patch_len 自动=56） |
| Chronos2_head (use_future_patch=0) | - | ~1.55M (pred_len=96) / ~11.6M (pred_len=720), Flatten_Head |
| Chronos2_head (use_future_patch=1) | - | ~314K (PatchwiseHead, fixed) |

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
│   ├── Chronos2_zeroshot.py   # Chronos2 direct inference test (no training)
│   └── PatchTST_FM_zeroshot.py # PatchTST-FM-R1 zero-shot inference test (no training)
├── exp/
│   └── exp_main.py            # Training & evaluation
├── scripts/                    # Training scripts
└── dataset/                    # Data files (not tracked in git)
```
