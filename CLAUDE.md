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

# PatchTST_REPA (Linear Projector + Contrastive Loss)
# stride=16 使 patch_num=21 与 Chronos2 past tokens 数量一致，可用 patch_wise_cos 对齐
python -u run_longExp.py --is_training 1 --model PatchTST_REPA --data custom \
  --root_path ./dataset/ --data_path weather.csv \
  --features M --seq_len 336 --pred_len 96 \
  --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 \
  --patch_len 16 --stride 16 --batch_size 128 --learning_rate 0.0001 \
  --feature_extractor chronos --lambda_contrastive 0.1 \
  --contrastive_type patch_wise_cos

# PatchTST_REPA_Fusion (Patch Fusion + Contrastive Loss)
# Recommended: split_MLP + patch_wise_cos + lambda=0.1
python -u run_longExp.py --is_training 1 --model PatchTST_REPA_Fusion --data custom \
  --root_path ./dataset/ --data_path weather.csv \
  --features M --seq_len 336 --pred_len 96 \
  --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 \
  --patch_len 16 --stride 8 --batch_size 128 --learning_rate 0.0001 \
  --feature_extractor chronos --lambda_contrastive 0.1 \
  --patch_fusion_type split_MLP --contrastive_type patch_wise_cos

# PatchTST_REPA_Fusion (none mode: 无 fusion MLP，patch_len 自动推导)
# patch_len/stride/padding_patch 参数会被忽略，自动计算：patch_len = seq_len // output_patch_num
# none 模式下对齐目标为 chronos.embed(batch_y)（未来序列）
python -u run_longExp.py --is_training 1 --model PatchTST_REPA_Fusion --data custom \
  --root_path ./dataset/ --data_path weather.csv \
  --features M --seq_len 336 --pred_len 96 \
  --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 \
  --batch_size 128 --learning_rate 0.0001 \
  --feature_extractor chronos --lambda_contrastive 0.1 \
  --patch_fusion_type none --contrastive_type patch_wise_cos

# Chronos2_head (frozen Chronos2 encoder + prediction head)
# --chronos_embed_type past:    past tokens + Flatten_Head (pred_len=96: ~172K with proj_down)
# --chronos_embed_type predict: future tokens + PatchwiseHead (~314K, fixed regardless of pred_len)
# --chronos_embed_type future:  ground-truth future tokens (teacher-forcing, always uses true future)
#   --proj_down 1: add Linear(768→d_model) bottleneck before head (验证 teacher 可行性)
python -u run_longExp.py --is_training 1 --model Chronos2_head --data custom \
  --root_path ./dataset/ --data_path weather.csv \
  --features M --seq_len 336 --pred_len 96 \
  --d_model 128 --patch_len 16 --batch_size 128 --learning_rate 0.0001 \
  --chronos_embed_type future --proj_down 1 --head_type flatten
```

Or use shell scripts:
```bash
sh ./scripts/PatchTST.sh              # Baseline
sh ./scripts/mantis.sh               # PatchTST_REPA + Mantis
sh ./scripts/Chronos2.sh             # PatchTST_REPA + Chronos (patch_wise_cos)
sh ./scripts/Chronos2_REPA_Fusion.sh # PatchTST_REPA_Fusion + Chronos (none mode)
sh ./scripts/Chronos2_featureHead.sh # Chronos2_head (future + proj_down)
sh ./scripts/Chronos2_zeroshot.sh    # Chronos2 direct inference (no training)
sh ./scripts/PatchTST_FM_zeroshot.sh # PatchTST-FM-R1 zero-shot inference (no training)
```

## Architecture

### Four Models
1. `PatchTST` - Original PatchTST (baseline)
2. `PatchTST_REPA` - PatchTST + Linear Projector + contrastive loss (外部 FM 对齐)
3. `PatchTST_REPA_Fusion` - PatchTST + Patch Fusion branch + contrastive loss
4. `Chronos2_head` - Chronos2 (frozen) + prediction head

### Chronos2_head Architecture

Chronos2_head uses a frozen Chronos2 encoder to extract features, then a trainable prediction head. **All outputs are denormalized back to original scale** using InstanceNorm inverse (same as Chronos2's native forward pass).

| embed_type | Features | Head | Trainable Params |
|------------|----------|------|------------------|
| `past`    | Past tokens (21 patches) | Flatten_Head | ~1.55M (pred_len=96) / ~11.6M (pred_len=720), linear in pred_len |
| `predict` | Future tokens only (6 patches) | PatchwiseHead | ~314K (fixed, independent of pred_len) |
| `future`  | Ground-truth future tokens (6 patches, teacher-forcing) | Flatten_Head or PatchwiseHead | depends on head_type and proj_down |

`future` 模式新增 `--proj_down 1` 选项：在 head 前插入 `Linear(768→d_model)` 瓶颈层，用于验证压缩后表示是否仍能保留预测能力（teacher path 验证实验）。

| future + proj_down | pred_len=96 | pred_len=720 |
|---|---|---|
| proj_down (768→128) | 98K | 98K |
| Flatten_Head (128×6→96) | 73K | 4.15M |
| **TOTAL** | **~172K** | **~4.25M** |

Flatten_Head 参数随 pred_len 二次增长（input_dim ∝ pred_len，output_dim = pred_len），pred_len 大时建议用 PatchwiseHead。

**Flow (embed_type="past")**:
```
Input x: (bs, seq_len, nvars)
  ↓ Chronos2.embed(x) - frozen
Feature: (bs, nvars, 21, 768)  [21 = seq_len/patch_len]
  ↓ permute(0,1,3,2): (bs, nvars, 768, 21)
  ↓ Flatten_Head → flatten: (bs*nvars, 768*21) → linear: (bs*nvars, pred_len)
  ↓ InstanceNorm.inverse (loc, scale from embed)
Output: (bs, pred_len, nvars) - denormalized
```

**Flow (embed_type="predict")** - Like Chronos2's native prediction:
```
Input x: (bs, seq_len, nvars)
  ↓ Chronos2.model.encode(x, num_output_patches) - frozen
Feature: (bs, nvars, 28, 768)  [21 + 1 (REG) + 6 (future)]
  ↓ extract ONLY future tokens: hidden_states[:, -6:, :]
Feature: (bs, nvars, 6, 768)
  ↓ PatchwiseHead → ResidualBlock per patch: d_model -> d_ff -> output_patch_size
  ↓ InstanceNorm.inverse (loc, scale from encode)
Output: (bs, pred_len, nvars) - denormalized
```

**Flow (embed_type="future")** - Teacher-forcing with ground-truth future:
```
Training & Inference (always uses true future):
  future_seq: (bs, pred_len, nvars) [ground truth]
    ↓ Chronos2.embed(future_seq) - frozen
  Feature: (bs, nvars, 6, 768)  [6 = pred_len/patch_len]
    ↓ [optional] proj_down: Linear(768→d_model) if --proj_down 1
    ↓ Flatten_Head or PatchwiseHead
    ↓ InstanceNorm.inverse (loc, scale from future embed)
Output: (bs, pred_len, nvars) - denormalized
```

**Note**: `future` 模式推理时也需要传入真实未来序列（无 fallback），适用于 teacher path 验证实验，不用于真实预测。

### Chronos2 Feature Extraction in REPA Models

Both `PatchTST_REPA` and `PatchTST_REPA_Fusion` use `Chronos2Pipeline.embed(batch_x)` to extract **past encoder tokens** as `zs_tilde`. Past tokens are bidirectionally contextualized (T5 encoder)，与 PatchTST 双向 attention 的表示空间更匹配。

```
batch_x: (bs, seq_len, nvars)
  → permute: (bs, nvars, seq_len)
  → chronos.embed(input_perm.cpu())        # pin_memory 需要 CPU tensor
  → embeddings: (bs, nvars, num_past+2, 768)  # +2 为 CLS/SEP special tokens
  → past tokens [:num_past]: (bs, nvars, 21, 768)  ← zs_tilde
```

- `num_past = seq_len // 16`（seq_len=336 时为 21）
- 与 `PatchTST_REPA`（stride=16，patch_num=21）完全匹配，可用 `patch_wise_cos` 对齐
- `PatchTST_REPA_Fusion` none 模式：对齐目标改为 `chronos.embed(batch_y)`（未来序列）

### Key Components
- **`build_linear(hidden_size, z_dim)`**: 单层 Linear，用于 `alignment_mlp`（取代原来的 2 层 MLP，强迫 encoder 自己做对齐）
- **`build_mlp(hidden_size, z_dim, projected_dim=256)`**: 2 层 MLP（Linear→SiLU→Linear），保留备用
- **Patch_Fusion_MLP**: 联合投影 `d_model*patch_num → d_model*output_patch_num`（`fusion_MLP` 模式）
- **`nn.Linear(patch_num, output_patch_num)`**: `split_MLP` 模式直接内联，仅投影时间维度，参数极少（~258）
- **`none` 模式**: 无 fusion MLP，`patch_len` 自动推导为 `seq_len // output_patch_num`，patch_num 天然等于 output_patch_num
- **TransformerDecoder**: patch fusion 后对 `(bs*nvars, output_patch_num, d_model)` 做自注意力，`d_ff` 与 backbone 一致
- **alignment_mlp**: `build_linear(d_model, d_extractor)`，将 encoder 输出投影到特征提取器空间用于对比损失
- **PatchwiseHead**: Lightweight head using shared ResidualBlock per patch

### alignment_mlp 规范
两种模式均使用 `build_linear(d_model, d_extractor)`（单层 Linear）：
- `PatchTST_REPA`（无 fusion）：输入 `(bs*nvars*patch_num, d_model)` → 输出 `(bs*nvars*patch_num, d_extractor)`
- `PatchTST_REPA_Fusion`：输入 `(bs*nvars*output_patch_num, d_model)` → 输出 `(bs*nvars*output_patch_num, d_extractor)`

`d_extractor` 由特征提取器决定（Mantis=256，Chronos2/TiViT=768）。单层 Linear 相比 2 层 MLP 强迫 encoder 本身做更多对齐工作，避免 projector 吸收所有对齐梯度。

### Contrastive Loss Types
- `mean_pool`: mean pooling 后做 cosine similarity
- `patch_wise_cos`: per-patch cosine similarity（需要 patch_num 匹配）
- `patch_wise_mse`: per-patch MSE loss（直接监督，信号更强）

### Head Types
- `flatten`: Flatten_Head (standard)，所有模型均支持
- `patch_wise`: PatchwiseHead，**仅 `PatchTST_REPA_Fusion` 和 `Chronos2_head` 支持**
- `quantile`: Quantile_Head for probabilistic forecasting

### PatchwiseHead 适用条件（重要观察）

PatchwiseHead 对每个 output patch 独立预测，其成立前提是：**latent patch i 在语义上与目标序列第 i 段空间对齐**。

| 场景 | PatchwiseHead 是否有效 | 原因 |
|------|----------------------|------|
| `Chronos2_head future` | **有效** | `Chronos.embed(x_future)` 的 patch i 直接编码未来第 i 段，局部独立假设成立 |
| `Chronos2_head predict` | **有效** | model.encode 输出的 future tokens 同样按未来时序排列 |
| `PatchTST_REPA_Fusion` | **效果差** | encoder 输出的是过去信息，TransformerDecoder 仅做自注意力（无 cross-attention），output patch i 不保证对应未来第 i 段 |

**结论**：`Chronos2_head` 场景本质是从已知未来表征中 decode（easy），`PatchTST_REPA_Fusion` 是从过去预测未来（hard）。后者 Flatten_Head 更优，因为它将所有 patch 拼接后全局预测，允许 head 自己学到跨 patch 的混合来弥补局部对齐不足。

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

**提取 Hidden State**：
```python
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
| `lambda_contrastive` | Contrastive loss weight (REPA models) | 0.5 (推荐 0.1) |
| `contrastive_type` | mean_pool / patch_wise_cos / patch_wise_mse | mean_pool |
| `chronos_embed_type` | Chronos2_head: past / predict / future | past |
| `proj_down` | Chronos2_head (future mode): 1=add Linear(768→d_model) before head | 0 |

## Parameter Comparison

### PatchTST_REPA_Fusion (split_MLP, d_model=128, seq_len=336, pred_len=96, nvars=21)

| Module | Params |
|--------|-------:|
| backbone (encoder) | 272,514 |
| patch_fusion_mlp (`nn.Linear(42,6)`) | 258 |
| transformer_decoder (d_ff=256) | 132,480 |
| alignment_mlp (`build_linear(128→768)`) | 98,304 |
| head (Flatten_Head) | 73,824 |
| revin_layer | 42 |
| **TOTAL** | **~577K** |

### PatchTST_REPA_Fusion (none, d_model=128, seq_len=336, pred_len=96, nvars=7)

patch_len 自动推导 = 336//6 = 56，patch_num = output_patch_num = 6，对齐目标为 chronos.embed(batch_y)（未来序列 6 tokens）。

| Module | Params |
|--------|-------:|
| backbone (encoder_depth layers, patch_len=56) | ~273K |
| patch_fusion_mlp | 0 |
| transformer_decoder (d_ff=256) | 132,480 |
| alignment_mlp (`build_linear(128→768)`) | 98,304 |
| head (Flatten_Head) | 73,824 |
| revin_layer | 14 |
| **TOTAL** | **~577K** |

### 各模型规模对比 (d_model=128, seq_len=336, pred_len=96)

| Model | alignment_mlp | TOTAL |
|-------|--------------|-------|
| PatchTST | - | ~921K |
| PatchTST_REPA | 98K (Linear 128→768) | ~510K |
| PatchTST_REPA_Fusion (fusion_MLP) | 98K | ~1.1M |
| PatchTST_REPA_Fusion (split_MLP) | 98K | ~577K |
| PatchTST_REPA_Fusion (none) | 98K | ~577K |

### PatchTST 参数规模 (seq_len=336, pred_len=96, e_layers=3, patch_len=16)

| enc_in | d_model | d_ff | n_heads | stride | patch_num | backbone | head | TOTAL |
|--------|---------|------|---------|--------|-----------|----------|------|-------|
| 21 | 128 | 256 | 16 | 8 | 42 | 404,995 | 516,192 | ~921K |
| 21 | 16 | 64 | 4 | 8 | 42 | 10,787 | 64,608 | ~75K |
| 21 | 16 | 128 | 4 | 8 | 42 | 17,123 | 64,608 | ~82K |
| 7 | 16 | 128 | 4 | 16 | 22 | 16,803 | 33,888 | ~51K |

head = `Linear(d_model × patch_num, pred_len)`，stride 越大 patch_num 越小，head 参数越少。

### Chronos2_head 参数规模

| embed_type | pred_len | 说明 | TOTAL |
|---|---|---|---|
| past | 96 | Flatten_Head(768×21→96) | ~1.55M |
| past | 720 | Flatten_Head(768×21→720) | ~11.6M |
| predict | any | PatchwiseHead，固定 | ~314K |
| future | 96 | Flatten_Head(768×6→96) | ~4.7M |
| future + proj_down | 96 | Linear(768→128) + Flatten_Head(128×6→96) | ~172K |
| future + proj_down | 720 | Linear(768→128) + Flatten_Head(128×45→720) | ~4.25M |

`alignment_mlp` 从 2 层 MLP（128→256→768，230K）简化为单层 Linear（128→768，98K），强迫 encoder 自己做对齐。

## Latent Space Quality Evaluation

**目标**：判断 latent space 本身的好坏（不依赖外部参考），以及对齐后是否有改善。

### 指标体系

| 指标 | 衡量什么 | 是否需要外部参考 |
|------|---------|----------------|
| Temporal Locality (TL) | patch 间表示的时序连续性 | 否 |
| CKA(zs, zs_tilde) | PatchTST 与 Chronos 的对齐程度 | 是（Chronos） |

### 1. Temporal Locality（patch-level）

```python
# latent: (B, C, P, D)
diff = latent[:, :, 1:, :] - latent[:, :, :-1, :]          # (B, C, P-1, D)
TL = (diff.norm(dim=-1) / (latent[:, :, :-1, :].norm(dim=-1) + 1e-8)).mean().item()
```

### 2. CKA（对齐程度）

```python
def cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    hsic_xy = (X @ Y.T).pow(2).sum()
    hsic_xx = (X @ X.T).pow(2).sum()
    hsic_yy = (Y @ Y.T).pow(2).sum()
    return (hsic_xy / (hsic_xx * hsic_yy).sqrt()).item()
```

### 参考：LatentTSF 的发现

LatentTSF（ICML，arXiv:2602.00297）提出了 **Latent Chaos** 概念：MSE 训练的模型预测精度高但 latent 时序混乱。
- 原始观测空间 TL ≈ 12.94（参考基线）
- 标准模型 latent TL ≈ 94.03（混乱 7×）
- 损失函数：`ℒ = α·ℒ_Pred + β·ℒ_Align`，α=10，β=15

## Directory Structure

```
PatchTST_REPA/
├── run_longExp.py              # Main entry point
├── layers/
│   ├── PatchTST_backbone.py   # Core model (build_linear, build_mlp, Patch_Fusion_MLP, TransformerDecoder, Flatten_Head, PatchwiseHead)
│   ├── PatchTST_layers.py
│   ├── RevIN.py
│   └── Tivit.py
├── models/
│   ├── PatchTST.py            # PatchTST / PatchTST_REPA / PatchTST_REPA_Fusion
│   ├── Chronos2_head.py       # Chronos2 (frozen) + Flatten/Patchwise head; supports proj_down
│   ├── Chronos2_zeroshot.py   # Chronos2 direct inference test (no training)
│   └── PatchTST_FM_zeroshot.py # PatchTST-FM-R1 zero-shot inference test (no training)
├── exp/
│   └── exp_main.py            # Training & evaluation
├── scripts/                    # Training scripts
└── dataset/                    # Data files (not tracked in git)
```
