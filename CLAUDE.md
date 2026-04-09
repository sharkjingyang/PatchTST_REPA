# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

Extended implementation of PatchTST with feature alignment using contrastive learning and knowledge distillation from Chronos2. Supports feature extractors: **TiViT**, **Mantis**, **Chronos2**. Models include **PatchTST_REPA** (contrastive alignment), **PatchTST_future_align** (joint distillation), **PatchTST_decoder** (FutureQueryDecoder + distillation), and **Chronos2_head** (frozen encoder + head).

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
  --feature_extractor chronos --lambda_alignment 0.1 \
  --alignment_type patch_wise_cos

# PatchTST_future_align (encoder + optional Chronos2 future distillation)
# patch_len 自动推导；--alignment 0 时退化为普通 encoder → head（不加载 Chronos2）

# Distillation mode (with Chronos2 teacher)
python -u run_longExp.py --is_training 1 --model PatchTST_future_align --data custom \
  --root_path ./dataset/ --data_path weather.csv \
  --features M --seq_len 336 --pred_len 96 \
  --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 \
  --dropout 0.3 --fc_dropout 0.3 --head_dropout 0.0 \
  --batch_size 128 --learning_rate 0.0001 \
  --alignment 1 --lambda_t 0.5 --lambda_a 0.5

# Standalone mode (no Chronos2, pure encoder → head) — no chronos package needed
python -u run_longExp.py --is_training 1 --model PatchTST_future_align --data custom \
  --root_path ./dataset/ --data_path weather.csv \
  --features M --seq_len 336 --pred_len 96 \
  --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 \
  --dropout 0.3 --fc_dropout 0.3 --head_dropout 0.0 \
  --batch_size 128 --learning_rate 0.0001 \
  --alignment 0

# PatchTST_decoder (FutureQueryDecoder + optional Chronos2 distillation)
# patch_len=16 (Chronos2 native), stride=16; decoder_layers=cross-attention 层数（默认 1）
# head_type=patch_wise 为推荐配置（FutureQueryDecoder 使 PatchwiseHead 语义成立）
# --alignment 0 时退化为无 Chronos2 版本（纯 FutureQueryDecoder 预测）

# Distillation mode (with Chronos2 teacher)
python -u run_longExp.py --is_training 1 --model PatchTST_decoder --data custom \
  --root_path ./dataset/ --data_path weather.csv \
  --features M --seq_len 336 --pred_len 96 \
  --e_layers 3 --n_heads 8 --d_model 128 --d_ff 256 \
  --dropout 0.3 --fc_dropout 0.3 --head_dropout 0.0 \
  --batch_size 128 --learning_rate 0.0001 \
  --alignment 1 --lambda_t 0.5 --lambda_a 0.5 \
  --decoder_layers 1 --head_type patch_wise

# Standalone mode (no Chronos2, pure FutureQueryDecoder) — no chronos package needed
python -u run_longExp.py --is_training 1 --model PatchTST_decoder --data custom \
  --root_path ./dataset/ --data_path weather.csv \
  --features M --seq_len 336 --pred_len 96 \
  --e_layers 3 --n_heads 8 --d_model 128 --d_ff 256 \
  --dropout 0.3 --fc_dropout 0.3 --head_dropout 0.0 \
  --batch_size 128 --learning_rate 0.0001 \
  --alignment 0 \
  --decoder_layers 1 --head_type patch_wise

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
sh ./scripts/Chronos2_REPA.sh        # PatchTST_REPA + Chronos (patch_wise_cos)
sh ./scripts/Chronos2_FutureAlign.sh  # PatchTST_future_align (joint distillation)
sh ./scripts/Chronos2_Decoder.sh      # PatchTST_decoder (FutureQueryDecoder)
sh ./scripts/Chronos2_featureHead.sh  # Chronos2_head (future + proj_down)
sh ./scripts/Chronos2_zeroshot.sh    # Chronos2 direct inference (no training)
sh ./scripts/PatchTST_FM_zeroshot.sh # PatchTST-FM-R1 zero-shot inference (no training)
```

## Architecture

### Five Models
1. `PatchTST` - Original PatchTST (baseline)
2. `PatchTST_REPA` - PatchTST + Linear Projector + contrastive loss (外部 FM 对齐)
3. `PatchTST_future_align` - Joint distillation: student encoder + optional Chronos2 future teacher (λ=0 → no Chronos2)
4. `PatchTST_decoder` - FutureQueryDecoder + Chronos2 distillation（encoder → cross-attn decoder → PatchwiseHead）
5. `Chronos2_head` - Chronos2 (frozen) + prediction head

### PatchTST_decoder Architecture

**动机**：`PatchTST_future_align` 直接对齐 `z_enc`（past-oriented）与 `z_teacher`（future-oriented），表示空间 gap 大。`PatchTST_decoder` 在 encoder 和 head 之间插入 **FutureQueryDecoder**（cross-attention），由 `output_patch_num` 个可学习 query 从过去 encoder token 中查询未来信息，生成 `z_future`（future-oriented）。

**两种模式**（由 `--alignment` 控制）：
- **Distillation 模式**（`--alignment 1`）：加载 Chronos2 作为 teacher，对齐 `z_future ↔ z_teacher`
- **独立模式**（`--alignment 0`）：**不加载 Chronos2**，纯 FutureQueryDecoder 预测，可作为轻量级 baseline

**数据流**（seq_len=336, pred_len=96: patch_num=21, output_patch_num=6）：

```
Student path (always):
x_past  → TSTiEncoder → z_enc: (bs, nvars, d_model, 21)
                             ↓ reshape: (bs*nvars, 21, d_model)
learned queries (6, d_model) → FutureQueryDecoder (cross-attn) → z_future_flat: (bs*nvars, 6, d_model)
                             ↓ reshape: (bs, nvars, 6, d_model)
                             ↓ permute: (bs, nvars, d_model, 6)
                             ↓ PatchwiseHead → pred_s: (bs, nvars, pred_len)
                             ↓ RevIN denorm → (bs, pred_len, nvars)

Teacher path (only when --alignment 1):
x_future → Chronos2 (frozen) → z_chron: (bs, nvars, 6, 768)
                             ↓ proj_down (768→d_model) → z_teacher: (bs, nvars, 6, d_model)
                             ↓ teacher_head → pred_t → denorm with x_future loc/scale

Alignment loss (distillation mode only): z_future ↔ z_teacher (cosine + MSE)
```

**推理时**：Chronos2 完全不参与，只走 student 路径。

**FutureQueryDecoder 结构**（`layers/PatchTST_Decoder_backbone.py`）：
- `future_queries`: `nn.Parameter(output_patch_num, d_model)` — 可学习未来位置查询
- 每层：`cross_attn(Q=queries, K=z_enc, V=z_enc)` + FFN + LayerNorm × 2
- 参数量（d_model=128, n_heads=8, d_ff=256, output_patch_num=6, n_layers=1）：~133K

**Patch 设置**（Chronos2 native patch_size=16）：
```
output_patch_num = pred_len // 16   (e.g., 96//16=6)
patch_len = 16                      (Chronos2 native)
stride = 16                         (no overlap)
patch_num = seq_len // 16           (e.g., 336//16=21)
→ FutureQueryDecoder maps 21 past patches → 6 future patches
```

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
    ↓ RevIN(x_past, 'norm') → stores x_past loc/scale (affine=False, consistent with PatchTST)
    ↓ Chronos2.embed(future_seq) - frozen (x_future embeddings only, loc/scale discarded)
  Feature: (bs, nvars, 6, 768)  [6 = pred_len/patch_len]
    ↓ [optional] proj_down: Linear(768→d_model) if --proj_down 1
    ↓ Flatten_Head or PatchwiseHead
    ↓ RevIN(x_past, 'denorm') - uses x_past stats (consistent with PatchTST_future_align)
Output: (bs, pred_len, nvars) - denormalized
```

**Note**: `future` 模式推理时也需要传入真实未来序列（无 fallback），适用于 teacher path 验证实验，不用于真实预测。

### Chronos2 Feature Extraction in REPA Models

`PatchTST_REPA` uses `Chronos2Pipeline.embed(batch_x)` to extract **past encoder tokens** as `zs_tilde`. Past tokens are bidirectionally contextualized (T5 encoder)，与 PatchTST 双向 attention 的表示空间更匹配。

```
batch_x: (bs, seq_len, nvars)
  → permute: (bs, nvars, seq_len)
  → chronos.embed(input_perm.cpu())        # pin_memory 需要 CPU tensor
  → embeddings: (bs, nvars, num_past+2, 768)  # +2 为 CLS/SEP special tokens
  → past tokens [:num_past]: (bs, nvars, 21, 768)  ← zs_tilde
```

- `num_past = seq_len // 16`（seq_len=336 时为 21）
- 与 `PatchTST_REPA`（stride=16，patch_num=21）完全匹配，可用 `patch_wise_cos` 对齐

### Normalization in PatchTST_REPA with Chronos2

通过 `--use_chronos_norm 1` 手动启用，用 Chronos2 的 `InstanceNorm(use_arcsinh=True)` 替换 RevIN，适用于任意模型（PatchTST / PatchTST_REPA 均可）：

- Chronos2 内部对输入做 `(x - mean) / std` 后再 `arcsinh`（`use_arcsinh=True`）
- 若 student 用线性 RevIN 而 teacher 用 arcsinh，两者编码的是不同变换后的信号，alignment loss 的梯度是噪声
- 启用 `use_chronos_norm` 后 student 和 teacher 看到相同的归一化输入，表示空间更对齐

```
PatchTST_backbone.forward (use_chronos_norm=True):
  z (bs, nvars, seq_len)
  → reshape (bs*nvars, seq_len)
  → ChronosInstanceNorm(arcsinh=True): normalized + 存储 loc_scale
  → reshape (bs, nvars, seq_len)
  → patching → encoder → head
  → reshape (bs*nvars, pred_len)
  → ChronosInstanceNorm.inverse(loc_scale)  ← sinh + rescale
  → reshape (bs, nvars, pred_len)
```

`use_chronos_norm` 需通过 `--use_chronos_norm 1` 手动指定，适用于任意模型（PatchTST / PatchTST_REPA 均可）。

### Key Components

**`layers/PatchTST_backbone.py`**：
- **`build_linear(hidden_size, z_dim)`**: 单层 Linear，用于 `alignment_mlp`
- **`build_mlp(hidden_size, z_dim, projected_dim=256)`**: 2 层 MLP（Linear→SiLU→Linear），保留备用
- **`alignment_mlp`**: `build_linear(d_model, d_extractor)`，将 encoder 输出投影到特征提取器空间用于对比损失（student 投影到 teacher 维度，per REPA 设计）
- **`use_chronos_norm`**: `--use_chronos_norm 1` 手动启用，用 `ChronosInstanceNorm(arcsinh=True)` 替换 RevIN，任意模型均可使用
- **`PatchwiseHead`**: Lightweight head，每个 patch 独立经过共享 ResidualBlock（d_model→d_ff→output_patch_size）
- **`Flatten_Head`**: 标准全局预测头，Linear(d_model×patch_num → pred_len)
- **`Quantile_Head`**: 分位数预测头

**`layers/PatchTST_Decoder_backbone.py`**：
- **`FutureQueryDecoderLayer`**: cross-attention（Q=learned queries，KV=encoder tokens）+ FFN + LayerNorm
- **`FutureQueryDecoder`**: n_layers 个 DecoderLayer + `output_patch_num` 个可学习 query 参数
- **`PatchTST_Decoder_backbone`**: encoder + FutureQueryDecoder + proj_down + head/teacher_head，提供 `forward_student` / `forward_teacher` 接口

**`layers/PatchTST_FutureAlign_backbone.py`**：
- **`PatchTST_FutureAlign_backbone`**: encoder → 直接 head（无 cross-attention decoder），提供相同接口

### alignment_mlp 规范（PatchTST_REPA）

使用 `build_linear(d_model, d_extractor)`（单层 Linear）：
- 输入 `(bs*nvars*patch_num, d_model)` → 输出 `(bs*nvars*patch_num, d_extractor)`

`d_extractor` 由特征提取器决定（Mantis=256，Chronos2/TiViT=768）。

### Contrastive Loss Types
- `mean_pool`: mean pooling 后做 cosine similarity
- `patch_wise_cos`: per-patch cosine similarity（需要 patch_num 匹配）
- `patch_wise_mse`: per-patch MSE loss（直接监督，信号更强）

### Head Types
- `flatten`: Flatten_Head (standard)，所有模型均支持
- `patch_wise`: PatchwiseHead，支持 `PatchTST_decoder`（推荐）和 `Chronos2_head`
- `quantile`: Quantile_Head for probabilistic forecasting

### PatchwiseHead 适用条件（重要观察）

PatchwiseHead 对每个 output patch 独立预测，其成立前提是：**latent patch i 在语义上与目标序列第 i 段空间对齐**。

| 场景 | PatchwiseHead 是否有效 | 原因 |
|------|----------------------|------|
| `Chronos2_head future` | **有效** | `Chronos.embed(x_future)` 的 patch i 直接编码未来第 i 段，局部独立假设成立 |
| `Chronos2_head predict` | **有效** | model.encode 输出的 future tokens 同样按未来时序排列 |
| `PatchTST_decoder` | **有效（设计目标）** | FutureQueryDecoder query i 通过 cross-attention 专注于未来第 i 段，经 Chronos2 distillation 监督后更对齐 |
| `PatchTST_future_align` | **效果差** | encoder 输出的是过去信息，z_enc patch i 不对应未来第 i 段，Flatten_Head 更优 |

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

**提取 Hidden State**：
```python
hidden = {}
handle = model.model.blocks[-1].register_forward_hook(lambda m, i, o: hidden.update({"h": o}))
output = model(inputs=inputs, prediction_length=96, quantile_levels=[0.5])
handle.remove()
# hidden["h"] shape: (B, n_patch, d_model)，seq_len=336 时为 (B, 21, 384)
```

## Key Hyperparameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model` | PatchTST / PatchTST_REPA / PatchTST_future_align / PatchTST_decoder / Chronos2_head | - |
| `alignment` | Enable alignment/distillation (1/0). REPA: load FM align; FutureAlign/Decoder: load Chronos2 | auto |
| `alignment_type` | mean_pool / patch_wise_cos / patch_wise_mse | mean_pool |
| `lambda_alignment` | Alignment loss weight (REPA models) | 0.5 (推荐 0.1) |
| `feature_extractor` | REPA only: tivit / mantis / chronos | mantis |
| `head_type` | flatten / patch_wise / quantile | flatten |
| `chronos_embed_type` | Chronos2_head: past / predict / future | past |
| `proj_down` | Chronos2_head (future mode): 1=add Linear(768→d_model) before head | 0 |
| `use_chronos_norm` | 用 ChronosInstanceNorm(arcsinh=True) 替换 RevIN，任意模型可用；PatchTST_REPA+chronos 做对比实验时推荐开启 | 0 |
| `lambda_t` | future_align/decoder: 教师路径预测损失权重 (Loss②, Phase 1 warmup) | 0.5 |
| `lambda_t2` | future_align/decoder: 教师路径预测损失权重 (Loss②, Phase 2) | 0.1 |
| `lambda_a` | future_align/decoder: 对齐损失权重 (Loss③) | 0.5 |
| `align_warmup_epochs` | future_align/decoder: Phase 1 teacher-only warmup epoch 数 | 5 |
| `decoder_layers` | PatchTST_decoder: FutureQueryDecoder cross-attention 层数 | 1 |

## Parameter Comparison

### 各模型规模对比 (d_model=128, seq_len=336, pred_len=96, nvars=7)

| Model | 主要模块 | TOTAL trainable |
|-------|---------|----------------|
| PatchTST (patch_len=16, stride=8) | encoder + Flatten_Head | ~921K |
| PatchTST_REPA | encoder + alignment_mlp(98K) + Flatten_Head | ~510K |
| PatchTST_future_align | encoder + proj_down(98K) + head + teacher_head | ~400K |
| PatchTST_decoder | encoder + FutureQueryDecoder(133K) + proj_down(98K) + head + teacher_head | ~525K |

### PatchTST_decoder 参数规模 (d_model=128, n_heads=8, d_ff=256, seq_len=336, pred_len=96)

patch 设置：patch_len=16, stride=16 → patch_num=21 (336//16)，output_patch_num=6 (96//16)

| Module | Params |
|--------|-------:|
| TSTiEncoder (e_layers=3) | ~273K |
| FutureQueryDecoder (decoder_layers=1) | ~133K |
| proj_down (768→128) | 98,304 |
| PatchwiseHead (student) | ~11K |
| PatchwiseHead (teacher) | ~11K |
| RevIN | 14 |
| **TOTAL** | **~525K** |

### PatchTST_future_align 参数规模 (d_model=128, seq_len=336, pred_len=96)

| Module | Params |
|--------|-------:|
| TSTiEncoder (e_layers=3) | ~273K |
| proj_down (768→128) | 98,304 |
| Student Head (Flatten_Head) | ~73K |
| Teacher Head (Flatten_Head) | ~73K |
| RevIN | 14 |
| **TOTAL** | **~518K** |

### Chronos2_head 参数规模

| embed_type | pred_len | 说明 | TOTAL |
|---|---|---|---|
| past | 96 | Flatten_Head(768×21→96) | ~1.55M |
| past | 720 | Flatten_Head(768×21→720) | ~11.6M |
| predict | any | PatchwiseHead，固定 | ~314K |
| future + proj_down | 96 | Linear(768→128) + Flatten_Head(128×6→96) | ~172K |
| future + proj_down | 720 | Linear(768→128) + Flatten_Head(128×45→720) | ~4.25M |

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

## Distillation 模型设计细节

### Normalization 设计（future_align 和 decoder 共用，distillation mode only）

| Path | Denorm 使用的统计量 | 原因 |
|------|-------------------|------|
| Student path | RevIN(x_past) loc/scale | 推理时无法获得 x_future，必须用 past |
| Teacher path (distillation mode) | Chronos2.embed(x_future) 返回的 loc/scale | 与 Chronos2 内部归一化自洽，Loss② 收敛快 |

**关键点**：如果 teacher path 也用 RevIN(x_past) 做 denorm，则 teacher 在有趋势的序列上会遇到 per-sample scale mismatch（future 和 past 的均值不同），导致 Loss② 收敛显著变慢。

### Alignment Loss（Loss③，两个模型相同）

```python
z_n = F.normalize(z, dim=-1)           # z_future 或 z_enc: (bs, nvars, patch_num, d_model)
z_tea_n = F.normalize(z_teacher.detach(), dim=-1)
loss_cosine = -(z_n * z_tea_n).sum(dim=-1).mean()   # range [-1, 1]，越负越对齐
loss_mse_align = F.mse_loss(z, z_teacher.detach())
loss_align = loss_cosine + loss_mse_align
```

对齐在 d_model 空间（通过 `proj_down: Linear(768→d_model)` 把 Chronos teacher 投影到 student 空间）。

### 训练阶段

| Phase | 条件 | 激活的 Loss |
|-------|------|------------|
| Phase 1 (warmup) | epoch < align_warmup_epochs | λ_t × Loss② (teacher MSE only) |
| Phase 2 (align) | epoch ≥ align_warmup_epochs | Loss① + λ_t2 × Loss② + λ_a × Loss③ |

### future_align vs decoder 的核心差异

| 对比项 | PatchTST_future_align | PatchTST_decoder |
|--------|----------------------|-----------------|
| 对齐 student 端 | z_enc（encoder 直接输出，past-oriented） | z_future（FutureQueryDecoder 输出，future-oriented） |
| 对齐 gap | 大（past ↔ future） | 小（future ↔ future） |
| 推荐 head_type | flatten（全局混合更优） | patch_wise（局部对齐语义成立） |
| 额外参数 | 无 | FutureQueryDecoder ~133K |
| `--alignment 0` | 纯 encoder → head（不加载 Chronos2） | 纯 encoder + decoder → head（不加载 Chronos2） |

**与 TimeAlign 的对比**（arXiv:2509.14181）：
TimeAlign 使用 **distribution-aware alignment loss**，显式建模 past/future 表示分布的统计差异。朴素 cosine 对齐梯度中包含大量"不可预测未来"的噪声，distribution-aware 方法在分布层面平滑了这些噪声。`PatchTST_decoder` 通过 cross-attention 使 z_future 主动 query 未来信息，从架构层面减少噪声。

## Directory Structure

```
PatchTST_REPA/
├── run_longExp.py                      # Main entry point
├── layers/
│   ├── PatchTST_backbone.py           # Core backbone (TSTiEncoder, Flatten_Head, PatchwiseHead, alignment_mlp)
│   ├── PatchTST_FutureAlign_backbone.py  # PatchTST_future_align backbone
│   ├── PatchTST_Decoder_backbone.py   # PatchTST_decoder backbone (FutureQueryDecoder)
│   ├── PatchTST_layers.py
│   ├── RevIN.py
│   └── Tivit.py
├── models/
│   ├── PatchTST.py                    # PatchTST / PatchTST_REPA
│   ├── PatchTST_future_align.py       # Joint distillation (encoder → head)
│   ├── PatchTST_decoder.py            # FutureQueryDecoder distillation
│   ├── Chronos2_head.py               # Chronos2 (frozen) + Flatten/Patchwise head
│   ├── Chronos2_zeroshot.py           # Chronos2 direct inference test (no training)
│   └── PatchTST_FM_zeroshot.py        # PatchTST-FM-R1 zero-shot inference test (no training)
├── exp/
│   └── exp_main.py                    # Training & evaluation
├── scripts/                            # Training scripts
└── dataset/                            # Data files (not tracked in git)
```
