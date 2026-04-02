# PatchTST + Feature Alignment

Extended implementation of [PatchTST](https://arxiv.org/abs/2211.14730) with feature alignment, patch fusion, and joint distillation training.

## Models

| Model | Description |
|-------|-------------|
| `PatchTST` | Original PatchTST (baseline) |
| `PatchTST_REPA` | PatchTST + Linear Projector + contrastive loss (past→FM alignment) |
| `PatchTST_REPA_Fusion` | PatchTST + Patch Fusion branch + contrastive loss |
| `PatchTST_future_align` | Joint distillation: Encoder (student) + Chronos2 future embeddings (teacher) |
| `Chronos2_head` | Frozen Chronos2 encoder + trainable prediction head |

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Data
Place CSV files in `./dataset/` (ETTh1, ETTh2, ETTm1, ETTm2, weather, electricity, traffic, etc.)

### Training Commands

```bash
# Baseline PatchTST
python -u run_longExp.py --is_training 1 --model PatchTST --data custom \
  --root_path ./dataset/ --data_path weather.csv \
  --features M --seq_len 336 --pred_len 96 \
  --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 \
  --patch_len 16 --stride 8 --batch_size 128 --learning_rate 0.0001

# PatchTST_REPA + Chronos2 (patch-wise cosine alignment)
# stride=16 makes patch_num=21, matching Chronos2 past token count
python -u run_longExp.py --is_training 1 --model PatchTST_REPA --data custom \
  --root_path ./dataset/ --data_path weather.csv \
  --features M --seq_len 336 --pred_len 96 \
  --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 \
  --patch_len 16 --stride 16 --batch_size 128 --learning_rate 0.0001 \
  --feature_extractor chronos --lambda_contrastive 0.1 \
  --contrastive_type patch_wise_cos

# PatchTST_REPA_Fusion + Chronos2 (recommended: split_MLP + patch_wise_cos)
python -u run_longExp.py --is_training 1 --model PatchTST_REPA_Fusion --data custom \
  --root_path ./dataset/ --data_path weather.csv \
  --features M --seq_len 336 --pred_len 96 \
  --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 \
  --patch_len 16 --stride 8 --batch_size 128 --learning_rate 0.0001 \
  --feature_extractor chronos --lambda_contrastive 0.1 \
  --patch_fusion_type split_MLP --contrastive_type patch_wise_cos

# PatchTST_future_align (joint distillation, patch_len auto-derived)
python -u run_longExp.py --is_training 1 --model PatchTST_future_align --data custom \
  --root_path ./dataset/ --data_path weather.csv \
  --features M --seq_len 336 --pred_len 96 \
  --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 \
  --batch_size 128 --learning_rate 0.0001 \
  --lambda_t 0.5 --lambda_a 0.1

# Chronos2_head (frozen encoder + trainable head, future teacher-forcing)
python -u run_longExp.py --is_training 1 --model Chronos2_head --data custom \
  --root_path ./dataset/ --data_path weather.csv \
  --features M --seq_len 336 --pred_len 96 \
  --d_model 128 --patch_len 16 --batch_size 128 --learning_rate 0.0001 \
  --chronos_embed_type future --proj_down 1 --head_type flatten
```

### Shell Scripts
```bash
sh ./scripts/PatchTST.sh               # Baseline
sh ./scripts/mantis.sh                 # PatchTST_REPA + Mantis
sh ./scripts/Chronos2.sh               # PatchTST_REPA + Chronos2 (patch_wise_cos)
sh ./scripts/Chronos2_REPA_Fusion.sh   # PatchTST_REPA_Fusion + Chronos2 (none mode)
sh ./scripts/FutureAlign.sh            # PatchTST_future_align (joint distillation)
sh ./scripts/Chronos2_featureHead.sh   # Chronos2_head (future + proj_down)
sh ./scripts/Chronos2_zeroshot.sh      # Chronos2 direct inference (no training)
sh ./scripts/PatchTST_FM_zeroshot.sh   # PatchTST-FM-R1 zero-shot inference
```

## Architecture

### PatchTST_future_align (Joint Distillation)

Training runs two paths simultaneously sharing the same Flatten_Head and RevIN:

```
Path A (Teacher):
  x_future → Chronos2 (frozen) → proj_down (768→d_model) → z_teacher → Head → pred_teacher

Path B (Student):
  x_past → Encoder (trainable) → z_enc → Head → pred_student

Loss = MSE(pred_student, y)                       # Loss① → Encoder + Head
     + λ_t · MSE(pred_teacher, y)                 # Loss② → proj_down + Head
     + λ_a · MSE(z_enc, z_teacher.detach())       # Loss③ → Encoder only

Inference: only Path B — Chronos2 not needed.
```

`patch_len` is auto-derived: `seq_len // (pred_len // 16)`, ensuring `patch_num == output_patch_num` for patch-wise alignment.

### PatchTST_REPA_Fusion

Three `patch_fusion_type` modes:

| Mode | Description | Params (fusion MLP) |
|------|-------------|---------------------|
| `fusion_MLP` | Joint projection `d_model×patch_num → d_model×output_patch_num` | ~670K |
| `split_MLP` | Projects patch dimension only `nn.Linear(patch_num, output_patch_num)` | 258 |
| `none` | `patch_len` auto-derived, no fusion MLP needed | 0 |

### Chronos2_head

| embed_type | Features | Head | Trainable Params |
|------------|----------|------|-----------------|
| `past` | Past tokens (21 patches × 768) | Flatten_Head | ~1.55M (pred=96) |
| `predict` | Future tokens (6 patches × 768) | PatchwiseHead | ~314K (fixed) |
| `future` | Ground-truth future (teacher-forcing) | Flatten_Head | ~4.7M |
| `future` + `proj_down` | Compressed via Linear(768→d_model) | Flatten_Head | ~172K (pred=96) |

## Key Hyperparameters

| Parameter | Models | Description | Default |
|-----------|--------|-------------|---------|
| `feature_extractor` | REPA, Fusion | `tivit` / `mantis` / `chronos` | `mantis` |
| `contrastive_type` | REPA, Fusion | `mean_pool` / `patch_wise_cos` / `patch_wise_mse` | `mean_pool` |
| `lambda_contrastive` | REPA, Fusion | Contrastive loss weight | 0.5 (recommended 0.1) |
| `patch_fusion_type` | Fusion | `fusion_MLP` / `split_MLP` / `none` | `fusion_MLP` |
| `contrastive` | REPA, Fusion | Enable contrastive loss (1/0/None=auto) | auto |
| `head_type` | All | `flatten` / `patch_wise` / `quantile` | `flatten` |
| `chronos_embed_type` | Chronos2_head | `past` / `predict` / `future` | `past` |
| `proj_down` | Chronos2_head | Add Linear(768→d_model) before head (future mode) | 0 |
| `lambda_t` | future_align | Teacher path loss weight (Loss②) | 0.5 |
| `lambda_a` | future_align | Alignment loss weight (Loss③) | 0.1 |

## PatchwiseHead Note

PatchwiseHead predicts each output patch independently and **only works well when latent patch i semantically corresponds to future segment i**. This holds for `Chronos2_head` (future tokens are positionally aligned) but not for `PatchTST_REPA_Fusion` (encoder sees only past, no guaranteed local alignment). Use `Flatten_Head` for REPA/Fusion models.

## Parameter Comparison (d_model=128, seq_len=336, pred_len=96)

| Model | Trainable Params |
|-------|----------------:|
| PatchTST | ~921K |
| PatchTST_REPA | ~510K |
| PatchTST_REPA_Fusion (split_MLP) | ~577K |
| PatchTST_future_align | ~445K |
| Chronos2_head (future + proj_down) | ~172K |

## Directory Structure

```
PatchTST_REPA/
├── run_longExp.py
├── layers/
│   ├── PatchTST_backbone.py          # Core backbone, heads, build_linear
│   ├── PatchTST_FutureAlign_backbone.py  # Joint distillation backbone
│   ├── PatchTST_layers.py
│   ├── RevIN.py
│   └── Tivit.py
├── models/
│   ├── PatchTST.py                   # PatchTST / REPA / REPA_Fusion
│   ├── PatchTST_future_align.py      # Joint distillation model
│   ├── Chronos2_head.py              # Frozen Chronos2 + head
│   ├── Chronos2_zeroshot.py          # Chronos2 zero-shot inference
│   └── PatchTST_FM_zeroshot.py       # PatchTST-FM-R1 zero-shot inference
├── exp/
│   └── exp_main.py
├── scripts/
│   ├── PatchTST.sh
│   ├── Chronos2.sh
│   ├── Chronos2_REPA_Fusion.sh
│   ├── FutureAlign.sh
│   ├── Chronos2_featureHead.sh
│   ├── Chronos2_zeroshot.sh
│   ├── PatchTST_FM_zeroshot.sh
│   └── mantis.sh
├── diagnose_results/
│   ├── test_Chronos2_head.py
│   ├── test_past_token_alignment.py
│   └── test_use_projector.py
└── dataset/                          # Data files (not tracked in git)
```

## Acknowledgements

- [PatchTST](https://github.com/yuqinie98/PatchTST)
- [Chronos](https://github.com/amazon-science/chronos-forecasting)
- [Informer](https://github.com/zhouhaoyi/Informer2020)
- [Autoformer](https://github.com/thuml/Autoformer)
