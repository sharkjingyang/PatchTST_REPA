# PatchTST + Feature Alignment (TiViT / Mantis / Chronos)

This is an extended implementation of PatchTST: [A Time Series is Worth 64 Words: Long-term Forecasting with Transformers](https://arxiv.org/abs/2211.14730)

This project extends PatchTST with feature alignment using contrastive learning and Patch Fusion architecture.

## Key Features

- **Patch-based Time Series Modeling**: Segmentation of time series into subseries-level patches
- **Channel-independence**: Each channel contains a univariate time series with shared embedding and Transformer weights
- **Feature Alignment**: MLP projector aligns PatchTST encoder features with TiViT, Mantis, or Chronos2-extracted features
- **Patch Fusion**: Alternative architecture using `split_MLP` or `fusion_MLP` to compress patch dimension
- **Three Feature Extractors**: TiViT (768-dim), Mantis (256-dim), Chronos2 (768-dim)
- **Backbone-only mode**: Use REPA/Fusion backbone without feature extractor via `--contrastive 0`
- **Quantile Prediction Head**: Supports probabilistic forecasting with `--head_type quantile`
- **Lightweight Heads**: PatchwiseHead with shared ResidualBlock (~11K params)

## Models

| Model | Description |
|-------|-------------|
| `PatchTST` | Original PatchTST (baseline) |
| `PatchTST_REPA` | PatchTST + MLP Projector + contrastive loss |
| `PatchTST_REPA_Fusion` | PatchTST + Patch Fusion branch + contrastive loss |
| `Chronos2_head` | Frozen Chronos2 encoder + trainable prediction head |

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Download Data
Place CSV files in `./dataset/` (ETTm1, ETTm2, ETTh1, ETTh2, electricity, traffic, weather, etc.)

### Training Commands

```bash
# Baseline PatchTST
python -u run_longExp.py --is_training 1 --model PatchTST --data custom \
  --root_path ./dataset/ --data_path weather.csv \
  --features M --seq_len 336 --pred_len 96 \
  --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 \
  --patch_len 16 --stride 8 --batch_size 128 --learning_rate 0.0001

# PatchTST_REPA backbone only (no feature extractor, no contrastive loss)
python -u run_longExp.py --is_training 1 --model PatchTST_REPA --data custom \
  --root_path ./dataset/ --data_path weather.csv \
  --features M --seq_len 336 --pred_len 96 \
  --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 \
  --patch_len 16 --stride 8 --batch_size 128 --learning_rate 0.0001 \
  --contrastive 0

# PatchTST_REPA + Chronos (MLP Projector + contrastive loss)
python -u run_longExp.py --is_training 1 --model PatchTST_REPA --data custom \
  --root_path ./dataset/ --data_path weather.csv \
  --features M --seq_len 336 --pred_len 96 \
  --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 \
  --patch_len 16 --stride 8 --batch_size 128 --learning_rate 0.0001 \
  --feature_extractor chronos --lambda_contrastive 0.1 \
  --contrastive_type mean_pool

# PatchTST_REPA_Fusion + Patch Fusion (split_MLP, no contrastive)
python -u run_longExp.py --is_training 1 --model PatchTST_REPA_Fusion --data custom \
  --root_path ./dataset/ --data_path weather.csv \
  --features M --seq_len 336 --pred_len 96 \
  --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 \
  --patch_len 16 --stride 8 --batch_size 128 --learning_rate 0.0001 \
  --patch_fusion_type split_MLP --contrastive 0

# PatchTST_REPA_Fusion + Patch Fusion + Chronos contrastive (recommended)
python -u run_longExp.py --is_training 1 --model PatchTST_REPA_Fusion --data custom \
  --root_path ./dataset/ --data_path weather.csv \
  --features M --seq_len 336 --pred_len 96 \
  --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 \
  --patch_len 16 --stride 8 --batch_size 128 --learning_rate 0.0001 \
  --feature_extractor chronos --lambda_contrastive 0.1 \
  --patch_fusion_type split_MLP --contrastive_type patch_wise

# Chronos2_head (frozen encoder + prediction head)
python -u run_longExp.py --is_training 1 --model Chronos2_head --data custom \
  --root_path ./dataset/ --data_path weather.csv \
  --features M --seq_len 336 --pred_len 96 \
  --patch_len 16 --batch_size 128 --learning_rate 0.0001 \
  --use_future_patch 0
```

Or use shell scripts:
```bash
sh ./scripts/PatchTST.sh              # Baseline
sh ./scripts/mantis.sh                # PatchTST_REPA + Mantis
sh ./scripts/Chronos2.sh              # PatchTST_REPA + Chronos (patch_wise)
sh ./scripts/Chronos2_featureHead.sh  # Chronos2_head (frozen encoder + head)
sh ./scripts/Chronos2_zeroshot.sh     # Chronos2 direct inference (no training)
```

## Device Selection

Use `--device` to specify the compute device (default: `cuda:0`):

```bash
--device cuda:0   # GPU 0 (default)
--device cuda:1   # GPU 1
--device cpu      # CPU
```

## Patch Fusion

Patch Fusion compresses encoder features from `patch_num` patches down to `output_patch_num` patches via a lightweight MLP, followed by a Transformer decoder.

### Two Projection Types

| Type | Description | Parameters |
|------|-------------|------------|
| `fusion_MLP` | Joint projection: `d_model × patch_num → d_model × output_patch_num` | ~670K |
| `split_MLP` | Projects patch dimension only: `nn.Linear(patch_num, output_patch_num)` | ~258 |

`split_MLP` reduces Patch Fusion parameters by **>99%**.

### Parameter Comparison (d_model=128, seq_len=336, pred_len=96, nvars=21)

| Model | patch_fusion_mlp | Total |
|-------|-----------------|-------|
| PatchTST | - | ~276K |
| PatchTST_REPA | - | ~1.1M |
| PatchTST_REPA_Fusion (fusion_MLP) | ~670K | ~1.8M |
| PatchTST_REPA_Fusion (split_MLP) | 258 | ~1.2M |
| Chronos2_head (use_future_patch=0) | - | ~1.5M |
| Chronos2_head (use_future_patch=1) | - | ~450K |

## Key Hyperparameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model` | PatchTST / PatchTST_REPA / PatchTST_REPA_Fusion / Chronos2_head | - |
| `device` | Compute device | `cuda:0` |
| `contrastive` | Enable contrastive loss (1/0); 0 skips feature extractor entirely | auto |
| `patch_fusion_type` | fusion_MLP or split_MLP | fusion_MLP |
| `feature_extractor` | tivit / mantis / chronos | mantis |
| `head_type` | flatten / patch_wise / quantile | flatten |
| `lambda_contrastive` | Contrastive loss weight | 0.5 (recommended 0.1) |
| `contrastive_type` | mean_pool / patch_wise | mean_pool |
| `use_future_patch` | Chronos2_head: 0=past tokens+Flatten, 1=future tokens+Patchwise | 0 |

## Directory Structure

```
PatchTST_REPA/
├── run_longExp.py              # Main entry point
├── layers/
│   ├── PatchTST_backbone.py   # Core model (build_mlp, Patch_Fusion_MLP, TransformerDecoder, heads)
│   ├── PatchTST_layers.py
│   ├── RevIN.py
│   └── Tivit.py
├── models/
│   ├── PatchTST.py            # PatchTST / PatchTST_REPA / PatchTST_REPA_Fusion
│   ├── Chronos2_head.py       # Chronos2 (frozen) + Flatten/Patchwise head
│   └── Chronos2_zeroshot.py   # Chronos2 direct inference (no training)
├── exp/
│   └── exp_main.py            # Training & evaluation
├── scripts/                    # Training scripts
└── dataset/                    # Data files (not tracked in git)
```

## Acknowledgements

- [PatchTST](https://github.com/yuqinie98/PatchTST)
- [Informer](https://github.com/zhouhaoyi/Informer2020)
- [Autoformer](https://github.com/thuml/Autoformer)
- [Chronos](https://github.com/amazon-science/chronos-forecasting)
