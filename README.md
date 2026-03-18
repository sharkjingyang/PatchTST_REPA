# PatchTST + Feature Alignment (TiViT / Mantis / Chronos)

This is an extended implementation of PatchTST: [A Time Series is Worth 64 Words: Long-term Forecasting with Transformers](https://arxiv.org/abs/2211.14730)

This project extends PatchTST with feature alignment using contrastive learning and Patch Fusion architecture.

## Key Features

- **Patch-based Time Series Modeling**: Segmentation of time series into subseries-level patches
- **Channel-independence**: Each channel contains a univariate time series with shared embedding and Transformer weights
- **Feature Alignment**: MLP projector aligns PatchTST encoder features with TiViT, Mantis, or Chronos2-extracted features
- **Patch Fusion**: Alternative architecture using Patch_Fusion_MLP or Patch_Split_MLP for reduced parameters
- **Three Feature Extractors**: TiViT (768-dim), Mantis (256-dim), Chronos2 (768-dim)
- **Quantile Prediction Head**: Supports probabilistic forecasting with `--head_type quantile`
- **Lightweight Heads**: PatchwiseHead with shared ResidualBlock (~11K params)

## Models

| Model | Description |
|-------|-------------|
| `PatchTST` | Original PatchTST (baseline) |
| `PatchTST_REPA` | PatchTST + MLP Projector + contrastive loss |
| `PatchTST_REPA_Fusion` | PatchTST + Patch Fusion branch + contrastive loss |

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
  --e_layers 3 --n_heads 16 --d_model 128

# PatchTST_REPA + Chronos (MLP Projector)
python -u run_longExp.py --is_training 1 --model PatchTST_REPA --data custom \
  --root_path ./dataset/ --data_path weather.csv \
  --features M --seq_len 336 --pred_len 96 \
  --e_layers 3 --n_heads 16 --d_model 128 \
  --feature_extractor chronos --lambda_contrastive 0.5

# PatchTST_REPA_Fusion + Patch Fusion (split_MLP for fewer params)
python -u run_longExp.py --is_training 1 --model PatchTST_REPA_Fusion --data custom \
  --root_path ./dataset/ --data_path weather.csv \
  --features M --seq_len 336 --pred_len 96 \
  --e_layers 3 --n_heads 16 --d_model 128 \
  --feature_extractor chronos --lambda_contrastive 0.5 \
  --patch_fusion_type split_MLP
```

Or use shell scripts:
```bash
sh ./scripts/PatchTST.sh     # Baseline
sh ./scripts/mantis.sh        # PatchTST_REPA + Mantis
sh ./scripts/Chronos2.sh     # PatchTST_REPA + Chronos
```

## Patch Fusion

Patch Fusion is an alternative architecture that uses a separate MLP to project encoder features to output patch space, combined with a lightweight Transformer decoder.

### Two Projection Types

| Type | Description | Parameters |
|------|-------------|------------|
| `fusion_MLP` | Joint projection (d_model × patch_num → d_channel × output_patch_num) | ~504K |
| `split_MLP` | Separable projection (d_model → d_channel, then patch_num → output_patch_num) | ~2.4K |

Using `split_MLP` reduces Patch Fusion parameters by **~99.5%**.

### Parameter Comparison (d_model=16, seq_len=336, pred_len=96)

| Model | Total (excl. Chronos) |
|-------|----------------------|
| PatchTST | ~24K |
| PatchTST_REPA | ~326K |
| PatchTST_REPA_Fusion (fusion_MLP) | ~735K |
| PatchTST_REPA_Fusion (split_MLP) | ~233K |

## Key Hyperparameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model` | PatchTST / PatchTST_REPA / PatchTST_REPA_Fusion | - |
| `patch_fusion_type` | fusion_MLP or split_MLP | fusion_MLP |
| `feature_extractor` | tivit / mantis / chronos | mantis |
| `head_type` | flatten / patchwise / quantile | flatten |
| `lambda_contrastive` | Contrastive loss weight | 0.5 |

## Directory Structure

```
PatchTST_REPA/
├── run_longExp.py              # Main entry point
├── layers/
│   ├── PatchTST_backbone.py    # Core model (Patch_Fusion_MLP, Patch_Split_MLP)
│   ├── PatchTST_layers.py
│   ├── RevIN.py
│   └── Tivit.py
├── models/
│   └── PatchTST.py             # Model wrapper
├── exp/
│   └── exp_main.py             # Training & evaluation
├── scripts/                     # Training scripts
└── dataset/                     # Data files
```

## Acknowledgements

- [PatchTST](https://github.com/yuqinie98/PatchTST)
- [Informer](https://github.com/zhouhaoyi/Informer2020)
- [Autoformer](https://github.com/thuml/Autoformer)
- [Chronos](https://github.com/amazon-science/chronos-forecasting)
