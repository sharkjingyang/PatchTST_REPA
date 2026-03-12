# PatchTST + Feature Alignment (TiViT / Mantis)

### This is an extended implementation of PatchTST: [A Time Series is Worth 64 Words: Long-term Forecasting with Transformers](https://arxiv.org/abs/2211.14730)

This project extends PatchTST with feature alignment using contrastive learning. Supports two feature extractors: **TiViT** (Time series to ViT) and **Mantis** (from mantis-tsfm). The MLP projector maps PatchTST's intermediate encoder features to align with extracted features from the target sequence.

## Key Features

:star2: **Patch-based Time Series Modeling**: Segmentation of time series into subseries-level patches served as input tokens to Transformer.

:star2: **Channel-independence**: Each channel contains a single univariate time series that shares the same embedding and Transformer weights.

:star2: **Feature Alignment**: MLP projector aligns PatchTST encoder features with TiViT or Mantis-extracted features using contrastive loss.

:star2: **Intermediate Layer Output**: Extract features from any encoder layer via `encoder_depth` parameter.

:star2: **Two Feature Extractors**:
  - **TiViT**: Converts time series to images, uses pre-trained ViT (768-dim features)
  - **Mantis**: Uses Mantis8M model from mantis-tsfm (256-dim features)

![alt text](https://github.com/yuqinie98/PatchTST/blob/main/pic/model.png)

## Getting Started

### Installation

```bash
pip install -r requirements.txt
```

### Data Setup

Download datasets from [Autoformer](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) and place CSV files in `./dataset/`.

### Training

#### Standard PatchTST Training
```bash
python -u run_longExp.py --is_training 1 --model PatchTST --data custom \
  --root_path ./dataset/ --data_path weather.csv \
  --features M --seq_len 336 --pred_len 96 \
  --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 \
  --patch_len 16 --stride 8 --batch_size 128 --learning_rate 0.0001
```

Or use provided shell scripts:
```bash
sh ./scripts/etth1_mantis.sh         # PatchTST_REPA with Mantis feature alignment
sh ./scripts/etth1_PatchTST.sh       # Original PatchTST (baseline)
```

- **etth1_REPA.sh**: Runs PatchTST with contrastive learning feature alignment using Mantis as feature extractor
- **etth1_PatchTST.sh**: Runs original PatchTST (baseline, no projector or feature extractor)

#### PatchTST_REPA + Mantis Feature Alignment (Default)
```bash
python -u run_longExp.py --is_training 1 --model PatchTST_REPA --data custom \
  --root_path ./dataset/ --data_path weather.csv \
  --features M --seq_len 336 --pred_len 96 \
  --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 \
  --patch_len 16 --stride 8 --batch_size 128 --learning_rate 0.0001 \
  --mantis_pretrained ./Mantis --lambda_contrastive 0.5
```

#### PatchTST_REPA + TiViT Feature Alignment
```bash
python -u run_longExp.py --is_training 1 --model PatchTST_REPA --data custom \
  --root_path ./dataset/ --data_path weather.csv \
  --features M --seq_len 336 --pred_len 96 \
  --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 \
  --patch_len 16 --stride 8 --batch_size 128 --learning_rate 0.0001 \
  --feature_extractor tivit --projector_dim 768 \
  --lambda_contrastive 0.5 \
  --tivit_pretrained ./open_clip/open_clip_model.safetensors
```

#### Original PatchTST (baseline)
```bash
python -u run_longExp.py --is_training 1 --model PatchTST --data custom \
  --root_path ./dataset/ --data_path weather.csv \
  --features M --seq_len 336 --pred_len 96 \
  --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 \
  --patch_len 16 --stride 8 --batch_size 128 --learning_rate 0.0001
```

## Architecture

### Model Components

1. **PatchTST_backbone**: Core model with RevIN, patching, Transformer encoder, and flatten head
2. **MLP Projector**: Maps intermediate encoder features to feature extractor space (768-dim for TiViT, 256-dim for Mantis)
3. **TiViT**: Converts time series to images and uses pre-trained ViT for feature extraction (768-dim)
4. **Mantis**: Uses Mantis8M from mantis-tsfm for feature extraction (256-dim)

### Data Flow

```
Input (Batch, Input Length, Channels)
  → RevIN norm → Patching → Transformer Encoder → Flatten Head → Output
  → RevIN denorm
```

### Model Output

The model returns a tuple `(output, zs)`:
- `output`: Final prediction output
- `zs`: Intermediate output from the encoder layer specified by `encoder_depth`

## Key Hyperparameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `patch_len` | Length of each patch | 16 |
| `stride` | Stride between patches | 8 |
| `seq_len` | Input sequence length (look-back window) | 336 |
| `pred_len` | Prediction sequence length | 96 |
| `d_model` | Transformer model dimension | 128 |
| `n_heads` | Number of attention heads | 16 |
| `e_layers` | Number of encoder layers | 4 |
| `encoder_depth` | Which encoder layer to extract intermediate output | 2 |
| `projector_dim` | MLP projector output dimension (768 for TiViT/Chronos, 256 for Mantis) | 768 |
| `feature_extractor` | Feature extractor: `tivit`, `mantis` or `chronos` | `mantis` |
| `lambda_contrastive` | Weight for contrastive loss | 0.5 |

Note: Using `model=PatchTST_REPA` automatically enables the projector and contrastive loss. Using `model=PatchTST` runs the original PatchTST baseline.

## Feature Alignment

The training uses a combined loss:

```
loss = MSE_loss + lambda_contrastive * contrastive_loss
```

Where:
- `MSE_loss`: Standard MSE between prediction and ground truth
- `contrastive_loss`: Cosine similarity loss between projected PatchTST features and feature extractor features

### Shape Transformation

#### DataLoader Output
```
batch_x: (batch, seq_len, nvars) = (32, 336, 7)
batch_y: (batch, seq_len+pred_len, nvars) = (32, 432, 7)
```

#### PatchTST_backbone Forward
```
输入: (32, 7, 336)              # (batch, nvars, seq_len)
  ↓ Patching: (32, 7, 41, 16)  # (batch, nvars, patch_num, patch_len)
  ↓ W_P (Linear): (32, 7, 41, 16)  # (batch, nvars, patch_num, d_model)
  ↓ Transformer Encoder (d_model stays constant)
  ↓ MLP Projector: (32, 7, 256)  # (batch, nvars, projector_dim)

output: (32, 7, 96)             # (batch, nvars, pred_len)
zs: (32, 7, 256)                # (batch, nvars, projector_dim)
```

#### Feature Extractor (TiViT / Mantis)
Feature is extracted from target prediction part only: `target[:, -pred_len:, :]`

- **TiViT**: Extracts 768-dim features per channel
- **Mantis**: Resizes to 512, extracts 256-dim features per channel

```
zs_tilde: (32, 7, 256)  # (batch, nvars, feature_dim)
```

#### Final Output
```
output:   (32, 96, 7)   # (batch, pred_len, nvars)
zs:       (32, 7, 256)  # (batch, nvars, projector_dim)
zs_tilde: (32, 7, 256)  # (batch, nvars, feature_dim)
```

## Directory Structure

```
PatchTST/
├── run_longExp.py              # Main entry point
├── layers/
│   ├── PatchTST_backbone.py    # Core model with MLP projector
│   ├── PatchTST_layers.py
│   ├── RevIN.py
│   ├── SelfAttention_Family.py
│   └── Tivit.py               # TiViT: Time series to ViT embedding
├── models/
│   └── PatchTST.py            # PatchTST model (includes TiViT/Mantis)
├── exp/
│   └── exp_main.py             # Training with feature alignment
├── data_provider/              # Data loading
├── scripts/                    # Training scripts
│   ├── etth1_REPA.sh            # PatchTST + Mantis feature alignment
│   └── etth1_PatchTST.sh       # Original PatchTST (baseline)
├── diagnose_results/          # Debug scripts
├── dataset/                   # Place downloaded CSV files here
├── open_clip/                 # Pre-trained CLIP model weights
└── Mantis/                    # Pre-trained Mantis model weights
```

## Acknowledgement

This project is based on the official PatchTST implementation. We appreciate the following repos for their valuable code base and datasets:

- https://github.com/yuqinie98/PatchTST
- https://github.com/cure-lab/LTSF-Linear
- https://github.com/zhouhaoyi/Informer2020
- https://github.com/thuml/Autoformer
- https://github.com/timeseriesAI/tsai
