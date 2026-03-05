# PatchTST + TiViT Feature Alignment

### This is an extended implementation of PatchTST: [A Time Series is Worth 64 Words: Long-term Forecasting with Transformers](https://arxiv.org/abs/2211.14730)

This project extends PatchTST with TiViT (Time series to ViT) feature alignment using contrastive learning. The MLP projector maps PatchTST's intermediate encoder features to align with TiViT-extracted features from the target sequence.

## Key Features

:star2: **Patch-based Time Series Modeling**: Segmentation of time series into subseries-level patches served as input tokens to Transformer.

:star2: **Channel-independence**: Each channel contains a single univariate time series that shares the same embedding and Transformer weights.

:star2: **TiViT Feature Alignment**: Optional MLP projector aligns PatchTST encoder features with TiViT-extracted features using contrastive loss.

:star2: **Intermediate Layer Output**: Extract features from any encoder layer via `encoder_depth` parameter.

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
sh ./scripts/PatchTST/weather.sh
```

#### PatchTST + TiViT Feature Alignment
```bash
python -u run_longExp.py --is_training 1 --model PatchTST --data custom \
  --root_path ./dataset/ --data_path weather.csv \
  --features M --seq_len 336 --pred_len 96 \
  --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 \
  --patch_len 16 --stride 8 --batch_size 128 --learning_rate 0.0001 \
  --use_projector 1 --projector_dim 768 --lambda_contrastive 0.5
```

## Architecture

### Model Components

1. **PatchTST_backbone**: Core model with RevIN, patching, Transformer encoder, and flatten head
2. **MLP Projector** (optional): Maps intermediate encoder features to TiViT feature space
3. **TiViT**: Converts time series to images and uses pre-trained ViT for feature extraction

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
| `use_projector` | Use MLP projector (1: use, 0: not use) | 0 |
| `projector_dim` | MLP projector output dimension | 768 |
| `lambda_contrastive` | Weight for contrastive loss | 0.5 |

## TiViT Feature Alignment

When `use_projector=1`, the training uses a combined loss:

```
loss = MSE_loss + lambda_contrastive * contrastive_loss
```

Where:
- `MSE_loss`: Standard MSE between prediction and ground truth
- `contrastive_loss`: Cosine similarity loss between projected PatchTST features and TiViT features

The MLP projector processes intermediate encoder outputs (`zs`) per patch:
- Input: `(bs, nvars, d_model, patch_num)` → reshape → `(bs*nvars*patch_num, d_model)`
- MLP: `d_model` → hidden → `projector_dim`
- Output: `(bs, nvars, projector_dim)` after mean pooling over patches

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
│   └── PatchTST.py
├── exp/
│   └── exp_main.py             # Training with TiViT alignment
├── data_provider/              # Data loading
├── scripts/PatchTST/           # Training scripts
├── diagnose_results/          # Debug scripts
└── dataset/                   # Place downloaded CSV files here
```

## Acknowledgement

This project is based on the official PatchTST implementation. We appreciate the following repos for their valuable code base and datasets:

- https://github.com/yuqinie98/PatchTST
- https://github.com/cure-lab/LTSF-Linear
- https://github.com/zhouhaoyi/Informer2020
- https://github.com/thuml/Autoformer
- https://github.com/timeseriesAI/tsai
