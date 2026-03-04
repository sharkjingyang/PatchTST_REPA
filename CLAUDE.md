# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PatchTST is an official implementation of the ICLR 2023 paper: "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers". It uses a patch-based approach for time series forecasting with Transformer architectures.

The repository contains the supervised learning implementation for forecasting.

## Common Commands

### Installation
```bash
pip install -r requirements.txt
```

### Data Setup
Download datasets from [Autoformer](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) and place CSV files in `./dataset/`

### Running Supervised Training
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

## Architecture

### Key Model Components

1. **PatchTST_backbone** (`layers/PatchTST_backbone.py`): Core model with:
   - RevIN (Reversible Instance Normalization) for domain-agnostic normalization
   - Patching layer that segments time series into sub-series patches
   - TSTiEncoder Transformer backbone
   - Flatten head for prediction

2. **RevIN** (`layers/RevIN.py`): Reversible Instance Normalization - normalizes each channel independently and denormalizes after prediction

3. **Attention** (`layers/SelfAttention_Family.py`): Various attention mechanisms including:
   - Full self-attention
   - ProbSparse self-attention (from Informer)

4. **Decomposition** (`layers/PatchTST_layers.py`): Optional series decomposition (trend + residual)

### Data Flow
```
Input (Batch, Input Length, Channels)
  → RevIN norm → Patching → Transformer Encoder → Flatten Head → Output
  → RevIN denorm
```

**Model Output**: The model returns a tuple `(output, zs)` where:
- `output`: Final prediction output
- `zs`: Intermediate output from the encoder layer specified by `encoder_depth` (default: layer 2)

### Key Hyperparameters
- `patch_len`: Length of each patch (default: 16)
- `stride`: Stride between patches (default: 8)
- `seq_len`: Input sequence length (look-back window)
- `pred_len`: Prediction sequence length (forecast horizon)
- `d_model`: Transformer model dimension
- `n_heads`: Number of attention heads
- `e_layers`: Number of encoder layers (default: 4)
- `encoder_depth`: Which encoder layer to extract intermediate output (default: 2)
- `revin`: Enable reversible instance normalization
- `decomposition`: Enable series decomposition
- `save_checkpoint`: Whether to save model checkpoint (1: save, 0: not save, default: 0)

## Directory Structure

```
PatchTST/
├── run_longExp.py              # Main entry point
├── layers/
│   ├── PatchTST_backbone.py
│   ├── PatchTST_layers.py
│   ├── RevIN.py
│   └── SelfAttention_Family.py
├── models/
│   └── PatchTST.py
├── exp/                        # Experiment classes
├── data_provider/              # Data loading
├── scripts/PatchTST/            # Training scripts
├── Formers/                    # Baseline models (Informer, Autoformer, etc.)
├── utils/                      # Utilities
├── dataset/                    # Place downloaded CSV files here
├── README.md
├── CLAUDE.md
└── LICENSE
```

## Datasets

Standard benchmark datasets: ETTm1, ETTm2, ETTh1, ETTh2, electricity, traffic, weather, illness, exchange_rate, ili (download from Autoformer drive).
