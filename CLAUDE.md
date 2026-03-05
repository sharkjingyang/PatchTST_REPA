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

### Running with PatchTST + TiViT Feature Alignment
```bash
python -u run_longExp.py --is_training 1 --model PatchTST --data custom \
  --root_path ./dataset/ --data_path weather.csv \
  --features M --seq_len 336 --pred_len 96 \
  --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 \
  --patch_len 16 --stride 8 --batch_size 128 --learning_rate 0.0001 \
  --projector_dim 768 --lambda_contrastive 0.5 \
  --tivit_pretrained ./open_clip/open_clip_model.safetensors
```
This trains PatchTST with a contrastive loss that aligns PatchTST's projected encoder features with TiViT-extracted features.

Or use provided shell scripts:
```bash
sh ./scripts/PatchTST/etth1.sh
```

## Architecture

### Key Model Components

1. **PatchTST_backbone** (`layers/PatchTST_backbone.py`): Core model with:
   - RevIN (Reversible Instance Normalization) for domain-agnostic normalization
   - Patching layer that segments time series into sub-series patches
   - TSTiEncoder Transformer backbone
   - Flatten head for prediction
   - **MLP Projector** (optional): Maps `zs` features to TiViT feature space

2. **RevIN** (`layers/RevIN.py`): Reversible Instance Normalization - normalizes each channel independently and denormalizes after prediction

3. **Attention** (`layers/SelfAttention_Family.py`): Various attention mechanisms including:
   - Full self-attention
   - ProbSparse self-attention (from Informer)

4. **Decomposition** (`layers/PatchTST_layers.py`): Optional series decomposition (trend + residual)

5. **TiViT** (`layers/Tivit.py`): Time series to ViT embedding - converts time series to images and uses pre-trained ViT for feature extraction

### Data Flow
```
Input (Batch, Input Length, Channels)
  → RevIN norm → Patching → Transformer Encoder → Flatten Head → Output
  → RevIN denorm
```

**Model Output**: The model returns a tuple `(output, zs, zs_tilde)` where:
- `output`: Final prediction output
- `zs`: Intermediate output from the encoder layer specified by `encoder_depth` (default: layer 2), after MLP projector if enabled
- `zs_tilde`: TiViT-extracted features from target sequence (returned when `return_projector=True`)

### TiViT Feature Alignment

The system combines PatchTST with TiViT for enhanced feature representation:

1. **MLP Projector** (`layers/PatchTST_backbone.py`): Maps `zs` (from encoder intermediate layer) to TiViT feature space
   - Input: `(bs, nvars, d_model, patch_num)` → reshape → `(bs*nvars*patch_num, d_model)`
   - MLP: `d_model` → hidden → `projector_dim` (default: 768)
   - Output: `(bs, nvars, projector_dim)` after mean pooling over patches

2. **TiViT Feature Extraction** (`models/PatchTST.py`): TiViT is created inside the PatchTST model. Extracts features from target sequence `batch_y` per channel and returns `(bs, nvars, d_vit)`

3. **Contrastive Loss**: Aligns PatchTST projected features with TiViT features
   - Combined loss: `MSE_loss + lambda * contrastive_loss`
   - Default lambda: 0.5

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
- `projector_dim`: MLP projector output dimension (default: 768)
- `lambda_contrastive`: Weight for contrastive loss (default: 0.5)
- `tivit_pretrained`: TiViT pretrained model path (default: `./open_clip/open_clip_model.safetensors`)

Note: Projector and TiViT are always created. Use `return_projector=True` in training to compute contrastive loss (vali/test will skip TiViT inference for speed).

## Directory Structure

```
PatchTST/
├── run_longExp.py              # Main entry point
├── layers/
│   ├── PatchTST_backbone.py
│   ├── PatchTST_layers.py
│   ├── RevIN.py
│   ├── SelfAttention_Family.py
│   └── Tivit.py                # TiViT: Time series to ViT embedding
├── models/
│   └── PatchTST.py
├── exp/                        # Experiment classes
├── data_provider/              # Data loading
├── scripts/PatchTST/           # Training scripts
├── Formers/                    # Baseline models (Informer, Autoformer, etc.)
├── utils/                      # Utilities
├── dataset/                    # Place downloaded CSV files here
├── open_clip/                  # Pre-trained CLIP model weights
├── README.md
├── CLAUDE.md
└── LICENSE
```

## TiViT (Time series to ViT)

TiViT module (`layers/Tivit.py`) converts time series to images and uses pre-trained Vision Transformer (ViT) for feature extraction.

### Key Functions

1. **`get_device()`** - Auto-detect GPU/CPU
2. **`get_processor_vit(model_name)`** - Load CLIP model processor and ViT
3. **`get_patch_size(patch_size, T)`** - Calculate patch size:
   - Integer: direct use
   - `"sqrt"`: `int(sqrt(T))`
   - `"linspace"`: list of patch sizes
4. **`get_tivit(...)`** - Create TiViT model with specified parameters
5. **`embed(model, dataloader, channels, device)`** - Extract embeddings from DataLoader
6. **`get_TS_Tivit_embed(model, dataloader, channels, device)`** - High-level wrapper

### Usage Example

```python
from layers.Tivit import get_tivit, get_patch_size, get_device

# Parameters
model_name = "laion/CLIP-ViT-B-16-laion2B-s34B-b88K"
model_layer = 6
aggregation = "mean"
stride = 0.1
patch_size = "sqrt"
seq_len = 336
device = get_device()

# Get actual patch size
actual_patch_size = get_patch_size(patch_size, seq_len)

# Create model
tivit = get_tivit(
    model_name=model_name,
    model_layer=model_layer,
    aggregation=aggregation,
    stride=stride,
    patch_size=actual_patch_size,
    device=device,
)
tivit.eval()

# Forward pass: (B, T, D) -> (B, D)
x = torch.randn(4, seq_len, 7)  # batch=4, seq_len=336, channels=7
with torch.no_grad():
    output = tivit(x)  # (4, 768)

# Extract embeddings from DataLoader
from torch.utils.data import DataLoader, TensorDataset

fake_data = torch.randn(100, 7, seq_len)  # (N, C, T)
dataset = TensorDataset(fake_data)
loader = DataLoader(dataset, batch_size=16)

embeds = embed(tivit, loader, channels=7, device=device)
# Output shape: (100, 7, 768)
```

### Input/Output Shapes

- **TiViT forward**: `(B, T, 1)` → `(B, 768)`
- **embed function**: `(N, C, T)` → `(N, C, 768)` - per-channel embeddings
- **Model**: ViT-B/16 outputs 768-dim features

## Datasets

Standard benchmark datasets: ETTm1, ETTm2, ETTh1, ETTh2, electricity, traffic, weather, illness, exchange_rate, ili (download from Autoformer drive).

## Debug & Diagnostics

- `diagnose_results/debug_shapes.py` - Debug script to check tensor shapes during training

## Bug Fixes

- **Projector input dimension**: Fixed MLP projector input dimension from `head_nf` (d_model * patch_num) to `d_model` in `layers/PatchTST_backbone.py`
- **TiViT moved to PatchTST model**: TiViT creation moved from `exp/exp_main.py` to `models/PatchTST.py` for better encapsulation
