# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an extended implementation of PatchTST: "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers" (ICLR 2023). It uses a patch-based approach for time series forecasting with Transformer architectures.

This project extends PatchTST with feature alignment using contrastive learning. Supports three feature extractors: **TiViT** (Time series to ViT), **Mantis** (from mantis-tsfm), and **Chronos2** (from Amazon). The MLP projector maps PatchTST's intermediate encoder features to align with extracted features from the target sequence.

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

### Running with PatchTST_REPA (Feature Alignment with Mantis)
```bash
python -u run_longExp.py --is_training 1 --model PatchTST_REPA --data custom \
  --root_path ./dataset/ --data_path weather.csv \
  --features M --seq_len 336 --pred_len 96 \
  --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 \
  --patch_len 16 --stride 8 --batch_size 128 --learning_rate 0.0001 \
  --mantis_pretrained ./Mantis --lambda_contrastive 0.5
```
This trains PatchTST with a contrastive loss that aligns PatchTST's projected encoder features with Mantis-extracted features.

### Running with PatchTST_REPA + TiViT Feature Alignment
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
This trains PatchTST with a contrastive loss that aligns PatchTST's projected encoder features with TiViT-extracted features.

### Running with PatchTST_REPA + Chronos Feature Alignment
```bash
python -u run_longExp.py --is_training 1 --model PatchTST_REPA --data custom \
  --root_path ./dataset/ --data_path weather.csv \
  --features M --seq_len 336 --pred_len 96 \
  --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 \
  --patch_len 16 --stride 8 --batch_size 128 --learning_rate 0.0001 \
  --feature_extractor chronos --projector_dim 768 \
  --lambda_contrastive 0.5 \
  --chronos_pretrained ./Chronos2
```
This trains PatchTST with a contrastive loss that aligns PatchTST's projected encoder features with Chronos2-extracted features.

### Running with Chronos patch_wise Contrastive Loss
```bash
python -u run_longExp.py --is_training 1 --model PatchTST_REPA --data custom \
  --root_path ./dataset/ --data_path weather.csv \
  --features M --seq_len 336 --pred_len 96 \
  --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 \
  --patch_len 16 --stride 8 --batch_size 128 --learning_rate 0.0001 \
  --feature_extractor chronos --projector_dim 768 \
  --lambda_contrastive 0.5 --contrastive_type patch_wise \
  --chronos_pretrained ./Chronos2
```
This uses patch-wise contrastive loss instead of mean pooling for Chronos.

### Running Original PatchTST (without projector)
```bash
python -u run_longExp.py --is_training 1 --model PatchTST --data custom \
  --root_path ./dataset/ --data_path weather.csv \
  --features M --seq_len 336 --pred_len 96 \
  --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 \
  --patch_len 16 --stride 8 --batch_size 128 --learning_rate 0.0001
```

Or use provided shell scripts:
```bash
sh ./scripts/etth1_PatchTST.sh   # Original PatchTST (baseline)
sh ./scripts/etth1_mantis.sh      # PatchTST_REPA: PatchTST + Mantis feature alignment
sh ./scripts/etth1_Chronos2.sh   # PatchTST_REPA: PatchTST + Chronos2 feature alignment
sh ./scripts/etth1_Chronos2_patchwise.sh  # PatchTST_REPA: Chronos with patch_wise contrastive loss
```

### Training Log Format
```
***Epoch: 1 | cost time: 123.457 | lr: 1.2346e-04  <- best model updated
Epoch: 2 | cost time: 123.457 | lr: 1.2346e-04     <- normal
Steps: 50 | Train Loss: 0.1234567 | Train MSE: 0.1000000 | Train Contrastive: 0.0234567 | Vali Loss: 0.1500000 | Test Loss: 0.1600000
```

- `***` prefix indicates best model was updated in this epoch
- `cost time`: 3 decimal places
- `lr`: scientific notation with 4 decimal places

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

**Model Output**: The model returns:
- When using `PatchTST` model: only `output` (original PatchTST)
- When using `PatchTST_REPA` model with `return_projector=False`: tuple `(output, zs)`
- When using `PatchTST_REPA` model with `return_projector=True`: tuple `(output, zs, zs_tilde)`
- `output`: Final prediction output
- `zs`: Intermediate output from the encoder layer specified by `encoder_depth` (default: 4), after MLP projector
- `zs_tilde`: Feature extractor (TiViT/Mantis/Chronos) extracted features from target sequence

### Feature Alignment (TiViT, Mantis or Chronos)

The system combines PatchTST with a feature extractor for enhanced feature representation. Supports two extractors:

1. **MLP Projector** (`layers/PatchTST_backbone.py`): Maps `zs` (from encoder intermediate layer) to feature extractor space
   - Input: `(bs, nvars, d_model, patch_num)` → reshape → `(bs*nvars*patch_num, d_model)`
   - MLP: `d_model` → hidden → `projector_dim` (default: 768 for TiViT, 256 for Mantis)
   - Output: `(bs, nvars, patch_num, projector_dim)` - mean pooling is done in contrastive loss computation

2. **TiViT Feature Extraction** (`layers/Tivit.py`): Converts time series to images and uses pre-trained ViT
   - Extracts features from target sequence per channel
   - Output: `(bs, nvars, 768)` - 768-dimensional features

3. **Mantis Feature Extraction** (`models/PatchTST.py`): Uses Mantis8M model from mantis-tsfm
   - Resizes input to 512 length, extracts 256-dim features per channel
   - Output: `(bs, nvars, 256)` - 256-dimensional features

4. **Contrastive Loss**: Aligns PatchTST projected features with feature extractor features
   - Combined loss: `MSE_loss + lambda * contrastive_loss`
   - Default lambda: 0.5

### Complete Shape Transformation (ETTh1 + Mantis Example)

Key parameters: seq_len=336, pred_len=96, nvars=7, patch_len=16, stride=8, d_model=16, projector_dim=256

#### DataLoader Output
```
batch_x: (batch, seq_len, nvars) = (32, 336, 7)
batch_y: (batch, seq_len+pred_len, nvars) = (32, 432, 7)
```

#### PatchTST_backbone Forward
```
输入: (32, 7, 336)                    # (batch, nvars, seq_len)
  ↓ 1. unfold (patching): (32, 7, 41, 16)     # (batch, nvars, patch_num, patch_len)
  ↓ 2. permute: (32, 7, 16, 41)               # (batch, nvars, patch_len, patch_num)
  ↓ 3. W_P (Linear: 16→16): (32, 7, 41, 16)  # (batch, nvars, patch_num, d_model)
  ↓ 4. reshape: (224, 41, 16)                 # (batch*nvars, patch_num, d_model)
  ↓ 5. Transformer Encoder (d_model=16 stays constant)
  ↓ 6. reshape back: (32, 7, 16, 41)          # (batch, nvars, d_model, patch_num)
  ↓ 7. MLP Projector: (32, 7, 41, 256)         # (batch, nvars, patch_num, projector_dim)

zs: (32, 7, 41, 256)                        # (batch, nvars, patch_num, projector_dim) - no mean pooling
output: (32, 7, 96)                         # (batch, nvars, pred_len)
```

#### Mantis Feature Extraction (uses target[:, -pred_len:, :])
```
target_pred: (32, 96, 7)     # target[:, -pred_len:, :] - prediction part only
  ↓ permute: (32, 7, 96)    # (batch, nvars, pred_len)
  ↓ interpolate: (32, 7, 512) # resize to 512
  ↓ Mantis.transform: (32, 1792)  # (batch, nvars*256)
  ↓ reshape: (32, 7, 256)    # (batch, nvars, 256) = zs_tilde

zs_tilde: (32, 7, 256)       # (batch, nvars, 256)
```

#### Final Output
```
output:   (32, 96, 7)   # (batch, pred_len, nvars)
zs:       (32, 7, patch_num, projector_dim)  # (batch, nvars, patch_num, projector_dim) - no mean pooling
zs_tilde: (32, 7, 256)  # (batch, nvars, 256) for Mantis, or (batch, nvars, num_patches, 768) for Chronos
```

Note: Mean pooling over patches is now done in `_compute_contrastive_loss` in `exp/exp_main.py` instead of in the model.

### Key Hyperparameters
- `patch_len`: Length of each patch (default: 16)
- `stride`: Stride between patches (default: 8)
- `seq_len`: Input sequence length (look-back window)
- `pred_len`: Prediction sequence length (forecast horizon)
- `d_model`: Transformer model dimension
- `n_heads`: Number of attention heads
- `e_layers`: Number of encoder layers (default: 3)
- `encoder_depth`: Which encoder layer to extract intermediate output (default: 4)
- `revin`: Enable reversible instance normalization
- `decomposition`: Enable series decomposition
- `projector_dim`: MLP projector output dimension (default: 768, auto-adjusts to 256 when using Mantis)
- `lambda_contrastive`: Weight for contrastive loss (default: 0.5)
- `tivit_pretrained`: TiViT pretrained model path (default: `./open_clip/open_clip_model.safetensors`)
- `feature_extractor`: Feature extractor for contrastive loss: `tivit`, `mantis` or `chronos` (default: `mantis`)
- `mantis_pretrained`: Mantis pretrained model path (default: `./Mantis`)
- `chronos_pretrained`: Chronos pretrained model path (default: `./Chronos2`)
- `contrastive_type`: Contrastive loss type for Chronos: `mean_pool` (mean pooling) or `patch_wise` (interpolate then per-patch) (default: `mean_pool`)

Note:
- Best model is automatically saved in memory during training and loaded for test
- Projector and feature extractor are only created when using `PatchTST_REPA` model
- `PatchTST` model runs original PatchTST (baseline)
- `PatchTST_REPA` model automatically enables projector and contrastive loss

## Directory Structure

```
PatchTST/
├── run_longExp.py              # Main entry point
├── layers/
│   ├── PatchTST_backbone.py   # Core model with MLP projector
│   ├── PatchTST_layers.py
│   ├── RevIN.py
│   ├── SelfAttention_Family.py
│   └── Tivit.py               # TiViT: Time series to ViT embedding
├── models/
│   └── PatchTST.py            # PatchTST model (includes TiViT/Mantis/Chronos)
├── exp/                        # Experiment classes
├── data_provider/              # Data loading
├── scripts/                    # Training scripts
│   ├── etth1_PatchTST.sh      # Original PatchTST (baseline)
│   ├── etth1_mantis.sh        # PatchTST + Mantis feature alignment
│   └── etth1_Chronos2.sh      # PatchTST + Chronos2 feature alignment
├── diagnose_results/           # Debug scripts
├── Formers/                    # Baseline models
├── utils/                      # Utilities
├── dataset/                    # Place downloaded CSV files here
├── open_clip/                  # Pre-trained CLIP model weights
├── Mantis/                     # Pre-trained Mantis model weights
├── Chronos2/                   # Pre-trained Chronos2 model weights
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

## Mantis (Time series to ViT via mantis-tsfm)

Mantis module uses Mantis8M from mantis-tsfm for feature extraction.

### Usage in PatchTST

```python
# Mantis is automatically created in models/PatchTST.py when feature_extractor='mantis'
# The feature extraction is done in PatchTST.forward() when return_projector=True
```

### Input/Output Shapes

- **Input**: `(batch, pred_len, nvars)` - uses only prediction part
- **Preprocessing**: Resize to 512 via linear interpolation
- **Mantis transform**: `(N, C, T)` → `(N, C*256)` - per-channel embeddings
- **Output**: `(batch, nvars, 256)` - 256-dimensional features

### Key Differences from TiViT

| Feature | TiViT | Mantis | Chronos |
|---------|-------|--------|---------|
| Output dimension | 768 | 256 | 768 |
| Input resize | Uses ViT native | Interpolate to 512 | Native (patched) |
| Model | CLIP ViT-B/16 | Mantis8M | Chronos2 (T5-based) |
| Pretrained path | `./open_clip/` | `./Mantis/` | `./Chronos2/` |

## Chronos (Time series via Chronos2)

Chronos module uses Chronos2 from Amazon for feature extraction.

### Usage in PatchTST

```python
# Chronos is automatically created in models/PatchTST.py when feature_extractor='chronos'
# The feature extraction is done in PatchTST.forward() when return_projector=True
```

### Input/Output Shapes

- **Input**: `(batch, pred_len, nvars)` - uses only prediction part, permuted to `(batch, nvars, pred_len)`
- **Chronos embed**: Returns list of embeddings per sample
- **Each embedding**: `(n_variates, num_patches, 768)` - includes REG token and output patch
- **Post-processing**: Remove REG token and output patch - no mean pooling in model
- **Output**: `(batch, nvars, num_patches, 768)` - patch-level features, mean pooling done in contrastive loss

### Key Features

- Uses T5-based transformer architecture
- 12 layers, 12 heads, 768 hidden dimensions
- Native patch-based processing (patch_size=16)
- Pre-trained on multiple time series datasets
- Supports two contrastive loss types:
  - `mean_pool`: Mean pool over patches before computing similarity
  - `patch_wise`: Interpolate zs_project to match zs_tilde patch count, compute per-patch similarity

## Datasets

Standard benchmark datasets: ETTm1, ETTm2, ETTh1, ETTh2, electricity, traffic, weather, illness, exchange_rate, ili (download from Autoformer drive).

## Debug & Diagnostics

- `diagnose_results/debug_shapes.py` - Debug script to check tensor shapes during training
- `diagnose_results/compare_models.py` - Compare our implementation with original PatchTST to verify numerical consistency (forward pass and gradients)
- `diagnose_results/compare_loss.py` - Compare loss curves between PatchTST and PatchTST_REPA experiments

## Bug Fixes

- **Projector input dimension**: Fixed MLP projector input dimension from `head_nf` (d_model * patch_num) to `d_model` in `layers/PatchTST_backbone.py`
- **TiViT moved to PatchTST model**: TiViT creation moved from `exp/exp_main.py` to `models/PatchTST.py` for better encapsulation
- **Mantis support**: Added support for Mantis8M feature extractor in addition to TiViT
- **Chronos support**: Added Chronos2 as third feature extractor option
- **Feature extraction from prediction part**: Feature extractor now uses only `target[:, -pred_len:, :]` for alignment
- **Model-based projector switch**: Using `model=PatchTST` runs original PatchTST, `model=PatchTST_REPA` enables projector and feature alignment
- **Original PatchTST compatibility**: PatchTST model behaves identically to original PatchTST:
  - TSTiEncoder only computes final output (no intermediate extraction)
  - PatchTST_backbone.forward returns only `output` (not tuple)
  - Model parameters and architecture match original PatchTST exactly (verified via `diagnose_results/compare_models.py`)
- **Best model in memory**: Best model is now saved in memory during training and automatically loaded for test (no checkpoint file saved by default)
- **Mean pooling moved to contrastive loss**: Mean pooling over patches is now done in `_compute_contrastive_loss` in `exp/exp_main.py` instead of in the model (PatchTST_backbone and Chronos extraction)
- **Chronos contrastive_type**: Added `contractive_type` hyperparameter with two options: `mean_pool` (mean pooling over patches) and `patch_wise` (interpolate zs_project to match zs_tilde patch count, compute per-patch similarity)
- **Training log format**: Fixed to use 3 decimal places for cost time, 4 decimal places scientific notation for lr, and `***` prefix at end of line for best model updates
- **is_best_update error**: Fixed local variable referenced before assignment error by computing vali_loss before using is_best_update
