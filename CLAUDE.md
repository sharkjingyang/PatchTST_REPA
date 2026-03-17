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

### Running with PatchTST_REPA_Fusion (Channel Fusion)
```bash
python -u run_longExp.py --is_training 1 --model PatchTST_REPA_Fusion --data custom \
  --root_path ./dataset/ --data_path weather.csv \
  --features M --seq_len 336 --pred_len 96 \
  --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 \
  --patch_len 16 --stride 8 --batch_size 128 --learning_rate 0.0001 \
  --feature_extractor chronos --projector_dim 768 \
  --lambda_contrastive 0.5 \
  --chronos_pretrained ./Chronos2
```
This trains PatchTST with Channel Fusion branch for better Chronos alignment.

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

### Running PatchTST with Quantile Head
```bash
python -u run_longExp.py --is_training 1 --model PatchTST --data custom \
  --root_path ./dataset/ --data_path weather.csv \
  --features M --seq_len 336 --pred_len 96 \
  --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 \
  --patch_len 16 --stride 8 --batch_size 128 --learning_rate 0.0001 \
  --head_type quantile --num_quantiles 20
```
This trains PatchTST with a quantile prediction head that outputs 20 quantiles (0.01, 0.05, 0.1, ..., 0.99). The model uses Quantile Loss (Pinball Loss) during training and outputs the median quantile (q=0.5) for inference.

Or use provided shell scripts:
```bash
sh ./scripts/PatchTST.sh   # Original PatchTST (baseline)
sh ./scripts/mantis.sh     # PatchTST_REPA: PatchTST + Mantis feature alignment
sh ./scripts/Chronos2.sh   # PatchTST_REPA: PatchTST + Chronos2 feature alignment (patch_wise)
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

6. **Quantile_Head** (`layers/PatchTST_backbone.py`): Quantile prediction head similar to Chronos2
   - Uses ResidualBlock (Linear → GELU → Linear + residual)
   - Output dimension: `(bs, nvars, num_quantiles, pred_len)`
   - Supports 20 quantiles by default (0.01, 0.05, 0.1, ..., 0.99)
   - Uses Quantile Loss (Pinball Loss) for training

7. **PatchwiseHead** (`layers/PatchTST_backbone.py`): Patch-wise prediction head for Channel Fusion
   - Uses shared ResidualBlock per patch (like Chronos2)
   - Input: `(bs, nvars, d_model, output_patch_num)` from Channel Fusion
   - ResidualBlock: `d_model` → `d_model//2` → `output_patch_size` with residual connection
   - Output: `(bs, nvars, pred_len)`
   -参数量比 Flatten_Head 更小 (约 300K vs 24M for d_model=768)

### Data Flow
```
Input (Batch, Input Length, Channels)
  → RevIN norm → Patching → Transformer Encoder → Flatten Head → Output
  → RevIN denorm
```

**Model Output**: The model returns:
- When using `PatchTST` model with `head_type=flatten`: only `output` (original PatchTST), shape `(batch, pred_len, nvars)`
- When using `PatchTST` model with `head_type=quantile`: `output` with shape `(batch, pred_len, nvars, num_quantiles)`
- When using `PatchTST_REPA` model: tuple `(output, zs)` or `(output, zs, zs_tilde)` depending on return_projector
- When using `PatchTST_REPA_Fusion` model: uses Channel Fusion branch, tuple `(output, zs)`
- `output`: Final prediction output
- `zs`: Intermediate output from the encoder layer specified by `encoder_depth` (default: 2)
- `zs_tilde`: Feature extractor (TiViT/Mantis/Chronos) extracted features from target sequence (only for PatchTST_REPA)

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

### Channel Fusion Branch (model=PatchTST_REPA_Fusion)

When using `model=PatchTST_REPA_Fusion`, an additional branch is added for better Chronos alignment:

**Architecture**:
```
Transformer Encoder (d_model)
  ├─ Branch 1 (original): Encoder output → Flatten Head → pred_1
  └─ Branch 2 (new):
       ↓ Extract zs from encoder_depth layer
       ↓ Channel Fusion MLP: flatten d_model*input_patch_num → Linear → d_extractor*output_patch_num
       ↓ Two Attention Blocks (d_model=d_extractor)
       ↓ zs_fused → Patchwise Head → pred_2
```

**Head Types**:
- PatchTST and PatchTST_REPA: uses Flatten_Head or Quantile_Head (controlled by head_type)
- PatchTST_REPA_Fusion: uses Flatten_Head / PatchwiseHead / Quantile_Head (controlled by head_type: 'flatten', 'patchwise', 'quantile')
  - 'patchwise': PatchwiseHead with shared ResidualBlock per patch (~300K params)
  - 'flatten': Flatten_Head with full linear layer (~24M params)
  - 'quantile': Quantile_Head for probabilistic forecasting

**Three Model Options**:
1. `model=PatchTST`: Original PatchTST (baseline)
2. `model=PatchTST_REPA`: PatchTST with MLP Projector + contrastive loss
3. `model=PatchTST_REPA_Fusion`: PatchTST with Channel Fusion branch

**Key Differences**:
- PatchTST_REPA uses MLP Projector to align encoder features with feature extractor
- PatchTST_REPA_Fusion uses Channel Fusion MLP (no projector needed)
- Channel Fusion uses Two Attention Blocks with d_extractor dimension (768 for Chronos)
- output_patch_num = pred_len // output_patch_size (e.g., 96/16 = 6)

**Shape Transformation Example (Chronos, seq_len=336, pred_len=96)**:
```
input_patch_num = 336/16 = 21
output_patch_num = 96/16 = 6
d_extractor = 768

Channel Fusion MLP:
  Input:  (B, nvars, 21, 128)  = (B, nvars, input_patch_num, d_model)
  Flatten → (B, nvars, 2688)   = (B, nvars, d_model*input_patch_num)
  Linear → (B, nvars, 4608)    = (B, nvars, d_extractor*output_patch_num)
  Reshape → (B, nvars, 6, 768) = (B, nvars, output_patch_num, d_extractor)
  Permute → (B, nvars, 768, 6) = (B, nvars, d_extractor, output_patch_num)
```

### Quantile Prediction Head

PatchTST now supports quantile prediction head similar to Chronos2:

1. **Quantile_Head** (`layers/PatchTST_backbone.py`):
   - Input: `(bs, nvars, d_model, input_patch_num)` from encoder
   - Flatten all patches: `(bs, nvars, d_model * input_patch_num)`
   - ResidualBlock: `d_model * input_patch_num` → `d_ff (= head_nf)` → `output_patch_num * num_quantiles * output_patch_size` with residual connection
   - Rearrange to: `(bs, nvars, num_quantiles, pred_len)`
   - Output: `(bs, nvars, num_quantiles, pred_len)`

2. **Quantile Loss** (`exp/exp_main.py`):
   - Implements Pinball Loss: `2 * |(y - ŷ) * (I(y < ŷ) - q)|`
   - Quantiles: 20 evenly spaced from 0.01 to 0.99
   - During training: uses all quantiles for loss computation
   - During inference: outputs median quantile (q=0.5) as point prediction

3. **Output Shape**:
   - Training: `(batch, pred_len, nvars, num_quantiles)`
   - Inference: `(batch, pred_len, nvars)` (median quantile)

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

#### Mantis Feature Extraction (target already sliced to pred_len in exp_main)
```
target_pred: (32, 96, 7)     # target (already sliced to pred_len in exp_main)
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
- `model`: Model name: `PatchTST` (baseline), `PatchTST_REPA` (projector + contrastive), or `PatchTST_REPA_Fusion` (channel fusion)
- `patch_len`: Length of each patch (default: 16)
- `stride`: Stride between patches (default: 16)
- `seq_len`: Input sequence length (look-back window)
- `pred_len`: Prediction sequence length (forecast horizon)
- `d_model`: Transformer model dimension
- `n_heads`: Number of attention heads
- `e_layers`: Number of encoder layers (default: 3)
- `encoder_depth`: Which encoder layer to extract intermediate output (default: 2)
- `revin`: Enable reversible instance normalization
- `decomposition`: Enable series decomposition
- `projector_dim`: MLP projector output dimension (default: 768, auto-adjusts to 256 when using Mantis)
- `lambda_contrastive`: Weight for contrastive loss (default: 0.5)
- `tivit_pretrained`: TiViT pretrained model path (default: `./open_clip/open_clip_model.safetensors`)
- `feature_extractor`: Feature extractor for contrastive loss: `tivit`, `mantis` or `chronos` (default: `mantis`)
- `mantis_pretrained`: Mantis pretrained model path (default: `./Mantis`)
- `chronos_pretrained`: Chronos pretrained model path (default: `./Chronos2`)
- `contrastive_type`: Contrastive loss type for Chronos: `mean_pool` (mean pooling) or `patch_wise` (per-patch similarity with consistent patch_num) (default: `mean_pool`)
- `head_type`: Prediction head type: `flatten` (Flatten_Head for point prediction), `patchwise` (PatchwiseHead for Channel Fusion, reduces params ~98%), `quantile` (Quantile_Head for probabilistic forecasting) (default: `flatten`)
- `num_quantiles`: Number of quantiles for quantile_head (default: 20)
- `output_patch_size`: Output patch size for Channel Fusion branch (default: 16). Used to compute output_patch_num = pred_len // output_patch_size

Note:
- Best model is automatically saved in memory during training and loaded for test
- Projector and feature extractor are only created when using `PatchTST_REPA` or `PatchTST_REPA_Fusion` model
- Model selection is done via `--model` parameter: `PatchTST`, `PatchTST_REPA`, or `PatchTST_REPA_Fusion`

### Parameter Statistics

训练时会自动打印详细的参数统计信息，显示每个模块的参数量：

**模型配置**:
- Model: PatchTST / PatchTST_REPA / PatchTST_REPA_Fusion
- Head type: Flatten_Head / Quantile_Head / PatchwiseHead
- Channel fusion: True / False

**模块参数量分解**:
- Backbone (encoder): Transformer 编码器参数量
- Projector: MLP 投影器参数量 (仅 PatchTST_REPA)
- ChannelFusionMLP: Channel Fusion MLP 参数量 (仅 PatchTST_REPA_Fusion)
- TransformerDecoder: Channel Fusion 的 Transformer Decoder 参数量
- Head: 预测头参数量 (Flatten_Head / PatchwiseHead)
- RevIN: 可逆实例归一化参数量

**示例参数量 (seq_len=336, pred_len=96, d_model=128, Chronos)**:
| 模型 | Backbone | Projector | ChannelFusionMLP | TransformerDecoder | Head | Total (excl. Chronos) |
|------|----------|-----------|------------------|-------------------|------|----------------------|
| PatchTST | 22,180 | - | - | - | 32,352 | 54,532 |
| PatchTST_REPA | 22,180 | 302,208 | - | - | 32,352 | 356,740 |
| PatchTST_REPA_Fusion | 11,394 | - | 1,552,896 | 3,546,625 | 313,760 | 5,424,675 |

注: PatchTST_REPA_Fusion 的 Head 使用 PatchwiseHead，参数量约为 313K，比原 Flatten_Head 的 24M 减少约 98.7%

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
│   ├── PatchTST.sh           # Original PatchTST (baseline)
│   ├── mantis.sh             # PatchTST + Mantis feature alignment
│   └── Chronos2.sh           # PatchTST + Chronos2 feature alignment (patch_wise)
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
  - `patch_wise`: Interpolation done in data prep (batch_y interpolated to seq_len), compute per-patch similarity with consistent patch_num

### Chronos2 Architecture and Prediction

Chronos2 uses a T5-based encoder architecture with patch-based input/output.

#### Token Sequence Construction

```
[Past Tokens] + [REG Token] + [Future Tokens (masked)]
   ↓                ↓               ↓
num_context_patches  1           num_output_patches
```

The input processing:
1. **Patching**: Input sequence is split into patches of size `input_patch_size` (16) with stride `input_patch_stride` (16)
2. **Embedding**: Each patch is embedded via `input_patch_embedding` (Linear: 16 → 768)
3. **Special Tokens**: If `use_reg_token=True`, a REG (representation) token is appended
4. **Future Tokens**: Masked future patches are appended for the decoder

#### Prediction Mechanism

The prediction process (from `chronos/chronos2/model.py`):

```python
# 1. Get encoder output
hidden_states = encoder_outputs[0]  # (batch, num_context_patches + 1 + num_output_patches, d_model)

# 2. Slice last num_output_patches hidden states
forecast_embeds = hidden_states[:, -num_output_patches:]  # (batch, num_output_patches, 768)

# 3. output_patch_embedding (ResidualBlock)
quantile_preds = self.output_patch_embedding(forecast_embeds)
# ResidualBlock: Linear(768→3072) → ReLU → Linear(3072→320) + Linear(768→320) residual
# Output: (batch, num_output_patches, 320) where 320 = 20 quantiles × 16 output_patch_size

# 4. Reshape to quantile predictions
quantile_preds = rearrange(quantile_preds, "b n (q p) -> b q (n p)", ...)
# Output: (batch, 20, num_output_patches * 16) = (batch, 20, pred_len)
```

#### Quantile Loss

Chronos2 uses **Quantile Loss (Pinball Loss)** to train the model with a single observation but 20 quantile predictions.

**Formula:**
```
quantile_loss = 2 * |(y - ŷ) * (I(y < ŷ) - q)|
```

Where:
- `y`: ground truth (broadcast to match prediction shape)
- `ŷ`: quantile prediction
- `q`: quantile value (0.01, 0.05, 0.1, ..., 0.99)
- `I`: indicator function

**Mathematical form:**
$$
\mathcal{L}_q(y, \hat{y}) =
\begin{cases}
q \cdot (y - \hat{y}) & \text{if } y \geq \hat{y} \\
(q - 1) \cdot (y - \hat{y}) & \text{if } y < \hat{y}
\end{cases}
$$

**Intuition:**
- For low quantiles (q=0.1): Model is penalized less when predicting low values
- For high quantiles (q=0.9): Model is penalized less when predicting high values
- For median (q=0.5): Similar to MSE loss

**Output:**
- Shape: `(batch, 20, pred_len)` - 20 quantile predictions for each time step
- Can extract specific quantile (e.g., q=0.5 for point prediction) or compute prediction intervals (e.g., q=0.9 - q=0.1)

#### Key Configuration (from `Chronos2/config.json`)

```json
{
  "d_model": 768,
  "input_patch_size": 16,
  "input_patch_stride": 16,
  "output_patch_size": 16,
  "quantiles": [0.01, 0.05, 0.1, ..., 0.99],  // 20 quantiles
  "num_layers": 12,
  "num_heads": 12,
  "d_ff": 3072
}
```

## Datasets

Standard benchmark datasets: ETTm1, ETTm2, ETTh1, ETTh2, electricity, traffic, weather, illness, exchange_rate, ili (download from Autoformer drive).

## Debug & Diagnostics

- `diagnose_results/debug_shapes.py` - Debug script to check tensor shapes with Chronos2.sh configuration (stride=16, padding_patch=None, head_type=quantile)
- `diagnose_results/compare_models.py` - Compare our implementation with original PatchTST to verify numerical consistency (forward pass and gradients)
- `diagnose_results/compare_loss.py` - Compare loss curves between PatchTST and PatchTST_REPA experiments

## Bug Fixes

- **Projector input dimension**: Fixed MLP projector input dimension from `head_nf` (d_model * patch_num) to `d_model` in `layers/PatchTST_backbone.py`
- **TiViT moved to PatchTST model**: TiViT creation moved from `exp/exp_main.py` to `models/PatchTST.py` for better encapsulation
- **Mantis support**: Added support for Mantis8M feature extractor in addition to TiViT
- **Chronos support**: Added Chronos2 as third feature extractor option
- **Feature extraction from prediction part**: Target is sliced to pred_len in `exp_main.py` before passing to model, feature extractor uses only this sliced portion for alignment
- **Model-based projector switch**: Using `model=PatchTST` runs original PatchTST, `model=PatchTST_REPA` enables projector and feature alignment
- **Original PatchTST compatibility**: PatchTST model behaves identically to original PatchTST:
  - TSTiEncoder only computes final output (no intermediate extraction)
  - PatchTST_backbone.forward returns only `output` (not tuple)
  - Model parameters and architecture match original PatchTST exactly (verified via `diagnose_results/compare_models.py`)
- **Best model in memory**: Best model is now saved in memory during training and automatically loaded for test (no checkpoint file saved by default)
- **Mean pooling moved to contrastive loss**: Mean pooling over patches is now done in `_compute_contrastive_loss` in `exp/exp_main.py` instead of in the model (both zs_project and zs_tilde are mean pooled for mean_pool type)
- **Chronos contrastive_type**: Added `contractive_type` hyperparameter with two options: `mean_pool` (mean pooling over patches for both zs_project and zs_tilde) and `patch_wise` (per-patch similarity; for Chronos, batch_y is interpolated to seq_len in data prep to keep patch_num consistent)
- **Training log format**: Fixed to use 3 decimal places for cost time, 4 decimal places scientific notation for lr, and `***` prefix at end of line for best model updates
- **is_best_update error**: Fixed local variable referenced before assignment error by computing vali_loss before using is_best_update
- **Quantile prediction head**: Added `head_type` and `num_quantiles` hyperparameters to support quantile prediction (similar to Chronos2) with Quantile Loss (Pinball Loss). Uses flatten-all-then-project approach: `d_model * input_patch_num` → `d_ff (=head_nf)` → `output_patch_num * num_quantiles * 16`.
- **--padding_patch parameter**: Fixed parameter name (was incorrectly using `--ending` in scripts)
- **Duplicate head_type argument**: Fixed duplicate `head_type=head_type` in PatchTST.__init__ calls
- **QuantileLoss not used**: Fixed training loop to actually use QuantileLoss when head_type=quantile (previously used MSE). Validation and test still use MSE with median quantile output.
- **Model selection via model_name**: Simplified model selection by using `--model` parameter instead of separate `--use_projector` and `--use_channel_fusion` flags. Three options: `PatchTST`, `PatchTST_REPA`, `PatchTST_REPA_Fusion`.
- **Unified head**: Renamed `head_fused` to `head` for channel fusion branch to simplify code.
- **PatchwiseHead for Channel Fusion**: 为 PatchTST_REPA_Fusion 实现 PatchwiseHead，替换 Flatten_Head。使用共享的 ResidualBlock (d_model → d_model//2 → output_patch_size)，大幅减少参数量 (从 24M 降至约 300K)。
- **Detailed parameter statistics**: 更新参数统计代码，显示每个模块的详细参数量分解 (Backbone, Projector, ChannelFusionMLP, TransformerDecoder, Head, RevIN)。
- **head_type='patchwise' option**: 添加 head_type='patchwise' 选项，允许在 PatchTST_REPA_Fusion 中选择使用 PatchwiseHead (patchwise) 或 Flatten_Head (flatten)。
- **Fix return value when return_projector=False**: 修复当 return_projector=False 时模型错误返回 (output, zs) 而不是只返回 output 的问题。
- **Fix output shape for REPA models**: 修复 PatchTST_REPA 和 PatchTST_REPA_Fusion 的输出形状，添加 permute 使输出为 (bs, pred_len, nvars) 与其他组件一致。
