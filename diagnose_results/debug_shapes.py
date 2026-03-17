import torch
import torch.nn.functional as F
import argparse
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import PatchTST

# Parse args - matching Chronos2.sh configuration
parser = argparse.ArgumentParser()

# Model args - Chronos2.sh defaults
parser.add_argument('--seq_len', type=int, default=336)
parser.add_argument('--pred_len', type=int, default=96)
parser.add_argument('--enc_in', type=int, default=7)
parser.add_argument('--dec_in', type=int, default=7)
parser.add_argument('--c_out', type=int, default=7)
parser.add_argument('--e_layers', type=int, default=4)
parser.add_argument('--n_heads', type=int, default=4)
parser.add_argument('--d_model', type=int, default=16)
parser.add_argument('--d_ff', type=int, default=128)
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--fc_dropout', type=float, default=0.3)
parser.add_argument('--head_dropout', type=float, default=0.0)
parser.add_argument('--patch_len', type=int, default=16)
parser.add_argument('--stride', type=int, default=16)  # Chronos2.sh: stride=16
parser.add_argument('--padding_patch', type=str, default=None)  # Chronos2.sh: padding_patch=None
parser.add_argument('--revin', type=int, default=1)
parser.add_argument('--affine', type=int, default=0)
parser.add_argument('--subtract_last', type=int, default=0)
parser.add_argument('--decomposition', type=int, default=0)
parser.add_argument('--kernel_size', type=int, default=25)
parser.add_argument('--individual', type=int, default=0)
parser.add_argument('--encoder_depth', type=int, default=2)
parser.add_argument('--feature_extractor', type=str, default='chronos', choices=['tivit', 'mantis', 'chronos'])
parser.add_argument('--projector_dim', type=int, default=768)
parser.add_argument('--chronos_pretrained', type=str, default='./Chronos2')
parser.add_argument('--head_type', type=str, default='quantile', choices=['flatten', 'quantile'])  # Chronos2.sh: head_type=quantile
parser.add_argument('--num_quantiles', type=int, default=20)
parser.add_argument('--contrastive_type', type=str, default='patch_wise', choices=['mean_pool', 'patch_wise'])
parser.add_argument('--use_projector', type=int, default=1)  # Enable projector for PatchTST_REPA
parser.add_argument('--lambda_contrastive', type=float, default=0.5)

args = parser.parse_args()

# Set model name to PatchTST_REPA
args.model = 'PatchTST_REPA'

# Compute patch_num
patch_num = int((args.seq_len - args.patch_len) / args.stride + 1) if args.padding_patch is None else int((args.seq_len - args.patch_len) / args.stride + 1) + 1
output_patch_num = args.pred_len // 16  # output_patch_size = 16

print("=" * 60)
print("Chronos2.sh Configuration:")
print("=" * 60)
print(f"seq_len:       {args.seq_len}")
print(f"pred_len:      {args.pred_len}")
print(f"patch_len:     {args.patch_len}")
print(f"stride:        {args.stride}")
print(f"padding_patch: {args.padding_patch}")
print(f"head_type:     {args.head_type}")
print(f"num_quantiles: {args.num_quantiles}")
print(f"feature_extractor: {args.feature_extractor}")
print(f"contrastive_type: {args.contrastive_type}")
print(f"patch_num (input):  {patch_num}")
print(f"output_patch_num:  {output_patch_num}")
print("=" * 60)

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

# Create model
model = PatchTST.Model(args).float().to(device)
model.eval()

print(f"\nFeature extractor: {args.feature_extractor}")
print(f"Model has TiViT: {model.tivit is not None}")
print(f"Model has Mantis: {model.mantis is not None}")
print(f"Model has Chronos: {model.chronos is not None}")

# Create fake batch data
batch_size = 32
batch_x = torch.randn(batch_size, args.seq_len, args.enc_in).to(device)  # (32, 336, 7)
batch_y = torch.randn(batch_size, args.seq_len + args.pred_len, args.enc_in).to(device)  # (32, 432, 7)

print("\n" + "=" * 60)
print("Input Shapes:")
print("=" * 60)
print(f"batch_x:     {batch_x.shape}  # (bs, seq_len, nvars)")
print(f"batch_y:     {batch_y.shape}  # (bs, seq_len+pred_len, nvars)")

# ============================================================
# Step 1: PatchTST_backbone forward (inside model)
# ============================================================
print("\n" + "=" * 60)
print("Step 1: PatchTST_backbone forward (inside model)")
print("=" * 60)

# The flow inside PatchTST_backbone:
# 1. Input: (bs, nvars, seq_len) = (32, 7, 336)
# 2. Patching: unfold -> (bs, nvars, patch_num, patch_len) = (32, 7, 21, 16)
# 3. Permute: (bs, nvars, patch_len, patch_num) = (32, 7, 16, 21)
# 4. W_P: Linear(patch_len -> d_model) -> (bs, nvars, patch_num, d_model) = (32, 7, 21, 16)
# 5. Reshape: (bs*nvars, patch_num, d_model) = (224, 21, 16)
# 6. Transformer Encoder: (224, 21, 16) -> (224, 21, 16)
# 7. Reshape back: (32, 7, 16, 21)
# 8. MLP Projector: (32, 7, 21, 16) -> (32, 7, 21, 768)

print(f"\n[Inside PatchTST_backbone]")
print(f"Input (after permute):           (bs, nvars, seq_len) = ({batch_size}, {args.enc_in}, {args.seq_len})")
print(f"After patching (unfold):         (bs, nvars, patch_num, patch_len) = ({batch_size}, {args.enc_in}, {patch_num}, {args.patch_len})")
print(f"After W_P (Linear):              (bs, nvars, patch_num, d_model) = ({batch_size}, {args.enc_in}, {patch_num}, {args.d_model})")
print(f"After Transformer:               (bs, nvars, d_model, patch_num) = ({batch_size}, {args.enc_in}, {args.d_model}, {patch_num})")
print(f"After MLP Projector (zs):        (bs, nvars, patch_num, projector_dim) = ({batch_size}, {args.enc_in}, {patch_num}, {args.projector_dim})")

# ============================================================
# Step 2: Head forward
# ============================================================
print("\n" + "=" * 60)
print("Step 2: Head forward")
print("=" * 60)

if args.head_type == 'quantile':
    # Quantile_Head:
    # Input: (bs, nvars, d_model, input_patch_num) = (32, 7, 16, 21)
    # Flatten: (bs, nvars, d_model * input_patch_num) = (32, 7, 336)
    # ResidualBlock: 336 -> 336 -> (6 * 20 * 16) = (32, 7, 1920)
    # Rearrange: (bs, nvars, num_quantiles, pred_len) = (32, 7, 20, 96)

    print(f"\n[Quantile_Head]")
    print(f"Input to head:                   (bs, nvars, d_model, patch_num) = ({batch_size}, {args.enc_in}, {args.d_model}, {patch_num})")
    print(f"After flatten:                   (bs, nvars, d_model*patch_num) = ({batch_size}, {args.enc_in}, {args.d_model * patch_num})")
    print(f"After ResidualBlock:             (bs, nvars, output_patch_num*num_quantiles*16) = ({batch_size}, {args.enc_in}, {output_patch_num * args.num_quantiles * 16})")
    print(f"After rearrange:                  (bs, nvars, num_quantiles, pred_len) = ({batch_size}, {args.enc_in}, {args.num_quantiles}, {args.pred_len})")
    head_output_shape = f"({batch_size}, {args.enc_in}, {args.num_quantiles}, {args.pred_len})"
else:
    # Flatten_Head:
    # Input: (bs, nvars, d_model, input_patch_num) = (32, 7, 16, 21)
    # Flatten: (bs, nvars, d_model * input_patch_num) = (32, 7, 336)
    # Linear: (bs, nvars, pred_len) = (32, 7, 96)

    print(f"\n[Flatten_Head]")
    print(f"Input to head:                   (bs, nvars, d_model, patch_num) = ({batch_size}, {args.enc_in}, {args.d_model}, {patch_num})")
    print(f"After flatten:                   (bs, nvars, d_model*patch_num) = ({batch_size}, {args.enc_in}, {args.d_model * patch_num})")
    print(f"After Linear:                    (bs, nvars, pred_len) = ({batch_size}, {args.enc_in}, {args.pred_len})")
    head_output_shape = f"({batch_size}, {args.enc_in}, {args.pred_len})"

# ============================================================
# Step 3: Model forward (actual)
# ============================================================
print("\n" + "=" * 60)
print("Step 3: Model forward (actual)")
print("=" * 60)

# For training, we need to pass batch_y for feature extraction
# batch_y is already sliced to pred_len in exp_main.py, but here we pass full batch_y
# The model will use the last pred_len portion internally

# Interpolate batch_y to seq_len for Chronos patch_wise (only for PatchTST_REPA, not Fusion)
# PatchTST_REPA_Fusion uses Channel Fusion MLP which handles patch_num conversion automatically
if args.feature_extractor == 'chronos' and args.contrastive_type == 'patch_wise' and args.model != 'PatchTST_REPA_Fusion':
    batch_y_for_model = batch_y[:, -args.pred_len:, :]  # (bs, pred_len, nvars)
    batch_y_interp = F.interpolate(
        batch_y_for_model.permute(0, 2, 1),  # (bs, nvars, pred_len)
        size=args.seq_len,
        mode='linear',
        align_corners=False
    ).permute(0, 2, 1)  # (bs, seq_len, nvars)
    print(f"\n[Chronos patch_wise preprocessing (PatchTST_REPA)]")
    print(f"batch_y (original):             {batch_y[:, -args.pred_len:, :].shape}")
    print(f"batch_y (interpolated to seq_len): {batch_y_interp.shape}")

    # Forward with interpolated target
    target_input = batch_y_interp
elif args.feature_extractor == 'chronos' and args.model == 'PatchTST_REPA_Fusion':
    # PatchTST_REPA_Fusion: no interpolation needed, Channel Fusion handles it
    batch_y_for_model = batch_y[:, -args.pred_len:, :]  # (bs, pred_len, nvars)
    print(f"\n[Chronos patch_wise preprocessing (PatchTST_REPA_Fusion - no interpolation)]")
    print(f"batch_y (original):             {batch_y_for_model.shape}")
    target_input = batch_y_for_model
else:
    target_input = batch_y[:, -args.pred_len:, :]  # (bs, pred_len, nvars)

# Forward pass
with torch.no_grad():
    outputs, zs, zs_tilde = model(batch_x, target_input, return_projector=True)

print(f"\n[Model Output]")
print(f"outputs:      {outputs.shape}")
print(f"zs:          {zs.shape}")
print(f"zs_tilde:    {zs_tilde.shape}")

# ============================================================
# Step 4: Output transformation
# ============================================================
print("\n" + "=" * 60)
print("Step 4: Output transformation")
print("=" * 60)

if args.head_type == 'quantile':
    # After model forward, outputs are permuted to (bs, pred_len, nvars, num_quantiles)
    print(f"\n[Quantile Head Output]")
    print(f"Model output (permute):          (bs, pred_len, nvars, num_quantiles) = {outputs.shape}")

    # For loss computation, we permute to (bs, nvars, num_quantiles, pred_len)
    outputs_for_loss = outputs.permute(0, 2, 3, 1)
    print(f"For QuantileLoss (permute):     (bs, nvars, num_quantiles, pred_len) = {outputs_for_loss.shape}")
else:
    print(f"\n[Flatten Head Output]")
    print(f"Model output:                    (bs, pred_len, nvars) = {outputs.shape}")

# ============================================================
# Step 5: Contrastive Loss
# ============================================================
print("\n" + "=" * 60)
print("Step 5: Contrastive Loss")
print("=" * 60)

print(f"\n[Contrastive Loss Input]")
print(f"zs (projected):                   {zs.shape}")
print(f"zs_tilde (feature extractor):     {zs_tilde.shape}")

if args.contrastive_type == 'mean_pool':
    # Mean pool over patch_num
    zs_pooled = zs.mean(dim=2)  # (bs, nvars, projector_dim)
    zs_tilde_pooled = zs_tilde.mean(dim=2)  # (bs, nvars, feature_dim)
    print(f"\n[Mean Pooling]")
    print(f"zs_pooled:                       {zs_pooled.shape}")
    print(f"zs_tilde_pooled:                 {zs_tilde_pooled.shape}")

    # Normalize
    zs_norm = F.normalize(zs_pooled, dim=-1)
    zs_tilde_norm = F.normalize(zs_tilde_pooled, dim=-1)

    # Cosine similarity
    similarity = (zs_norm * zs_tilde_norm).sum(dim=-1)  # (bs, nvars)
    print(f"similarity:                      {similarity.shape}")
else:
    # Patch-wise: keep all patches
    print(f"\n[Patch-wise (no pooling)]")
    # zs: (bs, nvars, patch_num, projector_dim)
    # zs_tilde: (bs, nvars, patch_num, feature_dim)

    # Normalize
    zs_norm = F.normalize(zs, dim=-1)
    zs_tilde_norm = F.normalize(zs_tilde, dim=-1)

    # Cosine similarity per patch
    similarity = (zs_norm * zs_tilde_norm).sum(dim=-1)  # (bs, nvars, patch_num)
    print(f"similarity (per patch):          {similarity.shape}")

print(f"similarity min:   {similarity.min().item():.4f}")
print(f"similarity max:   {similarity.max().item():.4f}")
print(f"similarity mean:  {similarity.mean().item():.4f}")

# Contrastive loss
contrastive_loss = -similarity.sum() / similarity.numel()
print(f"\ncontrastive_loss: {contrastive_loss.item():.6f}")

# ============================================================
# Step 6: Quantile Loss
# ============================================================
if args.head_type == 'quantile':
    print("\n" + "=" * 60)
    print("Step 6: Quantile Loss")
    print("=" * 60)

    # Import QuantileLoss
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from exp.exp_main import QuantileLoss

    criterion = QuantileLoss(num_quantiles=args.num_quantiles)

    # Prepare target
    batch_y_pred = batch_y[:, -args.pred_len:, :]  # (bs, pred_len, nvars)

    # outputs_for_loss: (bs, nvars, num_quantiles, pred_len)
    # batch_y_pred: (bs, pred_len, nvars)
    print(f"\n[Quantile Loss]")
    print(f"pred (outputs_for_loss):         {outputs_for_loss.shape}")
    print(f"target:                          {batch_y_pred.shape}")

    quantile_loss = criterion(outputs_for_loss, batch_y_pred)
    print(f"quantile_loss:  {quantile_loss.item():.6f}")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("Summary (Chronos2.sh Configuration)")
print("=" * 60)
print(f"Configuration:")
print(f"  seq_len={args.seq_len}, pred_len={args.pred_len}")
print(f"  patch_len={args.patch_len}, stride={args.stride}")
print(f"  padding_patch={args.padding_patch}")
print(f"  head_type={args.head_type}, num_quantiles={args.num_quantiles}")
print(f"  feature_extractor={args.feature_extractor}, contrastive_type={args.contrastive_type}")

print(f"\nKey Shapes:")
print(f"  Input batch_x:           (bs, seq_len, nvars)         = ({batch_size}, {args.seq_len}, {args.enc_in})")
print(f"  patch_num:               {patch_num}")
print(f"  zs (projected):         (bs, nvars, patch_num, pd) = {zs.shape}")
print(f"  zs_tilde (Chronos):      (bs, nvars, patch_num, 768) = {zs_tilde.shape}")

if args.head_type == 'quantile':
    print(f"  outputs (model):         (bs, pred_len, nvars, q)   = {outputs.shape}")
    print(f"  outputs_for_loss:        (bs, nvars, q, pred_len)   = {outputs_for_loss.shape}")
else:
    print(f"  outputs (model):         (bs, pred_len, nvars)       = {outputs.shape}")

print("=" * 60)

# Parameter statistics
print("\n" + "=" * 60)
print("Parameter Statistics:")
print("=" * 60)

def count_parameters(model):
    """Count trainable and non-trainable parameters"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total = trainable + non_trainable
    return trainable, non_trainable, total

# Total model parameters
all_total = sum(p.numel() for p in model.parameters())
all_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

# Detect which feature extractor is created
feature_extractor = None
fe_total = 0

if hasattr(model, 'tivit') and model.tivit is not None:
    feature_extractor = 'TiViT'
    fe_total = sum(p.numel() for p in model.tivit.parameters())
elif hasattr(model, 'mantis') and model.mantis is not None:
    feature_extractor = 'Mantis'
    fe_total = sum(p.numel() for p in model.mantis_network.parameters())
elif hasattr(model, 'chronos') and model.chronos is not None:
    feature_extractor = 'Chronos'
    fe_total = sum(p.numel() for p in model.chronos.model.parameters())

total_excl = all_total - fe_total

# Show model configuration
head_type = type(model.model.head).__name__
use_channel_fusion = model.model.use_channel_fusion

print(f"\nModel Configuration:")
print(f"  Model:         {args.model}")
print(f"  Head type:     {head_type}")
print(f"  Channel fusion: {use_channel_fusion}")
print(f"\nTotal parameters (all):              {all_total:,}")
if feature_extractor:
    print(f"Total parameters (excl. {feature_extractor}): {total_excl:,}")
print(f"Trainable parameters:                {all_trainable:,}")

# Projector parameters (if exists)
if hasattr(model, 'model') and hasattr(model.model, 'projector'):
    proj_trainable, proj_non_trainable, proj_total = count_parameters(model.model.projector)
    print(f"\nProjector parameters:")
    print(f"  Total:         {proj_total:,}")
elif hasattr(model, 'model_trend') and hasattr(model.model_trend, 'projector'):
    proj_trainable, proj_non_trainable, proj_total = count_parameters(model.model_trend.projector)
    print(f"\nProjector parameters (2 projectors):")
    print(f"  Total:         {proj_total * 2:,}")
else:
    print("\nNo projector found in model.")

# Head parameters
if hasattr(model, 'model') and hasattr(model.model, 'head'):
    head_total = sum(p.numel() for p in model.model.head.parameters())
    print(f"\nHead parameters ({head_type}):")
    print(f"  Total:         {head_total:,}")
