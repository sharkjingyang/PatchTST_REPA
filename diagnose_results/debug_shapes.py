import torch
import torch.nn.functional as F
import argparse
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import PatchTST

# Parse args
parser = argparse.ArgumentParser()

# Model args
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
parser.add_argument('--stride', type=int, default=8)
parser.add_argument('--padding_patch', type=str, default='end')
parser.add_argument('--revin', type=int, default=1)
parser.add_argument('--affine', type=int, default=0)
parser.add_argument('--subtract_last', type=int, default=0)
parser.add_argument('--decomposition', type=int, default=0)
parser.add_argument('--kernel_size', type=int, default=25)
parser.add_argument('--individual', type=int, default=0)
parser.add_argument('--encoder_depth', type=int, default=2)
parser.add_argument('--use_projector', type=int, default=1)
parser.add_argument('--feature_extractor', type=str, default='mantis')
parser.add_argument('--projector_dim', type=int, default=768)

args = parser.parse_args([])

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create model
model = PatchTST.Model(args).float().to(device)
model.eval()

print(f"\nModel has TiViT: {model.tivit is not None}")

# Create fake batch data
batch_size = 32
batch_x = torch.randn(batch_size, args.seq_len, args.enc_in).to(device)  # (32, 336, 7)
batch_y = torch.randn(batch_size, args.seq_len + args.pred_len, args.enc_in).to(device)  # (32, 432, 7)

print("\n" + "=" * 60)
print("Input Shapes:")
print("=" * 60)
print(f"batch_x:     {batch_x.shape}")
print(f"batch_y:     {batch_y.shape}")

# Forward pass with target (for TiViT extraction, training mode)
with torch.no_grad():
    outputs, zs, zs_tilde = model(batch_x, batch_y, return_projector=True)

print("\n" + "=" * 60)
print("Output Shapes:")
print("=" * 60)
print(f"outputs:      {outputs.shape}")
print(f"zs:          {zs.shape}")
print(f"zs_tilde:    {zs_tilde.shape}")

# Test contrastive loss computation
print("\n" + "=" * 60)
print("Contrastive Loss Computation:")
print("=" * 60)

# zs and zs_tilde are now both (bs, nvars, d) - no permute needed
# zs shape: (bs, nvars, d)
# zs_tilde shape: (bs, nvars, d_vit)

print(f"zs shape: {zs.shape}")
print(f"zs_tilde shape:           {zs_tilde.shape}")

# Normalize features
zs_norm = F.normalize(zs, dim=-1)
zs_tilde_norm = F.normalize(zs_tilde, dim=-1)

# Compute cosine similarity per nvar
similarity = (zs_norm * zs_tilde_norm).sum(dim=-1)  # (bs, nvars)
print(f"similarity shape:  {similarity.shape}")
print(f"similarity min:   {similarity.min().item():.4f}")
print(f"similarity max:   {similarity.max().item():.4f}")
print(f"similarity mean:  {similarity.mean().item():.4f}")

# Contrastive loss (sum then normalize)
contrastive_loss = -similarity.sum() / similarity.numel()
print(f"\ncontrastive_loss: {contrastive_loss.item():.6f}")

# MSE loss
criterion = torch.nn.MSELoss()
f_dim = 0  # features='M' - take all channels
batch_y_pred = batch_y[:, -args.pred_len:, f_dim:].to(device)  # (32, 96, 7)
outputs_pred = outputs[:, -args.pred_len:, :]  # (32, 96, 7)

mse_loss = criterion(outputs_pred, batch_y_pred)
print(f"mse_loss:         {mse_loss.item():.6f}")

# Combined loss
lambda_contrastive = 0.5
total_loss = mse_loss + lambda_contrastive * contrastive_loss
print(f"\nlambda_contrastive: {lambda_contrastive}")
print(f"total_loss (MSE + lambda * contrastive): {total_loss.item():.6f}")

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

# Total model parameters (excluding TiViT)
# Calculate all params, then subtract TiViT params
all_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_non_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)
all_total = all_trainable + all_non_trainable

# Get TiViT params
tivit_total = 0
tivit_non_trainable = 0
if hasattr(model, 'tivit') and model.tivit is not None:
    tivit_total = sum(p.numel() for p in model.tivit.parameters())
    tivit_non_trainable = sum(p.numel() for p in model.tivit.parameters() if not p.requires_grad)

# Exclude TiViT from total
total = all_total - tivit_total
trainable = all_trainable  # TiViT is non-trainable
non_trainable = all_non_trainable - tivit_non_trainable

print(f"Total parameters (excl. TiViT): {total:,}")
print(f"Trainable parameters:            {trainable:,}")
print(f"Non-trainable parameters:        {non_trainable:,}")

# Projector parameters (if exists)
if hasattr(model, 'model') and hasattr(model.model, 'projector'):
    projector = model.model.projector
    proj_trainable, proj_non_trainable, proj_total = count_parameters(projector)
    print(f"\nProjector parameters:")
    print(f"  Total:         {proj_total:,}")
    print(f"  Trainable:    {proj_trainable:,}")
    print(f"  Non-trainable: {proj_non_trainable:,}")
elif hasattr(model, 'model_trend') and hasattr(model.model_trend, 'projector'):
    # decomposition mode has two projectors
    projector_trend = model.model_trend.projector
    projector_res = model.model_res.projector
    proj_trainable, proj_non_trainable, proj_total = count_parameters(projector_trend)
    proj_total = proj_total * 2  # two projectors
    print(f"\nProjector parameters (2 projectors):")
    print(f"  Total:         {proj_total:,}")
    print(f"  Trainable:    {proj_trainable * 2:,}")
    print(f"  Non-trainable: {proj_non_trainable * 2:,}")
else:
    print("\nNo projector found in model.")

# TiViT parameters (frozen)
if hasattr(model, 'tivit') and model.tivit is not None:
    tivit_trainable, tivit_non_trainable, tivit_total = count_parameters(model.tivit)
    print(f"\nTiViT parameters (frozen, excluded from total):")
    print(f"  Total:         {tivit_total:,}")
    print(f"  Non-trainable: {tivit_non_trainable:,}")

print("\n" + "=" * 60)
print("Summary:")
print("=" * 60)
print(f"outputs:      {outputs.shape}  -> (bs, pred_len, nvars)")
print(f"zs:           {zs.shape}       -> (bs, nvars, d_model)")
print(f"zs_tilde:     {zs_tilde.shape} -> (bs, nvars, feature_dim)")
print(f"feature_dim (zs): {zs.shape[-1]}")
print("=" * 60)
