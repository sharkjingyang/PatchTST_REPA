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
parser.add_argument('--projector_dim', type=int, default=768)

args = parser.parse_args([])

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create model
model = PatchTST.Model(args).float().to(device)
model.eval()

print(f"\nModel use_projector: {model.use_projector}")

# Create fake batch data
batch_size = 32
batch_x = torch.randn(batch_size, args.seq_len, args.enc_in).to(device)  # (32, 336, 7)
batch_y = torch.randn(batch_size, args.seq_len + args.pred_len, args.enc_in).to(device)  # (32, 432, 7)

print("\n" + "=" * 60)
print("Input Shapes:")
print("=" * 60)
print(f"batch_x:     {batch_x.shape}")
print(f"batch_y:     {batch_y.shape}")

# Forward pass with target (for TiViT extraction)
with torch.no_grad():
    outputs, zs, zs_tilde = model(batch_x, batch_y)

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

# zs shape: (bs, projector_dim, nvars) -> need to permute to (bs, nvars, projector_dim)
# zs_tilde shape: (bs, nvars, d_vit) -> (32, 7, 768)

# Permute zs to match zs_tilde: (bs, d, nvars) -> (bs, nvars, d)
zs = zs.permute(0, 2, 1)  # (32, 7, 768)

print(f"zs shape (after permute): {zs.shape}")
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
f_dim = -1  # features='M'
batch_y_pred = batch_y[:, -args.pred_len:, f_dim:].to(device)  # (32, 96, 7)
outputs_pred = outputs[:, -args.pred_len:, :]  # (32, 96, 7)

mse_loss = criterion(outputs_pred, batch_y_pred)
print(f"mse_loss:         {mse_loss.item():.6f}")

# Combined loss
lambda_contrastive = 0.5
total_loss = mse_loss + lambda_contrastive * contrastive_loss
print(f"\nlambda_contrastive: {lambda_contrastive}")
print(f"total_loss (MSE + lambda * contrastive): {total_loss.item():.6f}")

print("\n" + "=" * 60)
print("Summary:")
print("=" * 60)
print(f"outputs:      {outputs.shape}  -> (bs, pred_len, nvars)")
print(f"zs:           {zs.shape}       -> (bs, nvars, d_model)")
print(f"zs_tilde:     {zs_tilde.shape} -> (bs, nvars, d_vit)")
print(f"projector_dim: {args.projector_dim}")
print("=" * 60)
