import torch
import argparse
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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

# Create model
model = PatchTST.Model(args).float().to(device)
model.eval()

# Create fake batch data
batch_size = 32
batch_x = torch.randn(batch_size, args.seq_len, args.enc_in).to(device)  # (32, 336, 7)
batch_y = torch.randn(batch_size, args.seq_len + args.pred_len, args.enc_in).to(device)  # (32, 432, 7)

print("=" * 50)
print("Shapes:")
print("=" * 50)
print(f"batch_x:     {batch_x.shape}")
print(f"batch_y:     {batch_y.shape}")
print("=" * 50)

# Forward pass
with torch.no_grad():
    outputs, zs_project = model(batch_x)

print(f"outputs:     {outputs.shape}")
print(f"zs_project: {zs_project.shape}")
print("=" * 50)

# Simulate the training processing - features='M' means multi-channel
f_dim = 0  # for MS mode, take specific channel; for M mode, use -1
pred_len = args.pred_len

# When features='M', f_dim = -1 means take all channels
batch_y_pred_all = batch_y[:, -pred_len:, :]  # (32, 96, 7)
print(f"batch_y_pred (batch_y[:, -pred_len:, :]): {batch_y_pred_all.shape}")

# When f_dim = -1 in the original code (features='MS')
batch_y_pred_f = batch_y[:, -pred_len:, f_dim:]
print(f"batch_y_pred (batch_y[:, -pred_len:, f_dim=0]): {batch_y_pred_f.shape}")
print("=" * 50)

# After extracting pred_len
outputs_trimmed = outputs  # outputs already has shape (32, 96, 7)
print(f"outputs_trimmed: {outputs_trimmed.shape}")
print("=" * 50)
print(f"Expected patch_num = (seq_len - patch_len) / stride + 1 = ({args.seq_len} - {args.patch_len}) / {args.stride} + 1 = {(args.seq_len - args.patch_len) // args.stride + 1}")
