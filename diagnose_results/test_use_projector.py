import torch
import argparse

# Add parent directory to path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import PatchTST

# Parse args
parser = argparse.ArgumentParser()
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
parser.add_argument('--stride', type=int, default=16)
parser.add_argument('--padding_patch', type=str, default=None)
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
parser.add_argument('--head_type', type=str, default='patch_wise', choices=['flatten', 'quantile', 'patch_wise'])
parser.add_argument('--num_quantiles', type=int, default=20)
parser.add_argument('--contrastive_type', type=str, default='patch_wise', choices=['mean_pool', 'patch_wise'])
parser.add_argument('--contrastive', type=int, default=None)
parser.add_argument('--output_patch_size', type=int, default=16)
parser.add_argument('--channel_fusion_n_heads', type=int, default=4)
parser.add_argument('--d_layers', type=int, default=1)
parser.add_argument('--patch_fusion_type', type=str, default='split_MLP', choices=['fusion_MLP', 'split_MLP'])

args = parser.parse_args([])

# Test 1: Default PatchTST_REPA_Fusion (contrastive=1, channel_fusion always on)
print("=" * 60)
print("Test 1: Default PatchTST_REPA_Fusion (contrastive=1)")
print("=" * 60)
args.model = 'PatchTST_REPA_Fusion'
args.contrastive = None
model1 = PatchTST.Model(args)
print(f"contrastive: {model1.contrastive}")
print(f"use_patch_fusion: {model1.use_patch_fusion}")

# Test 2: PatchTST_REPA_Fusion with contrastive=0
print("\n" + "=" * 60)
print("Test 2: PatchTST_REPA_Fusion with contrastive=0")
print("=" * 60)
args.contrastive = 0
model2 = PatchTST.Model(args)
print(f"contrastive: {model2.contrastive}")
print(f"use_patch_fusion: {model2.use_patch_fusion}")

# Test 3: Forward pass with contrastive=0 (no contrastive loss)
print("\n" + "=" * 60)
print("Test 3: Forward pass with contrastive=0")
print("=" * 60)
args.contrastive = 0
model3 = PatchTST.Model(args)
batch_x = torch.randn(4, 336, 7)
output = model3(batch_x)
print(f"Output shape: {output.shape}")

# Test 4: Forward pass with contrastive=1 (with contrastive loss)
print("\n" + "=" * 60)
print("Test 4: Forward pass with contrastive=1")
print("=" * 60)
args.contrastive = 1
model4 = PatchTST.Model(args)
batch_x = torch.randn(4, 336, 7)
batch_y = torch.randn(4, 96, 7)  # target for zs_tilde extraction
output, zs, zs_tilde = model4(batch_x, batch_y, return_projector=True)
print(f"Output shape: {output.shape}")
print(f"Zs shape: {zs.shape}")
print(f"Zs_tilde shape: {zs_tilde.shape}")

print("\n" + "=" * 60)
print("All tests passed!")
print("=" * 60)
