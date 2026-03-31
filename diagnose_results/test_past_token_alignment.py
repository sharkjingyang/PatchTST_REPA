"""
Diagnose: PatchTST_REPA with Chronos2 past token alignment
Check shape consistency between zs (PatchTST encoder) and zs_tilde (Chronos2 past tokens)
"""
import torch
import argparse
import sys
import os
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock layers.Tivit before importing project modules (open_clip/torchvision version incompatibility workaround)
tivit_mock = MagicMock()
tivit_mock.get_tivit = MagicMock(return_value=MagicMock())
tivit_mock.get_patch_size = MagicMock(return_value=16)
sys.modules['layers.Tivit'] = tivit_mock

from models import PatchTST

parser = argparse.ArgumentParser()
parser.add_argument('--seq_len', type=int, default=336)
parser.add_argument('--pred_len', type=int, default=720)
parser.add_argument('--enc_in', type=int, default=7)
parser.add_argument('--dec_in', type=int, default=7)
parser.add_argument('--c_out', type=int, default=7)
parser.add_argument('--e_layers', type=int, default=3)
parser.add_argument('--encoder_depth', type=int, default=3)
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
parser.add_argument('--feature_extractor', type=str, default='chronos')
parser.add_argument('--projector_dim', type=int, default=768)
parser.add_argument('--chronos_pretrained', type=str, default='./Chronos2')
parser.add_argument('--head_type', type=str, default='flatten')
parser.add_argument('--num_quantiles', type=int, default=20)
parser.add_argument('--contrastive_type', type=str, default='patch_wise')
parser.add_argument('--contrastive', type=int, default=1)
parser.add_argument('--output_patch_size', type=int, default=16)
parser.add_argument('--patch_fusion_n_heads', type=int, default=4)
parser.add_argument('--d_layers', type=int, default=0)
parser.add_argument('--patch_fusion_type', type=str, default='none')
parser.add_argument('--model', type=str, default='PatchTST_REPA')
args = parser.parse_args([])

bs, nvars, seq_len, pred_len = 2, 7, 336, 720
expected_patch_num = seq_len // 16  # = 21

print("=" * 60)
print("Test: PatchTST_REPA + Chronos2 past token alignment")
print(f"  seq_len={seq_len}, pred_len={pred_len}, patch_len=16, stride=16")
print(f"  expected patch_num = {expected_patch_num}")
print("=" * 60)

print("\nBuilding model...")
model = PatchTST.Model(args)
model.eval()

batch_x = torch.randn(bs, seq_len, nvars)
batch_y = torch.randn(bs, pred_len, nvars)

print("\nRunning forward pass...")
with torch.no_grad():
    output, zs, zs_tilde = model(batch_x, batch_y, return_projector=True)

print(f"\nOutput shape:    {output.shape}   (expected: [{bs}, {pred_len}, {nvars}])")
print(f"zs shape:        {zs.shape}   (PatchTST encoder projected)")
print(f"zs_tilde shape:  {zs_tilde.shape}   (Chronos2 past tokens)")

# Shape checks
assert output.shape == (bs, pred_len, nvars), f"Output shape mismatch: {output.shape}"
assert zs.shape[0] == bs and zs.shape[1] == nvars, f"zs batch/var dim mismatch: {zs.shape}"
assert zs_tilde.shape[0] == bs and zs_tilde.shape[1] == nvars, f"zs_tilde batch/var dim mismatch: {zs_tilde.shape}"

patch_num_zs = zs.shape[2]
patch_num_tilde = zs_tilde.shape[2]
d_zs = zs.shape[3]
d_tilde = zs_tilde.shape[3]

print(f"\nPatch count:  zs={patch_num_zs}, zs_tilde={patch_num_tilde}  ", end="")
print("✓ MATCH" if patch_num_zs == patch_num_tilde else f"✗ MISMATCH (need same for patch_wise)")

print(f"Feature dim:  zs={d_zs}, zs_tilde={d_tilde}  ", end="")
print("✓ MATCH" if d_zs == d_tilde else f"✗ MISMATCH")

assert patch_num_zs == patch_num_tilde, "patch_wise alignment requires equal patch counts"
assert patch_num_zs == expected_patch_num, f"patch_num={patch_num_zs}, expected {expected_patch_num}"
assert d_zs == d_tilde == 768, f"d_extractor mismatch: zs={d_zs}, tilde={d_tilde}"

print("\n" + "=" * 60)
print("All shape checks passed!")
print("=" * 60)
