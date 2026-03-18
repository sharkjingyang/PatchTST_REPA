"""
Compare two PatchTST implementations:
1. Original PatchTST (PatchTST-main/PatchTST_supervised)
2. Our modified version with contrastive=0

This script verifies that both implementations produce identical results.
"""

import torch
import torch.nn as nn
import os
import sys
import argparse

# Set random seed for reproducibility
torch.manual_seed(2021)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Device
device = torch.device('cpu')  # Use CPU for reproducibility
print(f"Using device: {device}")

# ============================================================
# Test Configuration (same as original PatchTST etth1.sh)
# ============================================================
seq_len = 336
pred_len = 720
enc_in = 7
e_layers = 4
n_heads = 4
d_model = 16
d_ff = 128
dropout = 0.0  # Set dropout to 0 for deterministic results
fc_dropout = 0.0
head_dropout = 0.0
patch_len = 16
stride = 8

batch_size = 32

# ============================================================
# Load Original PatchTST Model
# ============================================================
print("\n" + "=" * 60)
print("Loading Original PatchTST from PatchTST-main")
print("=" * 60)

# Add original path to sys.path
original_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'PatchTST-main', 'PatchTST_supervised'))
sys.path.insert(0, original_path)

# Import original model
from models.PatchTST import Model as Model_Original

# Create original model args
class ArgsOriginal:
    def __init__(self):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.dec_in = enc_in
        self.c_out = enc_in
        self.e_layers = e_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.fc_dropout = fc_dropout
        self.head_dropout = head_dropout
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = 'end'
        self.revin = 1
        self.affine = 0
        self.subtract_last = 0
        self.decomposition = 0
        self.kernel_size = 25
        self.individual = 0
        self.encoder_depth = 2

args_orig = ArgsOriginal()

torch.manual_seed(2021)
model_orig = Model_Original(args_orig).float().to(device)
model_orig.eval()

# Count parameters
orig_params = sum(p.numel() for p in model_orig.parameters())
print(f"Original model parameters: {orig_params:,}")


# ============================================================
# Load Our Modified PatchTST (contrastive=0)
# ============================================================
print("\n" + "=" * 60)
print("Loading Modified PatchTST (contrastive=0)")
print("=" * 60)

# Remove original path to avoid import conflicts
if original_path in sys.path:
    sys.path.remove(original_path)

# Import our model
from models.PatchTST import Model as Model_Ours

# Create our model args
class ArgsOurs:
    def __init__(self):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.dec_in = enc_in
        self.c_out = enc_in
        self.e_layers = e_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.fc_dropout = fc_dropout
        self.head_dropout = head_dropout
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = 'end'
        self.revin = 1
        self.affine = 0
        self.subtract_last = 0
        self.decomposition = 0
        self.kernel_size = 25
        self.individual = 0
        self.encoder_depth = 2
        self.contrastive = 0
        self.projector_dim = 768

args_ours = ArgsOurs()

torch.manual_seed(2021)
model_ours = Model_Ours(args_ours).float().to(device)
model_ours.eval()

# Count parameters
our_params = sum(p.numel() for p in model_ours.parameters())
print(f"Our model parameters: {our_params:,}")

if orig_params == our_params:
    print("✓ Parameter count matches!")
else:
    print(f"✗ Parameter count mismatch: {orig_params} vs {our_params}")


# ============================================================
# Create Identical Input
# ============================================================
print("\n" + "=" * 60)
print("Creating Identical Input")
print("=" * 60)

torch.manual_seed(2021)
batch_x = torch.randn(batch_size, seq_len, enc_in).to(device)  # (bs, seq_len, nvars)
batch_y = torch.randn(batch_size, seq_len + pred_len, enc_in).to(device)  # (bs, seq_len+pred_len, nvars)

# Model expects (bs, seq_len, nvars) - no need to permute
print(f"batch_x shape: {batch_x.shape}")
print(f"batch_y shape: {batch_y.shape}")


# ============================================================
# Forward Pass Comparison
# ============================================================
print("\n" + "=" * 60)
print("Forward Pass Comparison")
print("=" * 60)

# Make sure both are in eval mode
model_orig.eval()
model_ours.eval()

with torch.no_grad():
    output_orig = model_orig(batch_x)  # returns (bs, pred_len, nvars)
    output_ours = model_ours(batch_x)  # returns (bs, pred_len, nvars)

print(f"Original output shape: {output_orig.shape}")
print(f"Our output shape: {output_ours.shape}")

# Compare outputs
diff = torch.abs(output_orig - output_ours)
max_diff = diff.max().item()
mean_diff = diff.mean().item()

print(f"\nOutput difference:")
print(f"  Max diff:  {max_diff:.10f}")
print(f"  Mean diff: {mean_diff:.10f}")

if max_diff < 1e-6:
    print("✓ Forward pass outputs MATCH!")
else:
    print("✗ Forward pass outputs DIFFER!")


# ============================================================
# Gradient Comparison (Backward Pass)
# ============================================================
print("\n" + "=" * 60)
print("Gradient Comparison (Backward Pass)")
print("=" * 60)

# Create new input for backward with requires_grad
torch.manual_seed(2021)
batch_x_test = torch.randn(batch_size, seq_len, enc_in, requires_grad=True).to(device)  # (bs, seq_len, nvars)
batch_y_test = torch.randn(batch_size, pred_len, enc_in).to(device)  # (bs, pred_len, nvars)

# Make sure both models are in train mode
model_orig.train()
model_ours.train()

# Original model backward
output_orig_train = model_orig(batch_x_test)  # returns (bs, pred_len, nvars)

criterion = nn.MSELoss()
loss_orig = criterion(output_orig_train, batch_y_test)
loss_orig.backward()

# Get gradients from original model
orig_grads = {}
for name, param in model_orig.named_parameters():
    if param.grad is not None:
        orig_grads[name] = param.grad.clone()
    else:
        orig_grads[name] = None

print("Original model backward completed.")

# Our model backward
output_ours_train = model_ours(batch_x_test)  # returns (bs, pred_len, nvars)

loss_ours = criterion(output_ours_train, batch_y_test)
loss_ours.backward()

# Get gradients from our model
ours_grads = {}
for name, param in model_ours.named_parameters():
    if param.grad is not None:
        ours_grads[name] = param.grad.clone()
    else:
        ours_grads[name] = None

print("Our model backward completed.")

# Compare gradients
print("\nGradient comparison:")
print("-" * 60)

all_match = True

# Check common parameters
common_params = set(orig_grads.keys()) & set(ours_grads.keys())
for name in sorted(common_params):
    grad_orig = orig_grads[name]
    grad_ours = ours_grads[name]

    if grad_orig is None and grad_ours is None:
        print(f"  {name}: both have no gradient")
        continue

    if grad_orig is None or grad_ours is None:
        print(f"✗ {name}: one has gradient, other doesn't")
        all_match = False
        continue

    diff = torch.abs(grad_orig - grad_ours)
    max_diff_grad = diff.max().item()
    mean_diff_grad = diff.mean().item()

    if max_diff_grad < 1e-6:
        status = "✓"
    else:
        status = "✗"
        all_match = False

    print(f"{status} {name}: max_diff={max_diff_grad:.10f}")

# Check parameters only in original
for name in sorted(set(orig_grads.keys()) - set(ours_grads.keys())):
    print(f"  {name}: exists in original but not in ours")
    all_match = False

# Check parameters only in ours
for name in sorted(set(ours_grads.keys()) - set(orig_grads.keys())):
    print(f"  {name}: exists in ours but not in original")
    all_match = False

print("-" * 60)
if all_match:
    print("✓ ALL GRADIENTS MATCH!")
else:
    print("✗ SOME GRADIENTS DIFFER!")

# Compare loss values
print(f"\nLoss comparison:")
print(f"  Original loss: {loss_orig.item():.10f}")
print(f"  Our loss:      {loss_ours.item():.10f}")
print(f"  Loss diff:    {abs(loss_orig.item() - loss_ours.item()):.10f}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Forward pass max difference: {max_diff:.10f}")
print(f"Gradients match: {all_match}")
print("=" * 60)
