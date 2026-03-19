import torch
import argparse

# Add parent directory to path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.Chronos2_head import Model

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument('--seq_len', type=int, default=336)
parser.add_argument('--pred_len', type=int, default=96)
parser.add_argument('--enc_in', type=int, default=7)
parser.add_argument('--dec_in', type=int, default=7)
parser.add_argument('--c_out', type=int, default=7)
parser.add_argument('--patch_len', type=int, default=16)
parser.add_argument('--stride', type=int, default=8)
parser.add_argument('--head_dropout', type=float, default=0.0)
parser.add_argument('--individual', type=int, default=0)
parser.add_argument('--chronos_pretrained', type=str, default='./Chronos2')

args = parser.parse_args([])

print("=" * 60)
print("Test: Chronos2_head Model")
print("=" * 60)

try:
    model = Model(args)
    print("Model created successfully!")
except Exception as e:
    print(f"Error creating model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check trainable parameters
print("\n" + "=" * 60)
print("Parameter Check")
print("=" * 60)

total_params = 0
trainable_params = 0
frozen_params = 0

for name, param in model.named_parameters():
    total_params += param.numel()
    if param.requires_grad:
        trainable_params += param.numel()
        print(f"TRAINABLE: {name} - {param.shape} - {param.numel()}")
    else:
        frozen_params += param.numel()
        print(f"FROZEN:    {name} - {param.shape} - {param.numel()}")

print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Frozen parameters: {frozen_params:,}")

# Test forward pass
print("\n" + "=" * 60)
print("Forward Pass Test")
print("=" * 60)

try:
    batch_x = torch.randn(2, 336, 7)  # (batch, seq_len, nvars)
    print(f"Input shape: {batch_x.shape}")

    output = model(batch_x)
    print(f"Output shape: {output.shape}")

    assert output.shape == (2, 96, 7), f"Expected (2, 96, 7), got {output.shape}"
    print("Forward pass successful!")
except Exception as e:
    print(f"Error in forward pass: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test backward pass (gradients)
print("\n" + "=" * 60)
print("Backward Pass Test (Gradient Check)")
print("=" * 60)

try:
    # Enable gradient computation for testing
    model.train()

    batch_x = torch.randn(2, 336, 7)
    batch_y = torch.randn(2, 96, 7)

    output = model(batch_x)
    loss = torch.nn.MSELoss()(output, batch_y)
    loss.backward()

    # Check which parameters have gradients
    print("\nParameters with gradients:")
    grad_params = []
    no_grad_params = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_params.append(name)
            print(f"  HAS GRAD: {name}")
        else:
            no_grad_params.append(name)

    print(f"\nParameters WITH gradients: {len(grad_params)}")
    print(f"Parameters WITHOUT gradients: {len(no_grad_params)}")

    # Verify only Flatten_Head parameters have gradients
    flatten_head_has_grad = any('flatten_head' in name for name in grad_params)
    chronos_has_grad = any('chronos' in name for name in grad_params)

    print(f"\nFlatten_Head has gradients: {flatten_head_has_grad}")
    print(f"Chronos has gradients: {chronos_has_grad}")

    if chronos_has_grad:
        print("WARNING: Chronos parameters should NOT have gradients!")
        sys.exit(1)
    elif flatten_head_has_grad:
        print("SUCCESS: Only Flatten_Head parameters have gradients!")
    else:
        print("WARNING: No Flatten_Head gradients found!")
        sys.exit(1)

except Exception as e:
    print(f"Error in backward pass: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("All tests passed!")
print("=" * 60)
