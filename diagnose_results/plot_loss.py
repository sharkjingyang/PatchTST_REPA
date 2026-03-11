"""
Plot loss curves from training logs.
Usage: python plot_loss.py <setting_name>
Example: python plot_loss.py etth1_REPA_336_720
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

def plot_loss(setting):
    loss_file = f'./results/{setting}/loss_per_step.npz'

    if not os.path.exists(loss_file):
        print(f"Error: {loss_file} not found!")
        print("Available files:")
        if os.path.exists('./logs/loss_curves/'):
            for f in os.listdir('./logs/loss_curves/'):
                print(f"  - {f}")
        return

    data = np.load(loss_file)
    steps = data['steps']
    loss = data['loss']
    loss_mse = data['loss_mse']
    loss_contrastive = data['loss_contrastive']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Total Loss
    axes[0].plot(steps, loss, 'b-', linewidth=0.5)
    axes[0].set_xlabel('Steps')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Total Loss')
    axes[0].grid(True, alpha=0.3)

    # MSE Loss
    axes[1].plot(steps, loss_mse, 'g-', linewidth=0.5)
    axes[1].set_xlabel('Steps')
    axes[1].set_ylabel('MSE Loss')
    axes[1].set_title('MSE Loss')
    axes[1].grid(True, alpha=0.3)

    # Contrastive Loss
    axes[2].plot(steps, loss_contrastive, 'r-', linewidth=0.5)
    axes[2].set_xlabel('Steps')
    axes[2].set_ylabel('Contrastive Loss')
    axes[2].set_title('Contrastive Loss')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(f'Loss Curves - {setting}')
    plt.tight_layout()

    # Save plot
    output_file = f'./results/{setting}/loss_curve.png'
    plt.savefig(output_file, dpi=150)
    print(f"Plot saved to {output_file}")

    plt.show()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python plot_loss.py <setting_name>")
        print("Example: python plot_loss.py etth1_REPA_336_720")
        sys.exit(1)

    setting = sys.argv[1]
    plot_loss(setting)
