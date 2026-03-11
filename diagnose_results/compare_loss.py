import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. 参数设置 ---
path_repa = '/home/ma-user/work/PatchTST_REPA-main/results/336_720_PatchTST_ETTh1_ftM_sl336_ll48_pl720_dm16_nh4_el4_dl1_df128_fc1_ebtimeF_dtTrue_Exp_0/loss_per_step.npz'
path_base = '/home/ma-user/work/PatchTST_REPA-main/results/336_720_PatchTST_REPA_ETTh1_ftM_sl336_ll48_pl720_dm16_nh4_el4_dl1_df128_fc1_ebtimeF_dtTrue_Exp_0/loss_per_step.npz'

results_folder = './diagnose_results'
setting = 'PatchTST vs PatchTST_REPA'

# --- 2. 加载数据 ---
# 加载 REPA 版本
data_repa = np.load(path_repa)
mse_repa = data_repa['loss_mse']

# 加载 Baseline 版本
data_base = np.load(path_base)
mse_base = data_base['loss_mse']

# --- 3. 绘图逻辑 ---
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

steps_repa = np.arange(len(mse_repa))
steps_base = np.arange(len(mse_base))

# 绘制对比曲线
ax.plot(steps_base, mse_base, color='gray', label='PatchTST (Base)', linewidth=0.8, alpha=0.7)
ax.plot(steps_repa, mse_repa, color='royalblue', label='PatchTST_REPA', linewidth=0.8)

# 装饰图形
ax.set_xlabel('Steps')
ax.set_ylabel('MSE Loss')
ax.set_title(f'MSE Loss Comparison\n{setting}')
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend()

# --- 4. 保存 ---
plt.tight_layout()
save_path = os.path.join(results_folder, 'loss_comparison.png')
plt.savefig(save_path, dpi=200)
plt.show()
print(f"Comparison plot saved to {save_path}")
