# TCR (Temporal Contrastive Regularization) 实现计划

## Context

不依赖外部 FM，直接在 PatchTST encoder 的 patch 表示上施加时序结构约束（相邻 patch 近、远距离 patch 远）。先实现纯 TCR，不做双重约束。

### 动机

LatentTSF (ICML, arXiv:2602.00297) 发现 MSE 训练的时序模型存在 Latent Chaos：预测精度高但 latent 时序混乱。
当前 PatchTST_REPA 依赖外部 FM (Chronos2) 做对齐，但 Chronos2 past tokens 接 predict head 效果一般，其 latent 质量存疑。

### 核心思想

不依赖外部 FM，直接约束 PatchTST encoder 的 latent 时序结构：相邻 patch 的表示应更近，远距离 patch 应更远。

### 公式

```
L_temporal = -1/(P-1) * sum_t log[ exp(sim(h_t, h_{t+1}) / tau) / sum_k exp(sim(h_t, h_k) / tau) ]

总损失: Loss = MSE(pred, true) + alpha * L_temporal
```

- `h_t = zs[:, :, t, :]`，shape `(B, C, P, D)`
- `sim(a, b) = cosine_similarity`
- `tau`：温度系数，从 0.1 开始
- 正样本：相邻 patch `(t, t+1)`
- 负样本：同序列远距离 patch（更合理）或 batch 内其他序列

### 与现有工作的区别

| 方法 | 方式 | 本方法的不同 |
|------|------|-------------|
| TS2Vec (AAAI 2022) | 原始时间步 + 预训练范式 | Patch-level + 联合训练正则 |
| CoST (ICLR 2022) | 频域+时域对比，预训练 | 不分解，直接在 patch 表示上 |
| TNC (NeurIPS 2020) | 统计检验定义邻域，预训练 | 更简单，用 patch 相邻关系 |
| LatentTSF (ICML) | AutoEncoder + cosine 对齐 | 无需 AE 预训练阶段 |
| REPA (视觉) | 外部 FM 对齐 | 不依赖外部 FM，自约束 |

**新颖点**：Patch-level 时序对比 + 联合训练正则（非预训练），落在 TS2Vec / LatentTSF / REPA 三者之间的空白地带。

## 文件变动

| 文件 | 操作 | 说明 |
|------|------|------|
| `run_longExp.py` | 修改 | 加 `--lambda_temporal`, `--tau` 参数 |
| `layers/PatchTST_TCR_backbone.py` | **新建** | 独立 TCR backbone，基于 PatchTST_backbone 简化 |
| `models/PatchTST.py` | 修改 | 加 `PatchTST_TCR` 模型分支 |
| `exp/exp_main.py` | 修改 | 加 TCR loss 计算 + 训练循环集成 |
| `scripts/PatchTST_TCR.sh` | **新建** | TCR 训练脚本 |

## 详细设计

### 1. `run_longExp.py` — 新增参数 (line 63 附近)

```python
parser.add_argument('--lambda_temporal', type=float, default=0.0, help='weight for temporal contrastive loss (0=disabled)')
parser.add_argument('--tau', type=float, default=0.1, help='temperature for temporal contrastive loss')
```

### 2. `layers/PatchTST_TCR_backbone.py` — 新建

从 `PatchTST_backbone` 精简而来，去掉 alignment_mlp / patch_fusion / d_extractor 等 REPA 相关逻辑。

**核心区别**：
- 无 `alignment_mlp`，无 `patch_fusion_mlp`，无 `transformer_decoder`
- forward 返回 `(output, zs_raw)` 其中 `zs_raw: (bs, nvars, d_model, patch_num)` 是 `encoder_depth` 层的原始输出
- 复用现有组件：`TSTiEncoder`（从 PatchTST_backbone.py import）、`RevIN`、`Flatten_Head`

**结构**：
```python
class PatchTST_TCR_backbone(nn.Module):
    def __init__(self, c_in, context_window, target_window, patch_len, stride,
                 # 保留 PatchTST 核心参数
                 n_layers, d_model, n_heads, d_ff, ...,
                 encoder_depth=2,  # 中间层提取位置
                 head_type='flatten', ...):
        # RevIN
        # Patching (同 PatchTST_backbone)
        # TSTiEncoder (同 PatchTST_backbone)
        # Flatten_Head (同 PatchTST_backbone)
        # 无 alignment_mlp, 无 patch_fusion

    def forward(self, x):
        # RevIN norm
        # Patching
        # Encoder with return_intermediate=True
        #   z: final output, zs: encoder_depth 层输出
        # Head prediction: output = self.head(z)
        # RevIN denorm
        # return output, zs  
        #   zs: (bs, nvars, d_model, patch_num) — 用于 TCR loss
```

### 3. `models/PatchTST.py` — 加 PatchTST_TCR 分支

**`__init__` (line 69 附近)**：加判断
```python
elif self.model_name == 'PatchTST_TCR':
    self.contrastive = 0
    self.use_patch_fusion = False
    self.temporal_contrastive = 1
```

**backbone 构造 (line 242 附近)**：加分支
```python
elif self.model_name == 'PatchTST_TCR':
    from layers.PatchTST_TCR_backbone import PatchTST_TCR_backbone
    self.model = PatchTST_TCR_backbone(
        c_in=c_in, context_window=context_window, target_window=target_window,
        patch_len=patch_len, stride=stride, n_layers=n_layers, d_model=d_model,
        n_heads=n_heads, d_ff=d_ff, encoder_depth=encoder_depth,
        head_type=head_type, ...  # 同 PatchTST 的参数
    )
```

**`forward` (line 286 附近)**：加分支
```python
if self.model_name == 'PatchTST_TCR':
    x = x.permute(0, 2, 1)
    output, zs_raw = self.model(x)
    output = output.permute(0, 2, 1)  # (bs, pred_len, nvars)
    if return_projector:
        return output, zs_raw  # zs_raw for TCR loss
    return output
```

### 4. `exp/exp_main.py` — TCR loss + 训练循环

**新增方法 `_compute_temporal_contrastive_loss` (line 277 后)**：
```python
def _compute_temporal_contrastive_loss(self, zs_raw, tau=0.1):
    """
    zs_raw: (B, C, d_model, P) -> permute to (B, C, P, D)
    InfoNCE: 正样本=相邻 patch, 负样本=同序列所有 patch
    """
    zs = zs_raw.permute(0, 1, 3, 2)  # (B, C, P, D)
    B, C, P, D = zs.shape
    if P < 2:
        return torch.tensor(0.0, device=zs.device)
    h = F.normalize(zs, dim=-1)
    h = h.reshape(B * C, P, D)
    sim = torch.bmm(h, h.transpose(1, 2)) / tau  # (B*C, P, P)
    pos = sim[:, :-1, :].diagonal(offset=1, dim1=1, dim2=2)  # (B*C, P-1)
    log_sum_exp = torch.logsumexp(sim[:, :-1, :], dim=-1)     # (B*C, P-1)
    loss = -(pos - log_sum_exp).mean()
    return loss
```

**训练循环 forward (line 441)**：
```python
# 现有：
if hasattr(self.model, 'contrastive') and self.model.contrastive:
    outputs, zs_project, zs_tilde = self.model(...)
# 新增：
elif hasattr(self.model, 'temporal_contrastive') and self.model.temporal_contrastive:
    outputs, zs_raw = self.model(batch_x, batch_y_for_model, return_projector=True)
else:
    outputs = self.model(batch_x, batch_y_for_model)
```

**Loss 组装 (line 472)**：
```python
# 现有 contrastive loss 逻辑保持不动
# 新增 temporal loss：
if hasattr(self.model, 'temporal_contrastive') and self.model.temporal_contrastive:
    lambda_temporal = self.args.lambda_temporal
    tau = self.args.tau
    temporal_loss = self._compute_temporal_contrastive_loss(zs_raw, tau=tau)
    loss = mse_loss + lambda_temporal * temporal_loss
```

**日志**：
- 新增 `train_temporal_loss` 列表 + `loss_temporal_per_step`
- epoch 打印加 `Train Temporal: {:.7f}`
- `loss_per_step.npz` 加 `loss_temporal` 字段
- 每 100 步打印加 temporal loss

### 5. `scripts/PatchTST_TCR.sh` — 新建

参考 `PatchTST.sh` 格式，加入 TCR 相关参数：

```bash
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=PatchTST_TCR
device="cuda:0"

root_path_name=./dataset/
data_path_name=ETTh1.csv
data_name=ETTh1

random_seed=2021
pred_len=720
d_model=16
d_ff=128
e_layers=4

# TCR hyperparameters
lambda_temporal=0.1
tau=0.1

python -u run_longExp.py \
  --random_seed $random_seed \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id ${data_name}_${seq_len}_${pred_len} \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --e_layers $e_layers \
  --n_heads 4 \
  --d_model $d_model \
  --d_ff $d_ff \
  --dropout 0.3\
  --fc_dropout 0.3\
  --head_dropout 0\
  --patch_len 16\
  --stride 8\
  --des 'Exp' \
  --train_epochs 20\
  --itr 1 --batch_size 128 --learning_rate 0.0001 \
  --lambda_temporal $lambda_temporal \
  --tau $tau \
  --device $device \
  >logs/LongForecasting/${model_name}_${data_name}_sl${seq_len}_pl${pred_len}_dm${d_model}_el${e_layers}_lt${lambda_temporal}_tau${tau}.log
```

## 不修改的部分

- `layers/PatchTST_backbone.py`：完全不动
- decomposition 分支：PatchTST_TCR 不支持
- validation/test loop：不计算 TCR loss
- AMP 分支：同步更新即可

## 验证

```bash
# 1. PatchTST_TCR 基本功能
python -u run_longExp.py --is_training 1 --model PatchTST_TCR --data custom \
  --root_path ./dataset/ --data_path weather.csv \
  --features M --seq_len 336 --pred_len 96 \
  --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 \
  --patch_len 16 --stride 8 --batch_size 128 --learning_rate 0.0001 \
  --lambda_temporal 0.1 --tau 0.1

# 2. 回归测试：原有 PatchTST / PatchTST_REPA 行为不变
python -u run_longExp.py --is_training 1 --model PatchTST ...
```

## 可扩展方向

- **双重约束**：`Loss = MSE + alpha * L_temporal + beta * L_alignment(Chronos)`，自对齐保证 latent 质量，外部对齐注入 FM 知识
- **选择性对齐**：加 gating 机制，模型自动决定哪些 patch 需要对齐
- **逐层对齐**：PatchTST 不同层对齐 Chronos 不同层
