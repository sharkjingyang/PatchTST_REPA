# Research Ideas

## 1. Temporal Contrastive Regularization (TCR)

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

### 实现参考

```python
def temporal_contrastive_loss(zs, tau=0.1):
    """
    zs: (B, C, P, D) - encoder intermediate output
    """
    B, C, P, D = zs.shape
    h = F.normalize(zs, dim=-1)
    h = h.reshape(B * C, P, D)

    # 相似度矩阵: (B*C, P, P)
    sim = torch.bmm(h, h.transpose(1, 2)) / tau

    # 正样本: patch t -> patch t+1
    pos = sim[:, :-1, :].diagonal(offset=1, dim1=1, dim2=2)  # (B*C, P-1)

    # InfoNCE
    log_sum_exp = torch.logsumexp(sim[:, :-1, :], dim=-1)     # (B*C, P-1)
    loss = -(pos - log_sum_exp).mean()
    return loss
```

### 设计选择

| 选项 | 选择空间 | 推荐 |
|------|---------|------|
| 正样本 | 严格相邻 (t, t+1) / 窗口内 (t, t+-k) / 软权重 | 严格相邻 |
| 负样本 | 同序列远距离 / 跨序列 / 混合 | 同序列远距离 |
| 温度 tau | 0.07~0.5 | 0.1 |
| alpha | 0.01~0.5 | 先试 0.1 |

### 与现有工作的区别

| 方法 | 方式 | 本方法的不同 |
|------|------|-------------|
| TS2Vec (AAAI 2022) | 原始时间步 + 预训练范式 | Patch-level + 联合训练正则 |
| CoST (ICLR 2022) | 频域+时域对比，预训练 | 不分解，直接在 patch 表示上 |
| TNC (NeurIPS 2020) | 统计检验定义邻域，预训练 | 更简单，用 patch 相邻关系 |
| LatentTSF (ICML) | AutoEncoder + cosine 对齐 | 无需 AE 预训练阶段 |
| REPA (视觉) | 外部 FM 对齐 | 不依赖外部 FM，自约束 |

**新颖点**：Patch-level 时序对比 + 联合训练正则（非预训练），落在 TS2Vec / LatentTSF / REPA 三者之间的空白地带。

### 可扩展方向

- **双重约束**：`Loss = MSE + alpha * L_temporal + beta * L_alignment(Chronos)`，自对齐保证 latent 质量，外部对齐注入 FM 知识
- **选择性对齐**：加 gating 机制，模型自动决定哪些 patch 需要对齐
- **逐层对齐**：PatchTST 不同层对齐 Chronos 不同层
