# Research Ideas

## Idea 1: FutureQueryDecoder — 让 Encoder/Decoder/Head 分工明确

### 动机

**当前 FutureAlign + PatchwiseHead 的问题**：

用 PatchwiseHead 替换 Flatten_Head 后，train/vali loss 下降但 test loss 上升。根本原因：

- Flatten_Head 有 74K 参数，全局混合所有 patch，隐式承担了"从过去预测未来"的映射
- PatchwiseHead 只有 11K 参数，每个 patch 独立解码，要求 z_enc[i] 本身已经是 future[i] 的可解码表示
- 结果是 Encoder 被迫同时承担"编码过去"和"预测未来"两件事，用同一组参数完成更难的任务，模型在训练集上找到伪相关 shortcut，无法泛化到 test

**对比 Chronos2**：它有独立的 Decoder（cross-attention）专门负责"从过去生成未来表示"，PatchwiseHead 只需要解码。三者分工明确。

### 方案

在 Encoder 和 PatchwiseHead 之间插入一个轻量的 **FutureQueryDecoder**：

```
x_past
  ↓ TSTiEncoder
z_enc: (bs, nvars, patch_num, d_model)       ← 编码过去
  ↓ FutureQueryDecoder (cross-attention)
z_future: (bs, nvars, output_patch_num, d_model)  ← 预测未来
  ↓ PatchwiseHead
pred: (bs, nvars, pred_len)                  ← 解码
```

**FutureQueryDecoder 结构**：

```python
class FutureQueryDecoder(nn.Module):
    def __init__(self, output_patch_num, d_model, n_heads, d_ff, dropout=0.0):
        super().__init__()
        # output_patch_num 个可学习的未来位置查询（每个对应一个未来 patch）
        self.future_queries = nn.Parameter(torch.randn(output_patch_num, d_model))

        # Cross-attention: Q = future queries, K/V = past encoder tokens
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, z_enc):
        # z_enc: (bs*nvars, patch_num, d_model)
        queries = self.future_queries.unsqueeze(0).expand(z_enc.shape[0], -1, -1)
        attn_out, _ = self.cross_attn(queries, z_enc, z_enc)
        queries = self.norm1(queries + attn_out)
        out = self.norm2(queries + self.ffn(queries))
        return out  # (bs*nvars, output_patch_num, d_model)
```

### 参数量估算（d_model=128, n_heads=8, d_ff=256, output_patch_num=6）

| 模块 | 参数量 |
|------|-------:|
| future_queries | 768 |
| cross_attn (4 × d_model²) | 65,536 |
| FFN (2 × d_model × d_ff) | 65,536 |
| LayerNorm × 2 | 512 |
| **FutureQueryDecoder 合计** | **~132K** |

对比现有 TransformerDecoder（self-attention only）也是 ~132K，但 cross-attention 有明确的 Q（未来）/KV（过去）分工。

### Alignment Loss 的改进

加了 FutureQueryDecoder 之后，alignment target 更自然：

| 对齐方式 | gap 大小 | 合理性 |
|----------|---------|--------|
| 当前：z_enc（past encoder） ↔ encoder(x_future) | 大（past vs future） | 弱 |
| 新：z_future（decoder output） ↔ encoder(x_future) | 小（都是 future-oriented） | 强 |

Loss 不变，只是把 z_enc 换成 z_future 作为 alignment 的 student 端：

```python
loss_align = -(F.normalize(z_future, dim=-1) * F.normalize(z_teacher.detach(), dim=-1)).sum(dim=-1).mean()
           + F.mse_loss(z_future, z_teacher.detach())
```

### 核心 Claim（与 Chronos2 的本质区别）

表面上新方案的架构（Encoder + cross-attn Decoder + PatchwiseHead）和 Chronos2（T5 Encoder + T5 Decoder + PatchwiseHead）相似，但核心差异在于**知识来源**：

- Chronos2：靠 6B+ 样本预训练，Decoder 学会从 past 预测 future
- 新方案：靠 Chronos2 的 `encoder(x_future)` 作为监督信号，训练轻量 Decoder

> **用 Chronos2 的预训练时序知识蒸馏进 FutureQueryDecoder，使其在推理时仅从 x_past 就能生成 Chronos2-quality 的 future representations，且推理时完全不依赖 Chronos2。**

训练时的数据流：

```
x_past  → Encoder → z_enc ──────────────────────────────────────────┐
                              ↓ cross-attention                      │
          learned queries → FutureQueryDecoder → z_future            │
                                                     ↓               │
                               alignment loss: z_future ≈ z_teacher  │
                                                     ↑               │
x_future → Chronos2 (frozen) → encoder(x_future) = z_teacher        │
                                                                      │
z_enc ────────────────────────────────────────────────────────────────┘
（z_enc 仍然参与 MSE loss，通过 PatchwiseHead → pred → MSE(pred, y)）
```

推理时 Chronos2 完全不参与：
```
x_past → Encoder → z_enc → FutureQueryDecoder → z_future → PatchwiseHead → pred
```

### 与现有工作的对比

| 模型 | Encoder | "预测"由谁做 | Head | 推理需要 FM |
|------|---------|------------|------|------------|
| PatchTST + Flatten_Head | bidirectional | Head（全局 Linear） | Flatten | 否 |
| PatchTST + PatchwiseHead（当前） | bidirectional | Encoder（被迫） | Patchwise | 否 |
| **新方案** | bidirectional | **FutureQueryDecoder（Chronos2 监督）** | Patchwise | **否** |
| Chronos2 | T5 Encoder | T5 Decoder（预训练） | Patchwise | 是 |

### 实验验证方向

1. **基线对比**：PatchTST + Flatten_Head vs 新方案（参数量接近时）
2. **Decoder 消融**：有/无 FutureQueryDecoder，其他相同
3. **Alignment 消融**：对齐 z_enc vs 对齐 z_future
4. **PatchwiseHead 必要性**：新方案 + PatchwiseHead vs 新方案 + Flatten_Head

### 潜在风险

- Learned queries 可能退化为只关注固定的 past token，而不是根据输入动态查询
- cross-attention 的 output 是否真的 future-oriented，需要 CKA 或 temporal locality 指标验证
- 训练稳定性：三个损失（MSE + Teacher + Alignment）+ 新的 Decoder 参数
