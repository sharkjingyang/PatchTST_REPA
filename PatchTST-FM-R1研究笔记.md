# PatchTST-FM-R1 研究笔记

## 模型概览

**PatchTST-FM** 是 IBM Research 发布的时间序列预测基础模型（Foundation Model），基于 PatchTST 架构大幅扩展，在 GiftEval 基准上达到 SOTA。

- **论文**: [Revisiting the Generic Transformer: Deconstructing a Strong Baseline for Time Series Foundation Models](https://arxiv.org/abs/2602.06909)（arXiv:2602.06909，2026）
- **模型**: [ibm-research/patchtst-fm-r1](https://huggingface.co/ibm-research/patchtst-fm-r1)
- **代码库**: [ibm-granite/granite-tsfm @ patchtst-fm](https://github.com/ibm-granite/granite-tsfm/tree/patchtst-fm)
- **许可证**: CC-BY-NC-SA-4.0

---

## 模型规格

| 属性 | 值 |
|------|-----|
| 参数量 | ~260M |
| 上下文长度 | 8192 |
| Patch 大小 | 16 |
| Embedding 维度 | 1024 |
| 分位数头 | 99 个分位数 |

---

## 核心创新点

1. **超长上下文** — `context_length=8192`，能捕捉更长周期规律
2. **预测即重建** — 推理时将预测期 mask，通过重建而非自回归生成来预测
3. **概率预测** — 输出 99 个分位数，天然支持不确定性估计
4. **同时处理缺失值** — 对输入缺失值和预测期一起做重建，天然支持 imputation
5. **残差投影块** — 输入/输出 projection 引入残差结构

---

## 训练数据

- **GiftEvalPretrain**（Salesforce）
- **KernelSynth** 合成数据（多种周期核函数）
- **TSMixup**（基于 Chronos 论文的混合增强策略）

---

## 安装

```bash
pip install git+https://github.com/ibm-granite/granite-tsfm.git@patchtst-fm
```

> 注意：使用自定义的 `PatchTSTFMForPrediction`，**不是** transformers 标准的 `PatchTSTForPrediction`。

---

## 推理方式一：直接调用模型

适合：快速验证、数据已经是 Tensor 格式。

```python
import torch
from tsfm_public.models.patchtst_fm import PatchTSTFMForPrediction

# 加载模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model = PatchTSTFMForPrediction.from_pretrained(
    "ibm-research/patchtst-fm-r1",
    device_map=device,
)
model.eval()

# 输入：List of 1D float Tensor，每条长度可不同（上限 8192）
inputs = [
    torch.randn(512),
    torch.randn(1024),
]

with torch.no_grad():
    output = model(
        inputs=inputs,
        prediction_length=96,
        quantile_levels=[0.1, 0.5, 0.9],  # 不传则返回全部 99 个分位数
    )

# output.quantile_predictions: shape (batch_size, num_quantiles, prediction_length)
preds = output.quantile_predictions
print(preds.shape)  # (2, 3, 96)
```

---

## 推理方式二：使用 Predictor 封装

适合：跑完整评估、数据是 GluonTS 格式、关心时间戳、需要 OOM 保护。

```python
import numpy as np
import pandas as pd
import torch
from tsfm_public.models.patchtst_fm import (
    PatchTSTFMForPrediction,
    PatchTSTFMEvalPredictor,
)

# 加载模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model = PatchTSTFMForPrediction.from_pretrained(
    "ibm-research/patchtst-fm-r1",
    device_map=device,
)

# 准备输入：List of dict，每条含 target（1D numpy）和 start（pd.Period）
test_data_input = [
    {
        "target": np.random.randn(512).astype(np.float32),
        "start": pd.Period("2020-01-01", freq="H"),
    },
]

# 创建 Predictor
predictor = PatchTSTFMEvalPredictor(
    model=model,
    prediction_length=96,
    dataset_name="my_dataset",
    quantile_levels=[0.1, 0.5, 0.9],
)

# 推理
forecasts = predictor.predict(test_data_input, batch_size=128)

# 读取结果
for fc in forecasts:
    print(fc.forecast_keys)   # ['0.1', '0.5', '0.9']
    print(fc["0.5"])          # 中位数预测，shape: (prediction_length,)
```

---

## 两种推理方式对比

| 功能 | 直接调用模型 | 使用 Predictor |
|------|------------|---------------|
| 输入格式 | `List[torch.Tensor]`，需自己处理 NaN | `List[dict]`，含 `target`+`start`，自动处理 NaN |
| 批处理 | 手动分 batch | 自动分批 |
| OOM 处理 | 无，直接报错 | 自动将 `batch_size` 减半重试 |
| 输出格式 | `torch.Tensor`，shape `(B, Q, T)` | `List[QuantileForecast]`，含时间戳 |
| 时间戳 | 不感知 | 自动根据 `start` 推断预测起始时间 |
| 进度显示 | 无 | tqdm 进度条 |

Predictor 本质是薄封装，模型计算完全相同。

---

## 关键参数说明

| 参数 | 说明 |
|------|------|
| `prediction_length` | 预测步长，推理时可动态指定 |
| `quantile_levels` | 分位数列表，如 `[0.1, 0.5, 0.9]`，不传返回全 99 个 |
| `inputs` | 变长列表，每条序列长度不同都可以，上限 8192 |
| 缺失值 | 输入中的 `NaN` 自动处理（用均值填充） |

---

## 注意事项

- **零样本**：不需要任何微调，直接推理新数据集
- **输入无需对齐长度**：模型支持变长序列
- **归一化**：内部用 RevIN + sinh 变换，不需要手动标准化
- **生产环境**：建议参考 Predictor 的 OOM 保护逻辑
