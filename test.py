"""
Quick test: PatchTSTFMForPrediction forward pass with manual mini-batch loop
每次只送 infer_batch_size 条序列，避免 bs*nvars 全部堆进显存
"""
import torch
import numpy as np
from tsfm_public.models.patchtst_fm import PatchTSTFMForPrediction

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

print("Loading model...")
model = PatchTSTFMForPrediction.from_pretrained("./Patchtst-Fm-R1", device_map=device)
model.eval()
print("Model loaded.")

# simulate a batch: bs=2, nvars=7, seq_len=336
bs, nvars, seq_len, pred_len = 2, 7, 336, 96
batch = torch.randn(bs, seq_len, nvars)
# flatten: (bs*nvars, seq_len)
inputs = [batch[i, :, j] for i in range(bs) for j in range(nvars)]
print(f"inputs: {len(inputs)} sequences of length {seq_len}")

# 分批推理，每批 infer_batch_size 条，避免 OOM
infer_batch_size = 16
all_preds = []
with torch.no_grad():
    for start in range(0, len(inputs), infer_batch_size):
        chunk = inputs[start: start + infer_batch_size]
        output = model(inputs=chunk, prediction_length=pred_len, quantile_levels=[0.5])
        all_preds.append(output.quantile_predictions.cpu())  # (chunk, 1, pred_len)

preds = torch.cat(all_preds, dim=0)  # (bs*nvars, 1, pred_len)
print(f"output shape: {preds.shape}")  # expect (14, 1, 96)
assert preds.shape == (bs * nvars, 1, pred_len), f"unexpected shape: {preds.shape}"
print("OK")
