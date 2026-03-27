"""
PatchTST-FM-R1 Direct Inference Test (no training / zero-shot)
==============================================================
测试 PatchTST-FM-R1 原生推理的预测表现，不做任何训练/微调。

Usage:
    python models/PatchTST_FM_zeroshot.py \
        --data custom --root_path ./dataset/ --data_path weather.csv \
        --features M --seq_len 336 --pred_len 96 \
        --batch_size 32

    # Or with shell script:
    sh ./scripts/PatchTST_FM_zeroshot.sh
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from tsfm_public.models.patchtst_fm import PatchTSTFMForPrediction
from data_provider.data_factory import data_provider
from utils.metrics import metric


def set_seed(seed=2021):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def main():
    parser = argparse.ArgumentParser(description='PatchTST-FM-R1 Zero-shot Inference Test')

    # basic config
    parser.add_argument('--random_seed', type=int, default=2021, help='random seed')
    parser.add_argument('--model_id', type=str, default='test', help='model id')

    # data loader
    parser.add_argument('--data', type=str, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=336, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # model
    parser.add_argument('--fm_pretrained', type=str, default='./Patchtst-Fm-R1',
                        help='PatchTST-FM pretrained model path or HuggingFace model ID')
    # 每个 batch 含 bs*nvars 条序列同时送入 FM-R1，显存占用高，建议从 32 开始调
    parser.add_argument('--batch_size', type=int, default=8, help='batch size of data loader')
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')

    # device
    parser.add_argument('--device', type=str, default='cuda:0', help='device (e.g. cuda:0, cuda:1, cpu)')

    args = parser.parse_args()

    set_seed(args.random_seed)

    # device fallback
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        args.device = 'cpu'
    print(f"Using device: {args.device}")

    # load FM-R1 model
    print(f"Loading PatchTST-FM-R1 from {args.fm_pretrained}...")
    model = PatchTSTFMForPrediction.from_pretrained(
        args.fm_pretrained,
        device_map=args.device,
    )
    model.eval()
    print("Model loaded.")

    # load test data
    print(f"Loading test data: {args.data_path}...")
    _, test_loader = data_provider(args, flag='test')
    print(f"Test data size: {len(test_loader.dataset)}")

    preds = []
    trues = []

    print("Running PatchTST-FM-R1 zero-shot inference...")
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            # batch_x: (bs, seq_len, nvars)
            # batch_y: (bs, label_len + pred_len, nvars)
            bs, seq_len, nvars = batch_x.shape

            # Ground truth: last pred_len timesteps
            batch_y_true = batch_y[:, -args.pred_len:, :].float()  # (bs, pred_len, nvars)

            # FM-R1 is channel-independent: flatten (bs, seq_len, nvars) -> List of (bs*nvars) 1D tensors
            # permute: (bs, seq_len, nvars) -> (bs, nvars, seq_len) -> (bs*nvars, seq_len)
            batch_x_flat = batch_x.float().cpu().permute(0, 2, 1).reshape(bs * nvars, seq_len)
            inputs = [batch_x_flat[j] for j in range(bs * nvars)]

            # output.quantile_predictions: (bs*nvars, num_quantiles, pred_len)
            output = model(
                inputs=inputs,
                prediction_length=args.pred_len,
                quantile_levels=[0.5],  # only median
            )

            # (bs*nvars, 1, pred_len) -> (bs*nvars, pred_len)
            pred_flat = output.quantile_predictions[:, 0, :].cpu()

            # reshape: (bs*nvars, pred_len) -> (bs, nvars, pred_len) -> (bs, pred_len, nvars)
            pred_tensor = pred_flat.reshape(bs, nvars, args.pred_len).permute(0, 2, 1)

            preds.append(pred_tensor.numpy())
            trues.append(batch_y_true.numpy())

            if i % 20 == 0:
                print(f"  Batch {i}/{len(test_loader)}")

    # compute metrics
    preds = np.concatenate(preds, axis=0)  # (N, pred_len, nvars)
    trues = np.concatenate(trues, axis=0)  # (N, pred_len, nvars)
    print(f"Total samples: {preds.shape[0]}")

    mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)

    print(f"\n===== PatchTST-FM-R1 Zero-shot Inference Results =====")
    print(f"Dataset:  {args.data_path}")
    print(f"seq_len:  {args.seq_len},  pred_len: {args.pred_len}")
    print(f"MAE:  {mae:.6f}")
    print(f"MSE:  {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"======================================================")


if __name__ == '__main__':
    main()
