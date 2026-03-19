"""
Chronos2 Direct Inference Test (no training)
============================================
测试 Chronos2 原生 pipeline.predict() 的预测表现，不做任何训练/微调。

Usage:
    python test_Chronos2_direct.py \
        --data custom --root_path ./dataset/ --data_path weather.csv \
        --features M --seq_len 336 --pred_len 96 \
        --batch_size 128 --chronos_pretrained ./Chronos2

    # Or with shell script:
    sh ./scripts/Chronos_original.sh
"""

import argparse
import os
import torch
import numpy as np
from chronos import Chronos2Pipeline
from data_provider.data_factory import data_provider
from utils.metrics import metric


def set_seed(seed=2021):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def main():
    parser = argparse.ArgumentParser(description='Chronos2 Direct Inference Test')

    # basic config
    parser.add_argument('--random_seed', type=int, default=2021, help='random seed')
    parser.add_argument('--model_id', type=str, default='test', help='model id')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
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
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # model
    parser.add_argument('--chronos_pretrained', type=str, default='./Chronos2',
                        help='Chronos pretrained model path')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')

    args = parser.parse_args()

    # set random seed
    set_seed(args.random_seed)

    # device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    print(f"Using device: {device}")

    # load Chronos2 pipeline
    print(f"Loading Chronos2 from {args.chronos_pretrained}...")
    pipeline = Chronos2Pipeline.from_pretrained(args.chronos_pretrained)
    pipeline.model.to(device)
    pipeline.model.eval()

    # load test data
    print(f"Loading test data: {args.data}...")
    _, test_loader = data_provider(args, flag='test')
    print(f"Test data size: {len(test_loader.dataset)}")

    preds = []
    trues = []

    print("Running Chronos2 direct inference...")
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            # batch_x: (bs, seq_len, nvars)
            # batch_y: (bs, seq_len + pred_len, nvars)

            # Permute to Chronos2 input format: (bs, nvars, seq_len)
            batch_x_perm = batch_x.permute(0, 2, 1).float().to(device)

            # Get ground truth: last pred_len timesteps
            batch_y_true = batch_y[:, -args.pred_len:, :].float().to(device)  # (bs, pred_len, nvars)

            # Call Chronos2 predict
            # Returns: list of (nvars, n_quantiles, pred_len), length = bs
            forecast = pipeline.predict(batch_x_perm, prediction_length=args.pred_len)

            # Take median quantile (index = n_quantiles // 2)
            # Each element in list: (nvars, n_quantiles, pred_len) -> (nvars, pred_len)
            median_forecasts = []
            for f in forecast:
                n_quantiles = f.shape[1]
                mid_idx = n_quantiles // 2
                median_forecasts.append(f[:, mid_idx, :])  # (nvars, pred_len)

            # Stack to (bs, nvars, pred_len)
            pred_tensor = torch.stack(median_forecasts, dim=0)  # (bs, nvars, pred_len)

            # Permute back to (bs, pred_len, nvars) to match ground truth
            pred_tensor = pred_tensor.permute(0, 2, 1)  # (bs, pred_len, nvars)

            # Collect
            preds.append(pred_tensor.detach().cpu().numpy())
            trues.append(batch_y_true.detach().cpu().numpy())

            if i % 20 == 0:
                print(f"  Batch {i}/{len(test_loader)}")

    # Compute metrics
    preds = np.concatenate(preds, axis=0)  # (N, pred_len, nvars)
    trues = np.concatenate(trues, axis=0)    # (N, pred_len, nvars)
    print(f"Total samples: {preds.shape[0]}")

    # Compute metrics (metrics expect (N, pred_len, nvars))
    mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)

    print(f"\n===== Chronos2 Direct Inference Results =====")
    print(f"Dataset: {args.data}")
    print(f"seq_len: {args.seq_len}, pred_len: {args.pred_len}")
    print(f"MAE:  {mae:.6f}")
    print(f"MSE:  {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAPE: {mape:.6f}")
    print(f"MSPE: {mspe:.6f}")
    print(f"RSE:  {rse:.6f}")
    print(f"CORR: {np.mean(corr):.6f}")
    print(f"==============================================")

    # Save results
    result_folder = './test_results/Chronos2_direct/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    setting = f"{args.data}_{args.seq_len}_{args.pred_len}"
    np.save(os.path.join(result_folder, f'{setting}_pred.npy'), preds)
    np.save(os.path.join(result_folder, f'{setting}_true.npy'), trues)
    print(f"Results saved to {result_folder}")


if __name__ == '__main__':
    main()
