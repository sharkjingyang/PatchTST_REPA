from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST
from utils.tools import adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')


class QuantileLoss(nn.Module):
    """分位数损失 (Quantile Loss / Pinball Loss)

    pred: (bs, nvars, num_quantiles, pred_len)
    target: (bs, pred_len, nvars)
    """
    def __init__(self, num_quantiles=20):
        super().__init__()
        # Chronos2 的分位数: [0.01, 0.05, 0.1, 0.15, ..., 0.99]
        quantiles = torch.linspace(0.01, 0.99, num_quantiles)
        self.register_buffer('quantiles', quantiles)

    def forward(self, pred, target):
        """
        pred: (bs, nvars, num_quantiles, pred_len)
        target: (bs, pred_len, nvars)
        """
        # 调整 target 维度: target -> (bs, nvars, 1, pred_len)
        target = target.permute(0, 2, 1).unsqueeze(2)  # (bs, nvars, 1, pred_len)

        # 广播到相同维度
        # pred: (bs, nvars, num_quantiles, pred_len)
        # target: (bs, nvars, num_quantiles, pred_len) after broadcasting

        # Quantile loss formula: 2 * |(y - ŷ) * (I(y < ŷ) - q)|
        quantile_loss = 2 * torch.abs(
            (target - pred) * ((target <= pred).float() - self.quantiles.view(1, 1, -1, 1))
        )
        return quantile_loss.mean()

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        # TiViT is now created inside PatchTST model

    def _print_parameter_stats(self):
        """Print parameter statistics at the start of training"""
        print("\n" + "=" * 60)
        print("Parameter Statistics:")
        print("=" * 60)

        model = self.model
        model_name = self.args.model

        # Get model components
        model_backbone = model.model if hasattr(model, 'model') else None

        # Get head info
        head_type = type(model_backbone.head).__name__ if model_backbone and hasattr(model_backbone, 'head') else 'None'
        use_channel_fusion = model_backbone.use_channel_fusion if model_backbone and hasattr(model_backbone, 'use_channel_fusion') else False

        # Total parameters
        all_total = sum(p.numel() for p in model.parameters())
        all_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\nModel Configuration:")
        print(f"  Model:           {model_name}")
        print(f"  Head type:       {head_type}")
        print(f"  Channel fusion:  {use_channel_fusion}")

        # Detect feature extractor
        feature_extractor = None
        fe_total = 0

        if hasattr(model, 'tivit') and model.tivit is not None:
            feature_extractor = 'TiViT'
            fe_total = sum(p.numel() for p in model.tivit.parameters())
        elif hasattr(model, 'mantis') and model.mantis is not None:
            feature_extractor = 'Mantis'
            fe_total = sum(p.numel() for p in model.mantis.network.parameters())
        elif hasattr(model, 'chronos_model') and model.chronos_model is not None:
            feature_extractor = 'Chronos'
            fe_total = sum(p.numel() for p in model.chronos_model.parameters())

        print(f"\nTotal parameters (all):              {all_total:,}")
        if feature_extractor:
            print(f"Total parameters (excl. {feature_extractor}): {all_total - fe_total:,}")
        print(f"Trainable parameters:                {all_trainable:,}")

        # Detailed module breakdown
        print(f"\nModule Parameters:")

        # Backbone (encoder)
        if model_backbone and hasattr(model_backbone, 'backbone'):
            bb_total = sum(p.numel() for p in model_backbone.backbone.parameters())
            print(f"  Backbone (encoder):    {bb_total:,}")

        # Projector
        if model_name in ['PatchTST_REPA', 'PatchTST_REPA_Fusion']:
            if hasattr(model_backbone, 'projector') and model_backbone.projector is not None:
                proj_total = sum(p.numel() for p in model_backbone.projector.parameters())
                print(f"  Projector:             {proj_total:,}")
            elif hasattr(model, 'model_trend') and hasattr(model.model_trend, 'projector'):
                proj_total = sum(p.numel() for p in model.model_trend.projector.parameters()) * 2
                print(f"  Projector (2):        {proj_total:,}")

        # Channel Fusion components
        if use_channel_fusion:
            if hasattr(model_backbone, 'channel_fusion_mlp') and model_backbone.channel_fusion_mlp is not None:
                cf_mlp_total = sum(p.numel() for p in model_backbone.channel_fusion_mlp.parameters())
                print(f"  ChannelFusionMLP:      {cf_mlp_total:,}")
            if hasattr(model_backbone, 'transformer_decoder') and model_backbone.transformer_decoder is not None:
                td_total = sum(p.numel() for p in model_backbone.transformer_decoder.parameters())
                print(f"  TransformerDecoder:   {td_total:,}")

        # Head
        if model_backbone and hasattr(model_backbone, 'head'):
            head_total = sum(p.numel() for p in model_backbone.head.parameters())
            print(f"  Head ({head_type}):         {head_total:,}")

        # RevIN
        if model_backbone and hasattr(model_backbone, 'revin_layer'):
            revin_total = sum(p.numel() for p in model_backbone.revin_layer.parameters())
            print(f"  RevIN:                {revin_total:,}")

        # Feature extractor
        if feature_extractor:
            print(f"\n  {feature_extractor} (frozen):    {fe_total:,}")

        print("=" * 60)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
            'PatchTST_REPA': PatchTST,  # PatchTST with feature alignment (projector + contrastive loss)
            'PatchTST_REPA_Fusion': PatchTST,  # PatchTST with channel fusion branch
        }

        # Print model info based on model_name
        if self.args.model == 'PatchTST_REPA':
            print(f"\n>>> Using PatchTST_REPA: projector + contrastive loss")
        elif self.args.model == 'PatchTST_REPA_Fusion':
            print(f"\n>>> Using PatchTST_REPA_Fusion: channel fusion branch")
        else:
            print(f"\n>>> Using {self.args.model}: original PatchTST")

        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        # MSE loss for prediction
        head_type = getattr(self.args, 'head_type', 'flatten')
        if head_type == 'quantile':
            # 使用分位数损失
            num_quantiles = getattr(self.args, 'num_quantiles', 20)
            criterion = QuantileLoss(num_quantiles=num_quantiles)
        else:
            criterion = nn.MSELoss()
        return criterion

    def _compute_contrastive_loss(self, zs_project, zs_tilde):
        """
        Compute contrastive loss between projected features and TiViT/Mantis/Chronos features.
        对每个 nvar 单独计算 cosine similarity，然后求和。

        Args:
            zs_project: (bs, nvars, patch_num, d_model) or (bs, nvars, patch_num, projector_dim) - PatchTST projected features
            zs_tilde: (bs, nvars, d_vit) for TiViT/Mantis, or (bs, nvars, num_patches, d_vit) for Chronos
            contractive_type: 'mean_pool' or 'patch_wise'

        Returns:
            loss: scalar contrastive loss
        """
        contractive_type = getattr(self.args, 'contrastive_type', 'mean_pool')

        # For Chronos with patch_wise: patch_num is already consistent (interpolated in data prep)
        # For others or mean_pool: use mean pooling
        if zs_tilde.dim() == 4 and contractive_type == 'patch_wise':
            # patch_num is already consistent (interpolated to seq_len in data prep)
            # Normalize features
            zs_project = F.normalize(zs_project, dim=-1)
            zs_tilde = F.normalize(zs_tilde, dim=-1)

            # Compute cosine similarity per patch per nvar
            # Shape: (bs, nvars, patch_num)
            similarity = (zs_project * zs_tilde).sum(dim=-1)

            # Sum over all (batch_size * nvars * patch_num), then normalize
            loss = -similarity.sum() / similarity.numel()
        else:
            # mean_pool: use mean pooling over patch dimension
            zs_project = zs_project.mean(dim=2)  # -> (bs, nvars, d)
            if zs_tilde.dim() == 4:
                zs_tilde = zs_tilde.mean(dim=2)  # -> (bs, nvars, d)

            # Normalize features
            zs_project = F.normalize(zs_project, dim=-1)
            zs_tilde = F.normalize(zs_tilde, dim=-1)

            # Compute cosine similarity per nvar (each nvar separately)
            # Shape: (bs, nvars)
            similarity = (zs_project * zs_tilde).sum(dim=-1)

            # Sum over all (batch_size * nvars), then normalize by total count
            loss = -similarity.sum() / similarity.numel()

        return loss

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        head_type = getattr(self.args, 'head_type', 'flatten')
        num_quantiles = getattr(self.args, 'num_quantiles', 20)

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            # PatchTST: returns only output; PatchTST_REPA: returns (output, zs)
                            outputs = self.model(batch_x)
                            if isinstance(outputs, tuple):
                                outputs = outputs[0]
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        # PatchTST: returns only output; PatchTST_REPA: returns (output, zs)
                        outputs = self.model(batch_x)
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # Handle quantile head output
                if head_type == 'quantile':
                    # outputs: (bs, pred_len, nvars, num_quantiles)
                    # Select median quantile (q=0.5) for validation
                    mid_quantile = num_quantiles // 2
                    outputs = outputs[:, :, :, mid_quantile]  # (bs, pred_len, nvars)
                    # Already in (bs, pred_len, nvars) format

                f_dim = 0 if self.args.features == 'M' else -1
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                # Use MSE for validation and test (regardless of head_type)
                loss = nn.MSELoss()(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        # Print parameter statistics
        self._print_parameter_stats()

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        time_now = time.time()

        # Record loss per step for plotting
        loss_per_step = []
        loss_mse_per_step = []
        loss_contrastive_per_step = []

        train_steps = len(train_loader)
        best_val_loss = float('inf')  # Track best validation loss
        best_model_state = None  # Save best model state for test
        no_improve_count = 0  # Early stopping counter

        model_optim = self._select_optimizer()
        criterion = self._select_criterion().to(self.device)  # Move criterion to same device as model

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            train_mse_loss = []
            train_contrastive_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            # Slice target to pred_len for feature extraction
                            batch_y_pred = batch_y[:, -self.args.pred_len:, :]
                            if self.args.model in ['PatchTST_REPA', 'PatchTST_REPA_Fusion']:
                                outputs, _, _ = self.model(batch_x, batch_y_pred, return_projector=True)  # Get final output + features
                            else:
                                outputs = self.model(batch_x, batch_y_pred)  # Original PatchTST: returns only output
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = 0 if self.args.features == 'M' else -1
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                        # Record loss per step
                        loss_per_step.append(loss.item())
                        loss_mse_per_step.append(loss.item())  # Same as total in AMP mode
                        loss_contrastive_per_step.append(0.0)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        # Slice target to pred_len for feature extraction
                        batch_y_for_model = batch_y[:, -self.args.pred_len:, :]

                        # When using Chronos with PatchTST_REPA (not Fusion), interpolate to seq_len length to keep patch_num consistent
                        # PatchTST_REPA_Fusion uses Channel Fusion MLP which handles patch_num conversion automatically
                        if self.args.model == 'PatchTST_REPA' and getattr(self.args, 'feature_extractor', None) == 'chronos':
                            batch_y_for_model = F.interpolate(
                                batch_y_for_model.permute(0, 2, 1),  # (bs, nvars, pred_len)
                                size=self.args.seq_len,
                                mode='linear',
                                align_corners=False
                            ).permute(0, 2, 1)  # (bs, seq_len, nvars)

                        if self.args.model in ['PatchTST_REPA', 'PatchTST_REPA_Fusion']:
                            outputs, zs_project, zs_tilde = self.model(batch_x, batch_y_for_model, return_projector=True)  # Get final output + projected features + TiViT features
                        else:
                            outputs = self.model(batch_x, batch_y_for_model)  # Original PatchTST: returns only output
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)

                    # Handle quantile head output
                    head_type = getattr(self.args, 'head_type', 'flatten')
                    num_quantiles = getattr(self.args, 'num_quantiles', 20)

                    # print(outputs.shape,batch_y.shape)
                    f_dim = 0 if self.args.features == 'M' else -1
                    outputs = outputs[:, -self.args.pred_len:, f_dim:] if head_type != 'quantile' else outputs
                    batch_y_pred = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    # Loss computation
                    if head_type == 'quantile':
                        # outputs: (bs, pred_len, nvars, num_quantiles)
                        # QuantileLoss expects: pred (bs, nvars, num_quantiles, pred_len), target (bs, pred_len, nvars)
                        # Permute outputs to match: (bs, nvars, num_quantiles, pred_len)
                        outputs_for_quantile = outputs.permute(0, 2, 3, 1)  # (bs, nvars, num_quantiles, pred_len)
                        # batch_y_pred: (bs, pred_len, nvars) already matches target shape
                        mse_loss = criterion(outputs_for_quantile, batch_y_pred)
                    else:
                        mse_loss = criterion(outputs, batch_y_pred)

                    # Contrastive loss for feature alignment (only when using REPA model)
                    if self.args.model in ['PatchTST_REPA', 'PatchTST_REPA_Fusion']:
                        lambda_loss = self.args.lambda_contrastive
                        contrastive_loss = self._compute_contrastive_loss(zs_project, zs_tilde)
                        loss = mse_loss + lambda_loss * contrastive_loss
                    else:
                        contrastive_loss = torch.tensor(0.0, device=self.device)
                        loss = mse_loss

                    train_loss.append(loss.item())
                    train_mse_loss.append(mse_loss.item())
                    train_contrastive_loss.append(contrastive_loss.item())

                    # Record loss per step
                    loss_per_step.append(loss.item())
                    loss_mse_per_step.append(mse_loss.item())
                    loss_contrastive_per_step.append(contrastive_loss.item())

                if (i + 1) % 100 == 0:
                    # Only print detailed loss for PatchTST models with contrastive loss
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f} | mse: {3:.7f} | contrastive: {4:.7f}".format(
                            i + 1, epoch + 1, loss.item(), mse_loss.item(), contrastive_loss.item()))
                    else:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                    
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            cost_time = time.time() - epoch_time
            train_loss = np.average(train_loss)
            train_mse_loss = np.average(train_mse_loss)
            train_contrastive_loss = np.average(train_contrastive_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            # Check if best model updated (after vali_loss is computed)
            is_best_update = vali_loss < best_val_loss
            if is_best_update:
                best_val_loss = vali_loss
                # Save best model state for later test
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            # Get current learning rate
            current_lr = scheduler.get_last_lr()[0] if self.args.lradj == 'TST' else model_optim.param_groups[0]['lr']

            # Format: *** at the end if best model updated
            best_suffix = ' ***' if is_best_update else ''
            print("Epoch: {} | cost time: {:.3f} | lr: {:.4e} | Steps: {} | Train Loss: {:.7f} | Train MSE: {:.7f} | Train Contrastive: {:.7f} | Vali Loss: {:.7f} | Test Loss: {:.7f}{}".format(
                epoch + 1, cost_time, current_lr, train_steps, train_loss, train_mse_loss, train_contrastive_loss, vali_loss, test_loss, best_suffix))

            # Early stopping check
            if vali_loss >= best_val_loss:
                no_improve_count += 1
                if no_improve_count >= self.args.patience:
                    print("Early stopping")
                    break
            else:
                no_improve_count = 0

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)

        # Load best model for test
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        # Save loss per step for plotting (in results folder)
        results_folder = './results/' + setting + '/'
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        np.savez(results_folder + 'loss_per_step.npz',
                  steps=np.arange(len(loss_per_step)),
                  loss=np.array(loss_per_step),
                  loss_mse=np.array(loss_mse_per_step),
                  loss_contrastive=np.array(loss_contrastive_per_step))
        print(f"Loss curve saved to {results_folder}loss_per_step.npz")

        # Plot and save loss curves
        steps = np.arange(len(loss_per_step))
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].plot(steps, loss_per_step, 'b-', linewidth=0.5)
        axes[0].set_xlabel('Steps')
        axes[0].set_ylabel('Total Loss')
        axes[0].set_title('Total Loss')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(steps, loss_mse_per_step, 'g-', linewidth=0.5)
        axes[1].set_xlabel('Steps')
        axes[1].set_ylabel('MSE Loss')
        axes[1].set_title('MSE Loss')
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(steps, loss_contrastive_per_step, 'r-', linewidth=0.5)
        axes[2].set_xlabel('Steps')
        axes[2].set_ylabel('Contrastive Loss')
        axes[2].set_title('Contrastive Loss')
        axes[2].grid(True, alpha=0.3)

        plt.suptitle(f'Loss Curves - {setting}')
        plt.tight_layout()
        plt.savefig(results_folder + 'loss_curve.png', dpi=150)
        plt.close()
        print(f"Loss curve plot saved to {results_folder}loss_curve.png")

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        head_type = getattr(self.args, 'head_type', 'flatten')
        num_quantiles = getattr(self.args, 'num_quantiles', 20)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            # PatchTST: returns only output; PatchTST_REPA: returns (output, zs)
                            outputs = self.model(batch_x)
                            if isinstance(outputs, tuple):
                                outputs = outputs[0]
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        # PatchTST: returns only output; PatchTST_REPA: returns (output, zs)
                        outputs = self.model(batch_x)
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # Handle quantile head output
                if head_type == 'quantile':
                    # outputs: (bs, pred_len, nvars, num_quantiles)
                    # Select median quantile (q=0.5) for output
                    mid_quantile = num_quantiles // 2
                    outputs = outputs[:, :, :, mid_quantile]  # (bs, pred_len, nvars)

                f_dim = 0 if self.args.features == 'M' else -1
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            # PatchTST: returns only output; PatchTST_REPA: returns (output, zs)
                            outputs = self.model(batch_x)
                            if isinstance(outputs, tuple):
                                outputs = outputs[0]
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        # PatchTST: returns only output; PatchTST_REPA: returns (output, zs)
                        outputs = self.model(batch_x)
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
