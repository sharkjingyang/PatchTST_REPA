__all__ = ['Model']

import torch
from torch import nn

try:
    from chronos import Chronos2Pipeline
    HAS_CHRONOS = True
except ImportError:
    HAS_CHRONOS = False

from layers.PatchTST_FutureAlign_backbone import PatchTST_FutureAlign_backbone


class Model(nn.Module):
    """Joint Distillation Training model (PatchTST_future_align).

    Training: two paths share the same Head and RevIN.

        Path A (Teacher):
            x_future → Chronos2 (frozen) → z_chron
                     → proj_down (768→d_model, trainable)
                     → z_teacher → Head → pred_teacher

        Path B (Student):
            x_past → Encoder (trainable) → z_enc → Head → pred_student

        Loss = MSE(pred_student, y)                         # Loss①
             + λ_t * MSE(pred_teacher, y)                   # Loss②
             + λ_a * MSE(z_enc, z_teacher.detach())         # Loss③

    Inference: only Path B — Chronos2 not needed.
    """

    def __init__(self, configs):
        super().__init__()

        if not HAS_CHRONOS:
            raise ImportError(
                "chronos is not installed. "
                "Please install it with: pip install chronos-forecasting"
            )

        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.device_str = getattr(configs, 'device', 'cuda:0')

        # Number of future patches Chronos2 embeds
        self.num_output_patches = configs.pred_len // 16

        # ---- Chronos2 (frozen) ----
        chronos_pretrained = getattr(configs, 'chronos_pretrained', './Chronos2')
        self.chronos = Chronos2Pipeline.from_pretrained(
            chronos_pretrained, device_map=self.device_str
        )
        self.add_module('chronos_model', self.chronos.model)
        self.chronos.model.eval()
        for param in self.chronos.model.parameters():
            param.requires_grad = False

        # ---- Backbone (encoder + proj_down + shared head) ----
        self.backbone = PatchTST_FutureAlign_backbone(
            c_in=configs.enc_in,
            context_window=configs.seq_len,
            target_window=configs.pred_len,
            n_layers=configs.e_layers,
            d_model=configs.d_model,
            n_heads=configs.n_heads,
            d_ff=configs.d_ff,
            dropout=configs.dropout,
            head_dropout=getattr(configs, 'head_dropout', 0.0),
            individual=getattr(configs, 'individual', 0),
            revin=getattr(configs, 'revin', 1),
            affine=getattr(configs, 'affine', 0),
            subtract_last=getattr(configs, 'subtract_last', 0),
            head_type=getattr(configs, 'head_type', 'flatten'),
        )

    def forward(self, x_past, x_future=None):
        """
        Args:
            x_past:   (bs, seq_len,  nvars)
            x_future: (bs, pred_len, nvars)  — only needed during training

        Returns (training):
            pred_student: (bs, pred_len, nvars)
            pred_teacher: (bs, pred_len, nvars)
            z_enc:        (bs, nvars, output_patch_num, d_model)
            z_teacher:    (bs, nvars, output_patch_num, d_model)

        Returns (inference):
            pred_student: (bs, pred_len, nvars)
        """
        # (bs, seq_len, nvars) → (bs, nvars, seq_len)
        x_perm = x_past.permute(0, 2, 1)

        # Student path
        pred_s, z_enc = self.backbone.forward_student(x_perm)
        pred_student = pred_s.permute(0, 2, 1)  # (bs, pred_len, nvars)

        if x_future is not None:
            # Teacher path: embed ground-truth future with frozen Chronos2
            future_perm = x_future.permute(0, 2, 1)  # (bs, nvars, pred_len)

            embeddings_list, _ = self.chronos.embed(future_perm.cpu())
            # embeddings_list: list of length bs, each (nvars, num_tokens+2, 768)
            z_chron = torch.stack(embeddings_list, dim=0).to(x_past.device)
            # z_chron: (bs, nvars, num_tokens+2, 768)
            z_chron = z_chron[:, :, :self.num_output_patches, :]
            # z_chron: (bs, nvars, output_patch_num, 768)

            pred_t, z_teacher = self.backbone.forward_teacher(z_chron)
            pred_teacher = pred_t.permute(0, 2, 1)  # (bs, pred_len, nvars)

            return pred_student, pred_teacher, z_enc, z_teacher

        return pred_student
