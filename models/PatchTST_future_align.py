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

    Two modes controlled by --contrastive:
    - Distillation mode (contrastive=1): loads Chronos2 teacher for alignment
    - Standalone mode (contrastive=0): pure encoder → head, no Chronos2 needed

    Training (distillation mode):
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

        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.device_str = getattr(configs, 'device', 'cuda:0')

        # Number of future patches
        self.num_output_patches = configs.pred_len // 16

        # Check if we need teacher path (distillation mode)
        # Use --alignment to control: 1=enable distillation (load Chronos2), 0=standalone mode
        user_alignment = getattr(configs, 'alignment', None)
        self.use_teacher = 0 if user_alignment is not None and user_alignment == 0 else 1

        # ---- Chronos2 (frozen) ---- only load when needed
        if self.use_teacher:
            if not HAS_CHRONOS:
                raise ImportError(
                    "chronos is not installed. "
                    "Please install it with: pip install chronos-forecasting"
                )
            chronos_pretrained = getattr(configs, 'chronos_pretrained', './Chronos2')
            self.chronos = Chronos2Pipeline.from_pretrained(
                chronos_pretrained, device_map=self.device_str
            )
            self.add_module('chronos_model', self.chronos.model)
            self.chronos.model.eval()
            for param in self.chronos.model.parameters():
                param.requires_grad = False
        else:
            self.chronos = None

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
            x_future: unused, kept for API compatibility

        Returns (training, use_teacher=True):
            pred_student: (bs, pred_len, nvars)
            pred_teacher: (bs, pred_len, nvars)
            z_enc:        (bs, nvars, output_patch_num, d_model)
            z_teacher:    (bs, nvars, output_patch_num, d_model)

        Returns (inference / use_teacher=False):
            pred_student: (bs, pred_len, nvars)

        Teacher signal: Chronos2.model.encode(x_past, num_output_patches) future tokens.
        No x_future needed — train/inference consistent, both only see x_past.
        """
        bs, seq_len, nvars = x_past.shape
        # (bs, seq_len, nvars) → (bs, nvars, seq_len)
        x_perm = x_past.permute(0, 2, 1)

        # Student path
        pred_s, z_enc = self.backbone.forward_student(x_perm)
        pred_student = pred_s.permute(0, 2, 1)  # (bs, pred_len, nvars)

        if self.use_teacher and self.training:
            # Teacher path: Chronos2.model.encode(x_past) → future tokens
            # x_past only — no ground-truth future needed, consistent with inference
            x_flat = x_perm.reshape(bs * nvars, seq_len).float()

            encoder_out, loc_scale, _, _ = self.chronos.model.encode(
                context=x_flat.to(self.chronos.model.device),
                num_output_patches=self.num_output_patches,
            )
            # last num_output_patches tokens: (bs*nvars, num_output_patches, 768)
            future_hidden = encoder_out.last_hidden_state[:, -self.num_output_patches:, :]
            z_chron = future_hidden.reshape(bs, nvars, self.num_output_patches, 768).to(x_past.device)

            # loc/scale from Chronos2 encode (x_past stats, consistent with student RevIN)
            loc   = loc_scale[0].reshape(bs, nvars).to(x_past.device)
            scale = loc_scale[1].reshape(bs, nvars).to(x_past.device)

            pred_t, z_teacher = self.backbone.forward_teacher(z_chron, loc=loc, scale=scale)
            pred_teacher = pred_t.permute(0, 2, 1)  # (bs, pred_len, nvars)

            return pred_student, pred_teacher, z_enc, z_teacher

        # Student only (no teacher / no distillation)
        return pred_student
