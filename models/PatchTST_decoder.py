__all__ = ['Model']

import torch
from torch import nn

try:
    from chronos import Chronos2Pipeline
    HAS_CHRONOS = True
except ImportError:
    HAS_CHRONOS = False

from layers.PatchTST_Decoder_backbone import PatchTST_Decoder_backbone


class Model(nn.Module):
    """PatchTST_decoder — FutureQueryDecoder with optional Chronos2 distillation.

    Controlled by --contrastive:
    - Distillation mode (contrastive=1): loads Chronos2 teacher for alignment
    - Standalone mode (contrastive=0): pure FutureQueryDecoder, no Chronos2 needed

    Difference from PatchTST_future_align:
        - A FutureQueryDecoder (cross-attention) sits between encoder and head.
        - Alignment target z_future (decoder output) is future-oriented,
          closing the gap with z_teacher (Chronos2 future embeddings).
        - PatchwiseHead is the natural choice: decoder patch i semantically
          maps to future segment i.

    Training paths (distillation mode):
        Student: x_past → Encoder → FutureQueryDecoder → Head → pred_s
        Teacher: x_future → Chronos2 (frozen) → proj_down → teacher_head → pred_t

        Loss = MSE(pred_s, y)                              # Loss①
             + λ_t  * MSE(pred_t, y)                      # Loss② (warmup only / reduced in phase 2)
             + λ_a  * (cosine + MSE)(z_future, z_teacher)  # Loss③

    Inference: Chronos2 not needed.
    """

    def __init__(self, configs):
        super().__init__()

        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.device_str = getattr(configs, 'device', 'cuda:0')

        # Number of future patches (pred_len // 16)
        self.num_output_patches = configs.pred_len // 16

        # Check if we need teacher path (distillation mode)
        # Use --alignment to control: 1=enable distillation (load Chronos2), 0=standalone mode
        user_alignment = getattr(configs, 'alignment', None)
        self.use_teacher = False if user_alignment is not None and user_alignment == 0 else True

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

        # ---- Backbone (encoder + FutureQueryDecoder + proj_down + heads) ----
        self.backbone = PatchTST_Decoder_backbone(
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
            head_type=getattr(configs, 'head_type', 'patch_wise'),
            decoder_layers=getattr(configs, 'decoder_layers', 1),
        )

    def forward(self, x_past, x_future=None):
        """
        Args:
            x_past:   (bs, seq_len,  nvars)
            x_future: (bs, pred_len, nvars)  — only needed during training

        Returns (training):
            pred_student: (bs, pred_len, nvars)
            pred_teacher: (bs, pred_len, nvars)
            z_future:     (bs, nvars, output_patch_num, d_model)
            z_teacher:    (bs, nvars, output_patch_num, d_model)

        Returns (inference):
            pred_student: (bs, pred_len, nvars)
        """
        # (bs, seq_len, nvars) → (bs, nvars, seq_len)
        x_perm = x_past.permute(0, 2, 1)

        # Student path
        pred_s, z_future = self.backbone.forward_student(x_perm)
        pred_student = pred_s.permute(0, 2, 1)  # (bs, pred_len, nvars)

        if x_future is not None and self.use_teacher:
            # Teacher path: embed ground-truth future with frozen Chronos2
            future_perm = x_future.permute(0, 2, 1)  # (bs, nvars, pred_len)

            embeddings_list, loc_scales = self.chronos.embed(future_perm.cpu())
            # embeddings_list: list of length bs, each (nvars, num_tokens+2, 768)
            z_chron = torch.stack(embeddings_list, dim=0).to(x_past.device)
            # z_chron: (bs, nvars, num_tokens+2, 768)
            z_chron = z_chron[:, :, :self.num_output_patches, :]
            # z_chron: (bs, nvars, output_patch_num, 768)

            # x_future loc/scale from Chronos2 (for teacher path denorm)
            loc   = torch.stack([ls[0] for ls in loc_scales], dim=0).squeeze(-1).to(x_past.device)  # (bs, nvars)
            scale = torch.stack([ls[1] for ls in loc_scales], dim=0).squeeze(-1).to(x_past.device)  # (bs, nvars)

            pred_t, z_teacher = self.backbone.forward_teacher(z_chron, loc=loc, scale=scale)
            pred_teacher = pred_t.permute(0, 2, 1)  # (bs, pred_len, nvars)

            return pred_student, pred_teacher, z_future, z_teacher

        # Student only (no teacher / no distillation)
        return pred_student
