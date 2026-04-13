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

    Two modes controlled by --alignment:
    - Distillation mode (alignment=1): loads Chronos2 teacher for alignment
    - Standalone mode (alignment=0): pure encoder → head, no Chronos2 needed

    Training (distillation mode):
        Path A (Teacher):
            x_past → Chronos2.embed (frozen) → past tokens (patch_num, 768)
                   → proj_down (768→d_model, trainable)
                   → z_teacher → teacher_head → pred_teacher

        Path B (Student):
            x_past → Encoder (trainable) → z_enc → head → pred_student

        Loss = MSE(pred_student, y)                         # Loss①
             + λ_t * MSE(pred_teacher, y)                   # Loss②
             + λ_a * align(z_enc, z_teacher.detach())       # Loss③

    Patch settings: patch_len=16, stride=16, patch_num=seq_len//16
    Both student and teacher operate on the same patch grid as Chronos2 past tokens.

    Inference: only Path B — Chronos2 not needed.
    """

    def __init__(self, configs):
        super().__init__()

        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.device_str = getattr(configs, 'device', 'cuda:0')

        # Number of past patches (matches Chronos2 past token count)
        self.num_past_patches = configs.seq_len // 16

        # Check if we need teacher path (distillation mode)
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

        # ---- Backbone (encoder + proj_down + head) ----
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

    def train(self, mode=True):
        super().train(mode)
        # Keep Chronos2 always in eval mode regardless of model train/eval state
        if self.use_teacher and self.chronos is not None:
            self.chronos.model.eval()
        return self

    def forward(self, x_past, x_future=None):
        """
        Args:
            x_past:   (bs, seq_len, nvars)
            x_future: unused, kept for API compatibility

        Returns (training, use_teacher=True):
            pred_student: (bs, pred_len, nvars)
            pred_teacher: (bs, pred_len, nvars)
            z_enc:        (bs, nvars, patch_num, d_model)
            z_teacher:    (bs, nvars, patch_num, d_model)

        Returns (inference / use_teacher=False):
            pred_student: (bs, pred_len, nvars)

        Teacher signal: Chronos2.embed(x_past) past tokens — same patch grid as student.
        Both student and teacher only use x_past; no train/inference gap.
        """
        bs, seq_len, nvars = x_past.shape
        # (bs, seq_len, nvars) → (bs, nvars, seq_len)
        x_perm = x_past.permute(0, 2, 1)

        # Student path
        pred_s, z_enc = self.backbone.forward_student(x_perm)
        pred_student = pred_s.permute(0, 2, 1)  # (bs, pred_len, nvars)

        if self.use_teacher and self.training:
            # Teacher path: Chronos2.embed(x_past) → past tokens
            # x_perm: (bs, nvars, seq_len) — embed() accepts 3D (batch, n_variates, history)
            embeddings_list, loc_scales = self.chronos.embed(x_perm.cpu())
            # embeddings_list: list of (nvars, num_past+2, 768), length=bs
            # Take only past tokens (exclude REG token and masked future token at the end)
            z_chron = torch.stack(
                [emb[:, :self.num_past_patches, :] for emb in embeddings_list], dim=0
            ).to(x_past.device)
            # z_chron: (bs, nvars, num_past_patches, 768)

            # loc/scale from Chronos2 embed (x_past mean/std, consistent with student RevIN)
            loc   = torch.stack([ls[0] for ls in loc_scales], dim=0).squeeze(-1).to(x_past.device)
            scale = torch.stack([ls[1] for ls in loc_scales], dim=0).squeeze(-1).to(x_past.device)

            pred_t, z_teacher = self.backbone.forward_teacher(z_chron, loc=loc, scale=scale)
            pred_teacher = pred_t.permute(0, 2, 1)  # (bs, pred_len, nvars)

            return pred_student, pred_teacher, z_enc, z_teacher

        # Student only (no teacher / no distillation)
        return pred_student
