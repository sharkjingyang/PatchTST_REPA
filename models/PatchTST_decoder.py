__all__ = ['Model']

import torch
from torch import nn

try:
    from chronos import Chronos2Pipeline
    HAS_CHRONOS = True
except ImportError:
    HAS_CHRONOS = False

from layers.MiniChronos2_backbone import MiniChronos2_backbone


class Model(nn.Module):
    """PatchTST_decoder — MiniChronos2 student with optional Chronos2 teacher distillation.

    Student: small Chronos2-style encoder (128-dim, channel-independent)
        Input: [historical patches | masked future patches] with Chronos2-style time encoding
        Architecture: InstanceNorm → Patch(16) → input_proj → TransformerEncoder
        Output: last num_output_patches hidden states → PatchwiseHead → prediction

    Teacher (alignment=1): frozen large Chronos2 encodes x_past → future tokens (768-dim)
        → proj_down(768→d_model) → teacher_head → pred_teacher
        Both student and teacher use ONLY x_past (no ground-truth future needed).
        Time-encoding semantics match: both produce future-token representations from past.
        Alignment signal is clean — no past/future distribution mismatch.
        Teacher denorm uses Chronos2's own loc_scale from encode() (consistent with Chronos2_head predict mode).

    Training losses:
        Loss①  = MSE(pred_student, y)
        Loss②  = λ_t  * MSE(pred_teacher, y)           [warmup phase]
               = λ_t2 * MSE(pred_teacher, y)           [align phase]
        Loss③  = λ_a  * (cosine + MSE)(z_student, z_teacher)  [align phase]

    Inference: only student path, Chronos2 not loaded / not called.

    Difference from PatchTST_future_align:
        - Student uses Chronos2-style architecture (time encoding + masked future tokens)
          instead of TSTiEncoder + FutureQueryDecoder (cross-attention)
        - Teacher uses encode(x_past) instead of embed(x_future): no GT future needed
        - Both student/teacher future tokens have consistent positive time encoding
    """

    def __init__(self, configs):
        super().__init__()

        self.pred_len = configs.pred_len
        self.seq_len  = configs.seq_len
        self.device_str = getattr(configs, 'device', 'cuda:0')
        self.num_output_patches = configs.pred_len // 16

        # alignment=0 → standalone mode (no Chronos2)
        user_alignment = getattr(configs, 'alignment', None)
        self.use_teacher = not (user_alignment is not None and user_alignment == 0)

        # ---- Chronos2 (frozen) ---- only load when distillation is enabled
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

        # ---- MiniChronos2 backbone (student + teacher heads) ----
        self.backbone = MiniChronos2_backbone(
            c_in=configs.enc_in,
            context_window=configs.seq_len,
            target_window=configs.pred_len,
            n_layers=configs.e_layers,
            d_model=configs.d_model,
            n_heads=configs.n_heads,
            d_ff=configs.d_ff,
            dropout=configs.dropout,
            head_dropout=getattr(configs, 'head_dropout', 0.0),
            head_type=getattr(configs, 'head_type', 'patch_wise'),
        )

    def train(self, mode=True):
        super().train(mode)
        if self.use_teacher and self.chronos is not None:
            self.chronos.model.eval()
        return self

    def forward(self, x_past: torch.Tensor):
        """
        Args:
            x_past: (bs, seq_len, nvars)

        Returns (training, use_teacher=True):
            pred_student: (bs, pred_len, nvars)
            pred_teacher: (bs, pred_len, nvars)
            z_student:    (bs, nvars, output_patch_num, d_model)
            z_teacher:    (bs, nvars, output_patch_num, d_model)

        Returns (inference or use_teacher=False):
            pred_student: (bs, pred_len, nvars)
        """
        x_perm = x_past.permute(0, 2, 1)  # (bs, nvars, seq_len)

        # Student path — only x_past needed
        pred_s, z_student, loc_scale = self.backbone.forward_student(x_perm)
        pred_student = pred_s.permute(0, 2, 1)  # (bs, pred_len, nvars)

        if self.use_teacher and self.training:
            bs, nvars, seq_len = x_perm.shape
            x_flat = x_perm.reshape(bs * nvars, seq_len).float()

            # Teacher: frozen Chronos2 encodes x_past → future token representations
            # encode() returns (encoder_outputs, loc_scale, mask, num_ctx_patches)
            # last_hidden_state shape: (bs*nvars, num_ctx+1+num_out, 768)
            chronos_device = next(self.chronos.model.parameters()).device
            with torch.no_grad():
                enc_out, t_loc_scale, _, _ = self.chronos.model.encode(
                    context=x_flat.to(chronos_device),
                    num_output_patches=self.num_output_patches,
                )

            # Take last num_output_patches tokens as future representations
            z_chron_flat = enc_out.last_hidden_state[:, -self.num_output_patches:, :]
            z_chron_flat = z_chron_flat.to(x_past.device)
            z_chron = z_chron_flat.reshape(bs, nvars, self.num_output_patches, 768)

            # Teacher head: use Chronos2's own loc_scale (consistent with encode path)
            t_loc_scale = (t_loc_scale[0].to(x_past.device), t_loc_scale[1].to(x_past.device))
            pred_t, z_teacher = self.backbone.forward_teacher(z_chron, t_loc_scale)
            pred_teacher = pred_t.permute(0, 2, 1)  # (bs, pred_len, nvars)

            return pred_student, pred_teacher, z_student, z_teacher

        return pred_student
