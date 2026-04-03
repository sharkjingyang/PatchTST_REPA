__all__ = ['PatchTST_FutureAlign_backbone']

from typing import Optional
import torch
from torch import nn, Tensor

from layers.PatchTST_backbone import TSTiEncoder, Flatten_Head, PatchwiseHead
from layers.RevIN import RevIN


class PatchTST_FutureAlign_backbone(nn.Module):
    """Backbone for PatchTST_future_align.

    Architecture:
        - Student path: x_past → RevIN → patch → TSTiEncoder → Head → RevIN denorm
        - Teacher path: z_chron (from Chronos2) → proj_down → Head → RevIN denorm
          (reuses RevIN stats stored by forward_student)
        - head_type: 'flatten' (Flatten_Head) or 'patch_wise' (PatchwiseHead)
          Both student and teacher use the same head type.

    Patch auto-derivation:
        output_patch_num = pred_len // 16  (Chronos2 patch_size = 16)
        patch_len = seq_len // output_patch_num
        stride = patch_len  (no overlap, patch_num == output_patch_num)
    """

    def __init__(self, c_in: int, context_window: int, target_window: int,
                 n_layers: int = 3, d_model: int = 128, n_heads: int = 16,
                 d_k: Optional[int] = None, d_v: Optional[int] = None,
                 d_ff: int = 256, norm: str = 'BatchNorm',
                 attn_dropout: float = 0., dropout: float = 0.,
                 act: str = 'gelu', key_padding_mask: bool = 'auto',
                 padding_var: Optional[int] = None,
                 attn_mask: Optional[Tensor] = None,
                 res_attention: bool = True, pre_norm: bool = False,
                 store_attn: bool = False, pe: str = 'zeros',
                 learn_pe: bool = True, head_dropout: float = 0.,
                 individual: bool = False, revin: bool = True,
                 affine: bool = True, subtract_last: bool = False,
                 head_type: str = 'flatten',
                 max_seq_len: int = 1024, verbose: bool = False, **kwargs):
        super().__init__()

        # Chronos2 patch size is fixed at 16
        chronos_patch_size = 16
        assert target_window % chronos_patch_size == 0, (
            f"pred_len must be divisible by {chronos_patch_size} (Chronos2 patch size), "
            f"got pred_len={target_window}"
        )
        self.output_patch_num = target_window // chronos_patch_size

        assert context_window % self.output_patch_num == 0, (
            f"seq_len ({context_window}) must be divisible by output_patch_num "
            f"({self.output_patch_num}). "
            f"Try seq_len=336 with pred_len=96 (output_patch_num=6, patch_len=56)."
        )
        patch_len = context_window // self.output_patch_num
        stride = patch_len  # no overlap → patch_num == output_patch_num
        patch_num = self.output_patch_num

        self.patch_len = patch_len
        self.stride = stride
        self.patch_num = patch_num
        self.d_model = d_model
        self.n_vars = c_in

        # RevIN — shared stats between student and teacher paths
        self.revin = revin
        if revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # Encoder (student path only)
        self.backbone = TSTiEncoder(
            c_in, patch_num=patch_num, patch_len=patch_len,
            max_seq_len=max_seq_len,
            n_layers=n_layers, d_model=d_model, n_heads=n_heads,
            d_k=d_k, d_v=d_v, d_ff=d_ff,
            attn_dropout=attn_dropout, dropout=dropout, act=act,
            key_padding_mask=key_padding_mask, padding_var=padding_var,
            attn_mask=attn_mask, res_attention=res_attention,
            pre_norm=pre_norm, store_attn=store_attn,
            pe=pe, learn_pe=learn_pe,
            encoder_depth=n_layers,  # use all layers; no intermediate needed
            verbose=verbose, **kwargs
        )

        # proj_down: Chronos2 dim (768) → d_model (teacher path, trainable)
        self.proj_down = nn.Linear(768, d_model)

        self.head_type = head_type

        def _build_head():
            if head_type == 'patch_wise':
                return PatchwiseHead(
                    n_vars=c_in,
                    d_model=d_model,
                    output_patch_num=patch_num,
                    output_patch_size=16,
                    dropout=head_dropout
                )
            else:  # flatten
                return Flatten_Head(
                    individual=individual,
                    n_vars=c_in,
                    nf=d_model * patch_num,
                    target_window=target_window,
                    head_dropout=head_dropout
                )

        # Student and teacher use the same head type (independent weights)
        self.head = _build_head()
        self.teacher_head = _build_head()

    # ------------------------------------------------------------------
    # Student path
    # ------------------------------------------------------------------
    def forward_student(self, x):
        """
        Args:
            x: (bs, nvars, seq_len)
        Returns:
            pred:  (bs, nvars, pred_len)  — denormalized prediction
            z_enc: (bs, nvars, patch_num, d_model)  — for alignment loss
        """
        # RevIN norm (stores loc/scale internally for denorm)
        if self.revin:
            x = x.permute(0, 2, 1)
            x = self.revin_layer(x, 'norm')
            x = x.permute(0, 2, 1)

        # Patching — no padding needed (seq_len exactly divisible)
        z = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        z = z.permute(0, 1, 3, 2)  # (bs, nvars, patch_len, patch_num)

        # Encode
        z, _ = self.backbone(z, return_intermediate=False)
        # z: (bs, nvars, d_model, patch_num)

        # Save for alignment loss: (bs, nvars, patch_num, d_model)
        z_enc = z.permute(0, 1, 3, 2)

        # Head
        pred = self.head(z)  # (bs, nvars, pred_len)

        # RevIN denorm
        if self.revin:
            pred = pred.permute(0, 2, 1)
            pred = self.revin_layer(pred, 'denorm')
            pred = pred.permute(0, 2, 1)

        return pred, z_enc

    # ------------------------------------------------------------------
    # Teacher path
    # ------------------------------------------------------------------
    def forward_teacher(self, z_chron):
        """
        Must be called after forward_student (reuses RevIN stats).

        Args:
            z_chron: (bs, nvars, output_patch_num, 768)  — Chronos2 future embeddings
        Returns:
            pred:     (bs, nvars, pred_len)
            z_teacher: (bs, nvars, output_patch_num, d_model)  — for alignment loss
        """
        # proj_down: 768 → d_model
        z_teacher = self.proj_down(z_chron)  # (bs, nvars, P, d_model)

        # Reshape for Flatten_Head: (bs, nvars, d_model, P)
        z_perm = z_teacher.permute(0, 1, 3, 2)

        # Teacher head (independent from student head)
        pred = self.teacher_head(z_perm)  # (bs, nvars, pred_len)

        # RevIN denorm (reuses stats from forward_student)
        if self.revin:
            pred = pred.permute(0, 2, 1)
            pred = self.revin_layer(pred, 'denorm')
            pred = pred.permute(0, 2, 1)

        return pred, z_teacher

    # ------------------------------------------------------------------
    # Inference — student only
    # ------------------------------------------------------------------
    def forward(self, x):
        """
        Args:
            x: (bs, nvars, seq_len)
        Returns:
            pred: (bs, nvars, pred_len)
        """
        pred, _ = self.forward_student(x)
        return pred
