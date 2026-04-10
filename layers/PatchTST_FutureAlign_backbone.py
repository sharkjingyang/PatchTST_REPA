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
        - Teacher path: z_chron (Chronos2.embed past tokens) → proj_down → Head → denorm with x_past loc/scale
        - head_type: 'flatten' (Flatten_Head) recommended; 'patch_wise' requires pred_len % patch_num == 0

    Patch settings (fixed to Chronos2 native):
        patch_len = 16
        stride    = 16  (no overlap)
        patch_num = seq_len // 16  (e.g., 336//16=21, matches Chronos2 past token count)
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

        # Patch settings fixed to Chronos2 native patch_size=16
        patch_len = 16
        stride = 16
        assert context_window % patch_len == 0, (
            f"seq_len ({context_window}) must be divisible by 16 (Chronos2 patch size)"
        )
        patch_num = context_window // patch_len  # e.g., 336//16=21

        self.patch_len = patch_len
        self.stride = stride
        self.patch_num = patch_num
        self.d_model = d_model
        self.n_vars = c_in

        # RevIN
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
            encoder_depth=n_layers,
            verbose=verbose, **kwargs
        )

        # proj_down: Chronos2 dim (768) → d_model (teacher path, trainable)
        self.proj_down = nn.Linear(768, d_model)

        self.head_type = head_type

        def _build_head():
            if head_type == 'patch_wise':
                assert target_window % patch_num == 0, (
                    f"pred_len ({target_window}) must be divisible by patch_num ({patch_num}) "
                    f"for patch_wise head"
                )
                return PatchwiseHead(
                    n_vars=c_in,
                    d_model=d_model,
                    output_patch_num=patch_num,
                    output_patch_size=target_window // patch_num,
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
        # RevIN norm
        if self.revin:
            x = x.permute(0, 2, 1)
            x = self.revin_layer(x, 'norm')
            x = x.permute(0, 2, 1)

        # Patching: (bs, nvars, seq_len) → (bs, nvars, patch_num, patch_len)
        z = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        z = z.permute(0, 1, 3, 2)  # (bs, nvars, patch_len, patch_num)

        # Encode → (bs, nvars, d_model, patch_num)
        z, _ = self.backbone(z, return_intermediate=False)

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
    def forward_teacher(self, z_chron, loc=None, scale=None):
        """
        Args:
            z_chron: (bs, nvars, patch_num, 768)  — Chronos2 past token embeddings
            loc:     (bs, nvars)  — x_past mean from Chronos2.embed
            scale:   (bs, nvars)  — x_past std  from Chronos2.embed
                     If None, falls back to RevIN stats from forward_student.
        Returns:
            pred:      (bs, nvars, pred_len)
            z_teacher: (bs, nvars, patch_num, d_model)  — for alignment loss
        """
        # proj_down: 768 → d_model
        z_teacher = self.proj_down(z_chron)  # (bs, nvars, patch_num, d_model)

        # Reshape for head: (bs, nvars, d_model, patch_num)
        z_perm = z_teacher.permute(0, 1, 3, 2)

        # Teacher head
        pred = self.teacher_head(z_perm)  # (bs, nvars, pred_len)

        # Denorm
        if loc is not None and scale is not None:
            pred = pred * scale.unsqueeze(-1) + loc.unsqueeze(-1)
        elif self.revin:
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
