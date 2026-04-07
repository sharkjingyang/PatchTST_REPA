__all__ = ['FutureQueryDecoder', 'PatchTST_Decoder_backbone']

from typing import Optional
import torch
from torch import nn, Tensor

from layers.PatchTST_backbone import TSTiEncoder, Flatten_Head, PatchwiseHead
from layers.RevIN import RevIN


class FutureQueryDecoderLayer(nn.Module):
    """Single cross-attention layer for FutureQueryDecoder.

    Q = learned future queries
    K/V = past encoder tokens (z_enc)
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: Tensor, z_enc: Tensor) -> Tensor:
        """
        Args:
            queries: (bs*nvars, output_patch_num, d_model)  — Q
            z_enc:   (bs*nvars, patch_num, d_model)          — K / V
        Returns:
            (bs*nvars, output_patch_num, d_model)
        """
        attn_out, _ = self.cross_attn(queries, z_enc, z_enc)
        queries = self.norm1(queries + self.dropout(attn_out))
        queries = self.norm2(queries + self.dropout(self.ffn(queries)))
        return queries


class FutureQueryDecoder(nn.Module):
    """Cross-attention decoder that generates future-oriented patch representations.

    Maintains output_patch_num learnable query vectors that attend over
    past encoder tokens (z_enc) to produce z_future.

    Params (d_model=128, n_heads=8, d_ff=256, output_patch_num=6, n_layers=1): ~132K
    """
    def __init__(self, output_patch_num: int, d_model: int, n_heads: int,
                 d_ff: int, dropout: float = 0.0, n_layers: int = 1):
        super().__init__()
        self.future_queries = nn.Parameter(torch.randn(output_patch_num, d_model))
        self.layers = nn.ModuleList([
            FutureQueryDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, z_enc: Tensor) -> Tensor:
        """
        Args:
            z_enc: (bs*nvars, patch_num, d_model)
        Returns:
            z_future: (bs*nvars, output_patch_num, d_model)
        """
        queries = self.future_queries.unsqueeze(0).expand(z_enc.shape[0], -1, -1)
        for layer in self.layers:
            queries = layer(queries, z_enc)
        return queries


class PatchTST_Decoder_backbone(nn.Module):
    """Backbone for PatchTST_decoder.

    Architecture:
        Student path:  x_past → RevIN → patch → TSTiEncoder → FutureQueryDecoder → Head → RevIN denorm
        Teacher path:  z_chron (Chronos2 embeddings) → proj_down → teacher_head → denorm
                       (training only)

    Key difference from PatchTST_FutureAlign_backbone:
        - FutureQueryDecoder (cross-attention) between encoder and head
        - z_future (from decoder) aligns with z_teacher in d_model space
        - PatchwiseHead is the natural choice: decoder patch i semantically corresponds to future[i]

    Patch settings:
        output_patch_num = pred_len // 16
        patch_len = 16  (same as Chronos2 native patch size)
        stride = 16     (no overlap, patch_num = seq_len // 16)
    """

    def __init__(self, c_in: int, context_window: int, target_window: int,
                 n_layers: int = 3, d_model: int = 128, n_heads: int = 8,
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
                 head_type: str = 'patch_wise',
                 decoder_layers: int = 1,
                 max_seq_len: int = 1024, verbose: bool = False, **kwargs):
        super().__init__()

        # Patch settings: encoder uses patch_len=16, stride=16 (Chronos2 native)
        chronos_patch_size = 16
        assert target_window % chronos_patch_size == 0, (
            f"pred_len must be divisible by {chronos_patch_size} (Chronos2 patch size), "
            f"got pred_len={target_window}"
        )
        self.output_patch_num = target_window // chronos_patch_size

        patch_len = chronos_patch_size  # 16
        stride = chronos_patch_size     # 16
        patch_num = context_window // chronos_patch_size  # e.g., 336//16=21

        self.patch_len = patch_len
        self.stride = stride
        self.patch_num = patch_num
        self.d_model = d_model
        self.n_vars = c_in
        self.head_type = head_type

        # RevIN — shared stats
        self.revin = revin
        if revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # Encoder (full n_layers, no truncation)
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

        # FutureQueryDecoder — cross-attention to generate future-oriented representations
        # Maps patch_num (e.g., 21) past patches → output_patch_num (e.g., 6) future patches
        self.future_query_decoder = FutureQueryDecoder(
            output_patch_num=self.output_patch_num,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            n_layers=decoder_layers,
        )

        # proj_down: Chronos2 dim (768) → d_model (teacher path, trainable)
        self.proj_down = nn.Linear(768, d_model)

        def _build_head():
            if head_type == 'patch_wise':
                # PatchwiseHead: each output patch predicts output_patch_size steps
                return PatchwiseHead(
                    n_vars=c_in,
                    d_model=d_model,
                    output_patch_num=self.output_patch_num,  # e.g., 6 for pred_len=96
                    output_patch_size=16,
                    dropout=head_dropout,
                )
            else:  # flatten
                # Flatten_Head: uses decoder output patches (output_patch_num, not patch_num)
                return Flatten_Head(
                    individual=individual,
                    n_vars=c_in,
                    nf=d_model * self.output_patch_num,
                    target_window=target_window,
                    head_dropout=head_dropout,
                )

        # Student and teacher use the same head type (independent weights)
        self.head = _build_head()
        self.teacher_head = _build_head()

    # ------------------------------------------------------------------
    # Student path
    # ------------------------------------------------------------------
    def forward_student(self, x: Tensor):
        """
        Args:
            x: (bs, nvars, seq_len)
        Returns:
            pred:     (bs, nvars, pred_len)
            z_future: (bs, nvars, output_patch_num, d_model)  — for alignment loss
        """
        # RevIN norm
        if self.revin:
            x = x.permute(0, 2, 1)
            x = self.revin_layer(x, 'norm')
            x = x.permute(0, 2, 1)

        # Patching (no padding needed — seq_len exactly divisible)
        z = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        z = z.permute(0, 1, 3, 2)  # (bs, nvars, patch_len, patch_num)

        # Encode: (bs, nvars, d_model, patch_num)
        z, _ = self.backbone(z, return_intermediate=False)

        bs, nvars, d_model, patch_num = z.shape

        # Reshape for cross-attention: (bs*nvars, patch_num, d_model)  ← K/V
        z_flat = z.permute(0, 1, 3, 2).reshape(bs * nvars, patch_num, d_model)

        # FutureQueryDecoder: (bs*nvars, output_patch_num, d_model)
        z_future_flat = self.future_query_decoder(z_flat)

        # Reshape back: (bs, nvars, output_patch_num, d_model)
        z_future = z_future_flat.reshape(bs, nvars, self.output_patch_num, d_model)

        # Head: expects (bs, nvars, d_model, output_patch_num)
        z_for_head = z_future.permute(0, 1, 3, 2)
        pred = self.head(z_for_head)  # (bs, nvars, pred_len)

        # RevIN denorm
        if self.revin:
            pred = pred.permute(0, 2, 1)
            pred = self.revin_layer(pred, 'denorm')
            pred = pred.permute(0, 2, 1)

        return pred, z_future

    # ------------------------------------------------------------------
    # Teacher path
    # ------------------------------------------------------------------
    def forward_teacher(self, z_chron: Tensor, loc=None, scale=None):
        """
        Args:
            z_chron: (bs, nvars, output_patch_num, 768)  — Chronos2 future embeddings
            loc:     (bs, nvars)  — x_future mean from Chronos2.embed
            scale:   (bs, nvars)  — x_future std  from Chronos2.embed
        Returns:
            pred:      (bs, nvars, pred_len)
            z_teacher: (bs, nvars, output_patch_num, d_model)  — for alignment loss
        """
        # proj_down: 768 → d_model
        z_teacher = self.proj_down(z_chron)  # (bs, nvars, P, d_model)

        # Reshape for head: (bs, nvars, d_model, P)
        z_perm = z_teacher.permute(0, 1, 3, 2)

        # Teacher head (independent weights from student head)
        pred = self.teacher_head(z_perm)  # (bs, nvars, pred_len)

        # Denorm using x_future loc/scale
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
    def forward(self, x: Tensor) -> Tensor:
        pred, _ = self.forward_student(x)
        return pred
