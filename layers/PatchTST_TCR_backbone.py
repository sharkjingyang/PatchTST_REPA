"""
PatchTST_TCR_backbone: Temporal Contrastive Regularization backbone.

Simplified from PatchTST_backbone — no alignment_mlp, no patch_fusion, no d_extractor.
forward() returns (output, zs_raw) where zs_raw is the raw encoder intermediate output
at encoder_depth layer, used directly for TCR loss computation.
"""

from typing import Optional
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from layers.PatchTST_layers import TSTiEncoder, Flatten_Head
from layers.RevIN import RevIN


class PatchTST_TCR_backbone(nn.Module):
    def __init__(self, c_in: int, context_window: int, target_window: int,
                 patch_len: int, stride: int, max_seq_len: Optional[int] = 1024,
                 n_layers: int = 3, d_model: int = 128, n_heads: int = 16,
                 d_k: Optional[int] = None, d_v: Optional[int] = None,
                 d_ff: int = 256, norm: str = 'BatchNorm',
                 attn_dropout: float = 0., dropout: float = 0., act: str = "gelu",
                 key_padding_mask: bool = 'auto', padding_var: Optional[int] = None,
                 attn_mask: Optional[Tensor] = None, res_attention: bool = True,
                 pre_norm: bool = False, store_attn: bool = False,
                 pe: str = 'zeros', learn_pe: bool = True,
                 fc_dropout: float = 0., head_dropout: float = 0,
                 padding_patch: Optional[str] = None,
                 pretrain_head: bool = False, head_type: str = 'flatten',
                 individual: bool = False, revin: bool = True,
                 affine: bool = True, subtract_last: bool = False,
                 encoder_depth: int = 2,
                 verbose: bool = False, **kwargs):
        super().__init__()

        # RevIN
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len) / stride + 1)
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1

        # Encoder — full n_layers (不像 Fusion 模式只用 encoder_depth 层)
        self.encoder_depth = encoder_depth
        self.backbone = TSTiEncoder(
            c_in, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
            n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v,
            d_ff=d_ff, attn_dropout=attn_dropout, dropout=dropout, act=act,
            key_padding_mask=key_padding_mask, padding_var=padding_var,
            attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
            store_attn=store_attn, pe=pe, learn_pe=learn_pe,
            encoder_depth=encoder_depth, verbose=verbose, **kwargs
        )

        # Head
        self.head_type = head_type
        self.individual = individual
        self.n_vars = c_in
        self.target_window = target_window
        head_nf = d_model * patch_num

        if pretrain_head:
            self.head = nn.Sequential(
                nn.Dropout(fc_dropout),
                nn.Conv1d(head_nf, c_in, 1)
            )
        elif head_type == 'flatten':
            self.head = Flatten_Head(individual, c_in, head_nf, target_window,
                                     head_dropout=head_dropout)
        else:
            raise ValueError(f"PatchTST_TCR_backbone only supports head_type='flatten', got '{head_type}'")

    def forward(self, z):
        """
        Args:
            z: (bs, nvars, seq_len)
        Returns:
            output: (bs, nvars, target_window)
            zs_raw: (bs, nvars, d_model, patch_num) — encoder_depth 层输出，用于 TCR loss
        """
        # RevIN norm
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0, 2, 1)

        # Patching
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)   # (bs, nvars, patch_num, patch_len)
        z = z.permute(0, 1, 3, 2)                                          # (bs, nvars, patch_len, patch_num)

        # Encoder — always request intermediate for TCR
        z, zs_raw = self.backbone(z, return_intermediate=True)              # z: (bs, nvars, d_model, patch_num)
                                                                            # zs_raw: (bs, nvars, d_model, patch_num)

        # Head
        output = self.head(z)                                               # (bs, nvars, target_window)

        # RevIN denorm
        if self.revin:
            output = output.permute(0, 2, 1)
            output = self.revin_layer(output, 'denorm')
            output = output.permute(0, 2, 1)

        return output, zs_raw
