__all__ = ['PatchTST']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.PatchTST_backbone import PatchTST_backbone
from layers.PatchTST_layers import series_decomp
from layers.Tivit import get_tivit, get_patch_size


class Model(nn.Module):
    def __init__(self, configs, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs):
        
        super().__init__()
        
        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len
        
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        
        individual = configs.individual
    
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch
        
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        
        decomposition = configs.decomposition
        kernel_size = configs.kernel_size
        encoder_depth = configs.encoder_depth
        projector_dim = getattr(configs, 'projector_dim', 768)
        use_projector = getattr(configs, 'use_projector', False)

        # TiViT parameters
        self.use_projector = use_projector
        self.tivit_model_name = getattr(configs, 'tivit_model', 'laion/CLIP-ViT-B-16-laion2B-s34B-b88K')
        self.tivit_layer = getattr(configs, 'tivit_layer', 6)
        self.tivit_aggregation = getattr(configs, 'tivit_aggregation', 'mean')
        self.tivit_stride = getattr(configs, 'tivit_stride', 0.1)
        self.tivit_patch_size = getattr(configs, 'tivit_patch_size', 'sqrt')

        # Build TiViT if using projector
        self.tivit = None
        if self.use_projector:
            # Use context_window + target_window for TiViT (full sequence length)
            full_seq_len = context_window + target_window
            actual_patch_size = get_patch_size(self.tivit_patch_size, full_seq_len)
            self.tivit = get_tivit(
                model_name=self.tivit_model_name,
                model_layer=self.tivit_layer,
                aggregation=self.tivit_aggregation,
                stride=self.tivit_stride,
                patch_size=actual_patch_size,
                device='cuda' if torch.cuda.is_available() else 'cpu',
            )
            self.tivit.eval()

        # model
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride,
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, encoder_depth=encoder_depth, projector_dim=projector_dim, use_projector=use_projector, verbose=verbose, **kwargs)
            self.model_res = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride,
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, encoder_depth=encoder_depth, projector_dim=projector_dim, use_projector=use_projector, verbose=verbose, **kwargs)
        else:
            self.model = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride,
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch=padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, encoder_depth=encoder_depth, projector_dim=projector_dim, use_projector=use_projector, verbose=verbose, **kwargs)
    
    
    def forward(self, x, target=None):           # x: [Batch, Input length, Channel], target: [Batch, Target length, Channel]
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)  # x: [Batch, Channel, Input length]
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
            return x, x  # Return same output for both when using decomposition
        else:
            x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
            output, zs = self.model(x)  # always returns (output, zs)
            output = output.permute(0,2,1)    # output: [Batch, Input length, Channel]
            zs = zs.permute(0,2,1)          # zs: [Batch, Channel, d_model]

            # Extract TiViT features if using projector and target is provided
            zs_tilde = None
            if self.use_projector and self.tivit is not None and target is not None:
                with torch.no_grad():
                    # target: (bs, seq_len+pred_len, nvars) -> keep as (bs, seq_len, nvars)
                    # Process each channel separately
                    # TiViT expects input: (bs, seq_len, 1) for single channel
                    zs_tilde_list = []
                    for c in range(target.shape[2]):  # target.shape[2] = nvars
                        channel_input = target[:, :, c:c+1]  # (bs, seq_len, 1)
                        channel_embed = self.tivit(channel_input)  # (bs, d_vit)
                        zs_tilde_list.append(channel_embed)
                    zs_tilde = torch.stack(zs_tilde_list, dim=1)  # (bs, nvars, d_vit)

            return output, zs, zs_tilde