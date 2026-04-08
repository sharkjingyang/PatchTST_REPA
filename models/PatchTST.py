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

# Try to import Mantis
try:
    from mantis.architecture import Mantis8M
    from mantis.trainer import MantisTrainer
    import torch.nn.functional as F
    HAS_MANTIS = True
except ImportError:
    HAS_MANTIS = False

# Try to import Chronos
try:
    from chronos import Chronos2Pipeline
    HAS_CHRONOS = True
except ImportError:
    HAS_CHRONOS = False


class Model(nn.Module):
    def __init__(self, configs, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs):
        
        super().__init__()
        
        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len
        self.pred_len = target_window  # Store for feature extraction
        
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

        # Model type detection
        self.model_name = configs.model  # PatchTST, PatchTST_REPA

        # Get alignment setting from args
        user_alignment = getattr(configs, 'alignment', None)

        if self.model_name == 'PatchTST_REPA':
            self.alignment = 1 if user_alignment is None else user_alignment
        else:  # PatchTST
            self.alignment = 0

        # Feature extractor parameters
        feature_extractor = getattr(configs, 'feature_extractor', 'mantis')

        # d_extractor: based on feature extractor (Mantis=256, TiViT/Chronos=768)
        if feature_extractor == 'mantis':
            d_extractor = 256
            projector_dim = 256
            if self.alignment:
                print(f"Using Mantis feature extractor, d_extractor={d_extractor}")
        else:
            d_extractor = 768  # TiViT or Chronos
            projector_dim = 768
            if self.alignment:
                print(f"Using {feature_extractor} feature extractor, d_extractor={d_extractor}")

        self.feature_extractor = feature_extractor
        # Use Chronos2's InstanceNorm (arcsinh) when aligning with Chronos2 past tokens
        # so student and teacher see the same normalized input space
        self.use_chronos_norm = (self.model_name == 'PatchTST_REPA' and feature_extractor == 'chronos')
        self.device = getattr(configs, 'device', 'cuda:0')

        # TiViT parameters
        self.tivit_model_name = getattr(configs, 'tivit_model', 'laion/CLIP-ViT-B-16-laion2B-s34B-b88K')
        self.tivit_layer = getattr(configs, 'tivit_layer', 6)
        self.tivit_aggregation = getattr(configs, 'tivit_aggregation', 'mean')
        self.tivit_stride = getattr(configs, 'tivit_stride', 0.1)
        self.tivit_patch_size = getattr(configs, 'tivit_patch_size', 'sqrt')
        self.tivit_pretrained = getattr(configs, 'tivit_pretrained', './open_clip/open_clip_model.safetensors')

        # Mantis parameters
        self.mantis_pretrained = getattr(configs, 'mantis_pretrained', './Mantis')
        self.mantis_output_dim = 256  # Mantis default output dimension

        # Chronos parameters
        self.chronos_pretrained = getattr(configs, 'chronos_pretrained', './Chronos2')

        # Build feature extractor (TiViT, Mantis or Chronos) only when using REPA model with contrastive
        self.tivit = None
        self.mantis = None
        self.chronos = None

        if self.model_name == 'PatchTST_REPA' and self.alignment:
            if self.feature_extractor == 'tivit':
                # Build TiViT
                full_seq_len = context_window + target_window
                actual_patch_size = get_patch_size(self.tivit_patch_size, full_seq_len)
                self.tivit = get_tivit(
                    model_name=self.tivit_model_name,
                    model_layer=self.tivit_layer,
                    aggregation=self.tivit_aggregation,
                    stride=self.tivit_stride,
                    patch_size=actual_patch_size,
                    device=self.device,
                    pretrained=self.tivit_pretrained,
                )
                self.tivit.eval()
                for param in self.tivit.parameters():
                    param.requires_grad = False
            elif self.feature_extractor == 'mantis':
                if not HAS_MANTIS:
                    raise ImportError("mantis-tsfm is not installed. Please install it with: pip install mantis-tsfm")
                # Build Mantis
                network = Mantis8M(device=self.device)
                network = network.from_pretrained(self.mantis_pretrained)
                self.mantis = MantisTrainer(device=self.device, network=network)
                # Register network as sub-module so its params are included in model.parameters()
                self.add_module('mantis_network', self.mantis.network)
                self.mantis.network.eval()
                for param in self.mantis.network.parameters():
                    param.requires_grad = False
            elif self.feature_extractor == 'chronos':
                if not HAS_CHRONOS:
                    raise ImportError("chronos is not installed. Please install it with: pip install chronos-forecasting")
                # Build Chronos
                self.chronos = Chronos2Pipeline.from_pretrained(self.chronos_pretrained, device_map=self.device)
                # Register chronos.model as sub-module so its params are included in model.parameters()
                self.add_module('chronos_model', self.chronos.model)
                self.chronos.model.eval()
                for param in self.chronos.model.parameters():
                    param.requires_grad = False
            else:
                raise ValueError(f"Unknown feature_extractor: {self.feature_extractor}. Choose 'tivit', 'mantis' or 'chronos'.")
        # Prediction head parameters
        head_type = getattr(configs, 'head_type', 'flatten')
        num_quantiles = getattr(configs, 'num_quantiles', 20)

        # model
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PatchTST_backbone(c_in=c_in, context_window=context_window, target_window=target_window, patch_len=patch_len, stride=stride,
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch=padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, use_chronos_norm=self.use_chronos_norm, encoder_depth=encoder_depth,
                                  alignment=self.alignment, num_quantiles=num_quantiles, d_extractor=d_extractor,
                                  verbose=verbose, **kwargs)
            self.model_res = PatchTST_backbone(c_in=c_in, context_window=context_window, target_window=target_window, patch_len=patch_len, stride=stride,
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch=padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, use_chronos_norm=self.use_chronos_norm, encoder_depth=encoder_depth,
                                  alignment=self.alignment, num_quantiles=num_quantiles, d_extractor=d_extractor,
                                  verbose=verbose, **kwargs)
        else:
            self.model = PatchTST_backbone(c_in=c_in, context_window=context_window, target_window=target_window, patch_len=patch_len, stride=stride,
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch=padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, use_chronos_norm=self.use_chronos_norm, encoder_depth=encoder_depth,
                                  alignment=self.alignment, num_quantiles=num_quantiles, d_extractor=d_extractor,
                                  verbose=verbose, **kwargs)
    
    
    def forward(self, x, target=None, return_projector=False):
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
            x = x.permute(0,2,1)
            return x
        else:
            x_original = x  # (bs, seq_len, nvars)
            x = x.permute(0,2,1)  # (bs, nvars, seq_len)

            head_type = getattr(self.model, 'head_type', 'flatten')

            if not self.alignment:
                output = self.model(x)
                if head_type == 'quantile':
                    return output.permute(0, 3, 1, 2)
                else:
                    return output.permute(0, 2, 1)

            # contrastive=1: returns (output, zs_projected)
            output, zs = self.model(x)
            if head_type == 'quantile':
                output = output.permute(0, 3, 1, 2)
            else:
                output = output.permute(0, 2, 1)

            zs_tilde = None
            if return_projector and target is not None:
                with torch.no_grad():
                    target_pred = target  # (bs, pred_len, nvars)

                    if self.feature_extractor == 'tivit' and self.tivit is not None:
                        zs_tilde_list = []
                        for c in range(target_pred.shape[2]):
                            channel_input = target_pred[:, :, c:c+1]
                            channel_embed = self.tivit(channel_input)
                            zs_tilde_list.append(channel_embed)
                        zs_tilde = torch.stack(zs_tilde_list, dim=1)
                    elif self.feature_extractor == 'mantis' and self.mantis is not None:
                        target_perm = target_pred.permute(0, 2, 1)
                        target_scaled = F.interpolate(target_perm.float(), size=512, mode='linear', align_corners=False)
                        target_np = target_scaled.cpu().numpy()
                        bs, nvars, _ = target_scaled.shape
                        zs_tilde_flat = self.mantis.transform(target_np)
                        zs_tilde_flat = torch.from_numpy(zs_tilde_flat).float().to(self.device)
                        zs_tilde = zs_tilde_flat.reshape(bs, nvars, -1)
                    elif self.feature_extractor == 'chronos' and self.chronos is not None:
                        input_perm = x_original.permute(0, 2, 1)  # (bs, nvars, seq_len)
                        num_past = x_original.shape[1] // 16
                        embeddings_list, _ = self.chronos.embed(input_perm.cpu())
                        embeddings = torch.stack(embeddings_list, dim=0).to(self.device)
                        zs_tilde = embeddings[:, :, :num_past, :]  # (bs, nvars, num_past, 768)

            if return_projector:
                return output, zs, zs_tilde
            else:
                return output