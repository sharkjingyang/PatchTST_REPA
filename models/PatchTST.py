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
        self.model_name = configs.model  # PatchTST, PatchTST_REPA, PatchTST_REPA_Fusion

        # Get contrastive setting from args
        user_contrastive = getattr(configs, 'contrastive', None)

        # Set contrastive and use_patch_fusion based on model_name
        # PatchTST_REPA_Fusion always uses Channel Fusion (cannot be disabled)
        if self.model_name == 'PatchTST_REPA':
            self.contrastive = 1 if user_contrastive is None else user_contrastive
            self.use_patch_fusion = False
        elif self.model_name == 'PatchTST_REPA_Fusion':
            self.contrastive = 1 if user_contrastive is None else user_contrastive
            self.use_patch_fusion = True  # Always enabled
        else:  # PatchTST
            self.contrastive = 0
            self.use_patch_fusion = False

        # Feature extractor parameters
        feature_extractor = getattr(configs, 'feature_extractor', 'mantis')
        mantis_pretrained = getattr(configs, 'mantis_pretrained', './Mantis')

        # Channel fusion parameters
        output_patch_size = getattr(configs, 'output_patch_size', 16)
        patch_fusion_n_heads = getattr(configs, 'patch_fusion_n_heads', 4)
        d_layers = getattr(configs, 'd_layers', 1)  # Transformer Decoder layers
        patch_fusion_type = getattr(configs, 'patch_fusion_type', 'fusion_MLP')  # Channel fusion MLP type

        # d_extractor: based on feature extractor (Mantis=256, TiViT/Chronos=768)
        if self.model_name == 'PatchTST_REPA_Fusion':
            if feature_extractor == 'mantis':
                d_extractor = 256
                print(f"Using patch fusion with Mantis, d_extractor={d_extractor}")
            else:
                d_extractor = 768  # TiViT or Chronos
                print(f"Using patch fusion with {feature_extractor}, d_extractor={d_extractor}")
        else:
            d_extractor = 768  # default

        # Auto-adjust projector_dim based on feature extractor (only for REPA models)
        if self.model_name in ['PatchTST_REPA', 'PatchTST_REPA_Fusion']:
            if feature_extractor == 'mantis':
                projector_dim = 256  # Mantis output dimension
                print(f"Using Mantis feature extractor, auto-adjusting projector_dim to {projector_dim}")
            elif feature_extractor == 'chronos':
                projector_dim = 768  # Chronos2 output dimension
                print(f"Using Chronos feature extractor, auto-adjusting projector_dim to {projector_dim}")
        self.feature_extractor = getattr(configs, 'feature_extractor', 'mantis')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        self.chronos_output_dim = 768  # Chronos2 default output dimension
        self.num_output_patches = configs.pred_len // 16  # Chronos2 native patch_len = 16

        # Build feature extractor (TiViT, Mantis or Chronos) only when using REPA model with contrastive
        self.tivit = None
        self.mantis = None
        self.chronos = None

        if self.model_name in ['PatchTST_REPA', 'PatchTST_REPA_Fusion'] and self.contrastive:
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
        else:
            # contrastive=0: original PatchTST, no TiViT/Mantis
            pass
            pass

        # Prediction head parameters
        head_type = getattr(configs, 'head_type', 'flatten')
        num_quantiles = getattr(configs, 'num_quantiles', 20)

        # model
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            # Use projector if model_name is PatchTST_REPA or PatchTST_REPA_Fusion
            if self.model_name in ['PatchTST_REPA', 'PatchTST_REPA_Fusion']:
                self.model_trend = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride,
                                      max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                      n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                      dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                      attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                      pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                      pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                      subtract_last=subtract_last, encoder_depth=encoder_depth,
                                      contrastive=self.contrastive, num_quantiles=num_quantiles,
                                      output_patch_size=output_patch_size, use_patch_fusion=self.use_patch_fusion,
                                      patch_fusion_n_heads=patch_fusion_n_heads, d_extractor=d_extractor, d_layers=d_layers,
                                      patch_fusion_type=patch_fusion_type,
                                      verbose=verbose, **kwargs)
                self.model_res = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride,
                                      max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                      n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                      dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                      attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                      pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                      pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                      subtract_last=subtract_last, encoder_depth=encoder_depth,
                                      contrastive=self.contrastive, num_quantiles=num_quantiles,
                                      output_patch_size=output_patch_size,
                                      patch_fusion_n_heads=patch_fusion_n_heads, d_extractor=d_extractor, d_layers=d_layers,
                                      patch_fusion_type=patch_fusion_type,
                                      verbose=verbose, **kwargs)
            else:
                # Original PatchTST: no projector params
                self.model_trend = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride,
                                      max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                      n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                      dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                      attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                      pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                      pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                      subtract_last=subtract_last, num_quantiles=num_quantiles,
                                      verbose=verbose, **kwargs)
                self.model_res = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride,
                                      max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                      n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                      dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                      attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                      pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                      pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                      subtract_last=subtract_last, num_quantiles=num_quantiles,
                                      verbose=verbose, **kwargs)
        else:
            # Use projector/channel_fusion only if contrastive=1 or use_patch_fusion=1
            if self.contrastive or self.use_patch_fusion:
                self.model = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride,
                                      max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                      n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                      dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                      attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                      pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch=padding_patch,
                                      pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                      subtract_last=subtract_last, encoder_depth=encoder_depth,
                                      contrastive=self.contrastive, num_quantiles=num_quantiles,
                                      output_patch_size=output_patch_size, use_patch_fusion=self.use_patch_fusion,
                                      patch_fusion_n_heads=patch_fusion_n_heads, d_extractor=d_extractor, d_layers=d_layers,
                                      patch_fusion_type=patch_fusion_type,
                                      verbose=verbose, **kwargs)
            else:
                # Original PatchTST: no projector params
                self.model = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride,
                                      max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                      n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                      dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                      attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                      pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch=padding_patch,
                                      pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                      subtract_last=subtract_last, num_quantiles=num_quantiles,
                                      verbose=verbose, **kwargs)
    
    
    def forward(self, x, target=None, return_projector=False):           # x: [Batch, Input length, Channel], target: [Batch, Target length, Channel]
        """
        Args:
            x: input sequence [Batch, Input length, Channel]
            target: target sequence [Batch, Target length, Channel], used for TiViT feature extraction
            return_projector: if True, return zs_project and zs_tilde for contrastive loss (training only)
        """
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)  # x: [Batch, Channel, Input length]
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
            return x, x  # Return same output for both when using decomposition
        else:
            x_original = x  # (bs, seq_len, nvars), 用于 Chronos encode
            x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]

            # Check head_type
            head_type = getattr(self.model, 'head_type', 'flatten') if hasattr(self, 'model') else 'flatten'

            # Channel Fusion always returns tuple (output, zs)
            if self.use_patch_fusion:
                output, zs = self.model(x)  # returns (output2, zs_projected) - only patch fusion branch
                # Permute output based on head_type
                if head_type == 'quantile':
                    # output: (bs, nvars, num_quantiles, pred_len) -> (bs, pred_len, nvars, num_quantiles)
                    output = output.permute(0, 3, 1, 2)
                else:
                    # output: (bs, nvars, pred_len) -> (bs, pred_len, nvars)
                    output = output.permute(0, 2, 1)

                # If contrastive=0, don't return zs for contrastive loss
                if not self.contrastive:
                    return output

                # Extract zs_tilde for contrastive learning
                zs_tilde = None
                if return_projector and target is not None:
                    with torch.no_grad():
                        target_pred = target  # (bs, pred_len, nvars) - already sliced in exp_main.py

                        if self.feature_extractor == 'tivit' and self.tivit is not None:
                            # TiViT extraction
                            zs_tilde_list = []
                            for c in range(target_pred.shape[2]):
                                channel_input = target_pred[:, :, c:c+1]
                                channel_embed = self.tivit(channel_input)
                                zs_tilde_list.append(channel_embed)
                            zs_tilde = torch.stack(zs_tilde_list, dim=1)
                        elif self.feature_extractor == 'mantis' and self.mantis is not None:
                            # Mantis extraction
                            target_perm = target_pred.permute(0, 2, 1)
                            target_scaled = F.interpolate(
                                target_perm.float(),
                                size=512,
                                mode='linear',
                                align_corners=False
                            )
                            target_np = target_scaled.cpu().numpy()
                            bs, nvars, _ = target_scaled.shape
                            zs_tilde_flat = self.mantis.transform(target_np)
                            zs_tilde_flat = torch.from_numpy(zs_tilde_flat).float().to(self.device)
                            zs_tilde = zs_tilde_flat.reshape(bs, nvars, -1)
                        elif self.feature_extractor == 'chronos' and self.chronos is not None:
                            # 用 embed() 从 batch_x 提取 past tokens 用于对齐
                            input_perm = x_original.permute(0, 2, 1)  # (bs, nvars, seq_len)
                            num_past = x_original.shape[1] // 16  # seq_len // chronos patch_len=16
                            embeddings_list, _ = self.chronos.embed(input_perm)
                            embeddings = torch.stack(embeddings_list, dim=0).to(self.device)  # (bs, nvars, num_past+2, 768)
                            zs_tilde = embeddings[:, :, :num_past, :]  # (bs, nvars, num_past, 768)

                # Return (output, zs, zs_tilde) for contrastive learning
                return output, zs, zs_tilde

            # Original PatchTST (non-channel-fusion)
            # Original PatchTST (contrastive=0): return only output
            if not self.contrastive:
                output = self.model(x)  # returns only output
                if head_type == 'quantile':
                    # output: (bs, nvars, num_quantiles, pred_len) -> (bs, pred_len, nvars, num_quantiles)
                    output = output.permute(0, 3, 1, 2)  # (bs, pred_len, nvars, num_quantiles)
                    return output
                else:
                    output = output.permute(0,2,1)    # output: [Batch, Input length, Channel]
                    return output

            # With projector (contrastive=1): return output and zs
            output, zs = self.model(x)  # returns (output, zs_projected)
            # Permute output based on head_type
            if head_type == 'quantile':
                # output: (bs, nvars, num_quantiles, pred_len) -> (bs, pred_len, nvars, num_quantiles)
                output = output.permute(0, 3, 1, 2)
            else:
                # output: (bs, nvars, pred_len) -> (bs, pred_len, nvars)
                output = output.permute(0, 2, 1)

            # Only extract feature extractor features when return_projector=True (training)
            # Note: target should already be sliced to pred_len in exp_main.py
            zs_tilde = None
            if return_projector and target is not None:
                with torch.no_grad():
                    target_pred = target  # (bs, pred_len, nvars) - already sliced in exp_main.py

                    if self.feature_extractor == 'tivit' and self.tivit is not None:
                        # TiViT extraction
                        # target_pred: (bs, pred_len, nvars)
                        # TiViT expects input: (bs, seq_len, 1) for single channel
                        zs_tilde_list = []
                        for c in range(target_pred.shape[2]):  # target_pred.shape[2] = nvars
                            channel_input = target_pred[:, :, c:c+1]  # (bs, pred_len, 1)
                            channel_embed = self.tivit(channel_input)  # (bs, d_vit)
                            zs_tilde_list.append(channel_embed)
                        zs_tilde = torch.stack(zs_tilde_list, dim=1)  # (bs, nvars, d_vit)
                    elif self.feature_extractor == 'mantis' and self.mantis is not None:
                        # Mantis extraction
                        # Mantis expects: (n_samples, channels, time_steps)
                        # target_pred: (bs, pred_len, nvars) -> (bs, nvars, pred_len)
                        target_perm = target_pred.permute(0, 2, 1)  # (bs, nvars, pred_len)
                        # Resize to 512 for Mantis
                        target_scaled = F.interpolate(
                            target_perm.float(),
                            size=512,
                            mode='linear',
                            align_corners=False
                        )  # (bs, nvars, 512)
                        # Mantis transform: (n_samples, channels, time_steps) -> (n_samples, channels * 256)
                        target_np = target_scaled.cpu().numpy()
                        bs, nvars, _ = target_scaled.shape
                        zs_tilde_flat = self.mantis.transform(target_np)  # (bs, nvars * 256)
                        # Convert back to tensor and reshape
                        zs_tilde_flat = torch.from_numpy(zs_tilde_flat).float().to(self.device)
                        zs_tilde = zs_tilde_flat.reshape(bs, nvars, -1)  # (bs, nvars, 256)
                    elif self.feature_extractor == 'chronos' and self.chronos is not None:
                        # 用 embed() 从 batch_x 提取 past tokens 用于对齐
                        input_perm = x_original.permute(0, 2, 1)  # (bs, nvars, seq_len)
                        num_past = x_original.shape[1] // 16  # seq_len // chronos patch_len=16
                        embeddings_list, _ = self.chronos.embed(input_perm)
                        embeddings = torch.stack(embeddings_list, dim=0).to(self.device)  # (bs, nvars, num_past+2, 768)
                        zs_tilde = embeddings[:, :, :num_past, :]  # (bs, nvars, num_past, 768)

            if return_projector:
                return output, zs, zs_tilde
            else:
                return output