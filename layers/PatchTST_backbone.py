__all__ = ['PatchTST_backbone', 'Patch_Fusion_MLP', 'TransformerDecoder', 'PatchwiseHead']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

#from collections import OrderedDict
from layers.PatchTST_layers import *
from layers.RevIN import RevIN


def build_mlp(hidden_size, z_dim, projected_dim=512):
    return nn.Sequential(
        nn.Linear(hidden_size, projected_dim),
        nn.SiLU(),
        nn.Linear(projected_dim, projected_dim),
        nn.SiLU(),
        nn.Linear(projected_dim, z_dim),
    )


class Patch_Fusion_MLP(nn.Module):
    """将 (B, nvars, d_model, input_patch_num) 投影到 (B, nvars, d_model, output_patch_num)
    直接投影: d_model * input_patch_num → d_model * output_patch_num（无 hidden 层）
    """
    def __init__(self, input_patch_num, output_patch_num, d_model, dropout=0.0):
        super().__init__()
        self.output_patch_num = output_patch_num
        self.d_model = d_model
        self.projection = nn.Linear(d_model * input_patch_num, d_model * output_patch_num)

    def forward(self, x):
        # x: (B, nvars, d_model, input_patch_num)
        B, nvars, d_model, input_patch_num = x.shape

        # Flatten → project → reshape → (B, nvars, d_model, output_patch_num)
        x = x.reshape(B, nvars, -1)
        x = self.projection(x)
        x = x.reshape(B, nvars, d_model, self.output_patch_num)

        return x



class TransformerDecoder(nn.Module):
    """Transformer Decoder，用于在 patch_fusion 之后进一步处理
    使用与 PatchTST_backbone 相同的 n_heads 参数"""
    def __init__(self, output_patch_num, d_model, n_heads, d_layers=1, d_ff=None, dropout=0.0):
        super().__init__()
        if d_ff is None:
            d_ff = d_model

        self.layers = nn.ModuleList([
            TSTEncoderLayer(
                q_len=output_patch_num, d_model=d_model,
                n_heads=n_heads, d_ff=d_ff, dropout=dropout, activation='gelu'
            )
            for _ in range(d_layers)
        ])

    def forward(self, x):
        # x: (B*nvars, output_patch_num, d_model)
        for layer in self.layers:
            x = layer(x)
        return x


# Cell
class PatchTST_backbone(nn.Module):
    def __init__(self, c_in:int, context_window:int, target_window:int, patch_len:int, stride:int, max_seq_len:Optional[int]=1024,
                 n_layers:int=3, d_model=128, n_heads=16, d_k:Optional[int]=None, d_v:Optional[int]=None,
                 d_ff:int=256, norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", key_padding_mask:bool='auto',
                 padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, fc_dropout:float=0., head_dropout = 0, padding_patch = None,
                 pretrain_head:bool=False, head_type = 'flatten', individual = False, revin = True, affine = True, subtract_last = False,
                 encoder_depth: int = 2, contrastive: int = 0,
                 num_quantiles: int = 20,
                 output_patch_size: int = 16, use_patch_fusion: bool = False, patch_fusion_n_heads: int = 4,
                 d_extractor: int = 768, d_layers: int = 1, patch_fusion_type: str = 'fusion_MLP',
                 verbose:bool=False, **kwargs):

        super().__init__()

        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len)/stride + 1)
        if padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1

        # Patch Fusion parameters
        self.use_patch_fusion = use_patch_fusion
        self.output_patch_size = output_patch_size
        self.patch_fusion_n_heads = patch_fusion_n_heads
        self.d_extractor = d_extractor

        # Calculate output_patch_num
        assert target_window % output_patch_size == 0, f"pred_len must be divisible by output_patch_size, got pred_len={target_window}, output_patch_size={output_patch_size}"
        output_patch_num = target_window // output_patch_size

        # Backbone
        self.encoder_depth = encoder_depth
        # When use_patch_fusion=True, only need encoder_depth layers (extract zs from that layer)
        # No need for the remaining layers since we use Patch Fusion MLP + flatten head instead
        n_layers_backbone = encoder_depth if use_patch_fusion else n_layers
        self.backbone = TSTiEncoder(c_in, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
                                n_layers=n_layers_backbone, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, encoder_depth=encoder_depth, verbose=verbose, **kwargs)

        # Head
        self.head = None
        self.use_patch_fusion = use_patch_fusion
        self.head_type = head_type  # Always set head_type for denorm
        if not self.use_patch_fusion:
            self.head_nf = d_model * patch_num
            self.n_vars = c_in
            self.pretrain_head = pretrain_head
            self.individual = individual
            self.target_window = target_window
            self.num_quantiles = num_quantiles
            self.patch_num = patch_num

            if self.pretrain_head:
                self.head = self.create_pretrain_head(self.head_nf, c_in, fc_dropout) # custom head passed as a partial func with all its kwargs
            elif head_type == 'quantile':
                self.head = Quantile_Head(self.n_vars, d_model, patch_num, target_window,
                                           num_quantiles=num_quantiles, dropout=head_dropout)
            elif head_type == 'flatten':
                self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)

        self.contrastive = contrastive
        self.alignment_mlp_dim = d_extractor
        self.alignment_mlp = None
        if self.contrastive and not self.use_patch_fusion:
            self.alignment_mlp = build_mlp(d_model, d_extractor)

        # Patch Fusion components (only when use_patch_fusion=True)
        self.patch_fusion_mlp = None
        self.transformer_decoder = None

        # Patch Fusion 使用 d_channel=128 减少参数量，然后通过 alignment_mlp 投影到 d_extractor 用于对比学习
        # split_MLP 只做 patch_num 投影，保留 d_model，所以 d_channel 直接等于 d_model
        if self.use_patch_fusion:
            # Set required variables for head creation
            self.individual = individual
            self.n_vars = c_in
            self.target_window = target_window
            self.patch_fusion_type = patch_fusion_type

            # Patch_Fusion_MLP: (B, nvars, d_model, patch_num) -> (B, nvars, d_model, output_patch_num)
            if patch_fusion_type == 'split_MLP':
                # 只投影 patch_num 维度，保留 d_model
                self.patch_fusion_mlp = nn.Linear(patch_num, output_patch_num)
            else:
                self.patch_fusion_mlp = Patch_Fusion_MLP(
                    input_patch_num=patch_num,
                    output_patch_num=output_patch_num,
                    d_model=d_model
                )

            # Transformer Decoder: (B*nvars, output_patch_num, d_model)
            self.transformer_decoder = TransformerDecoder(
                output_patch_num=output_patch_num,
                d_model=d_model,
                n_heads=patch_fusion_n_heads,
                d_layers=d_layers,
                dropout=dropout
            )

            # Alignment MLP: d_model → d_extractor (用于对比学习，contrastive=0 时跳过)
            if self.contrastive:
                self.alignment_mlp = build_mlp(d_model, d_extractor)

            # Head for Patch Fusion branch
            if head_type == 'patch_wise':
                self.head = PatchwiseHead(
                    n_vars=self.n_vars,
                    d_model=d_model,
                    output_patch_num=output_patch_num,
                    output_patch_size=output_patch_size,
                    dropout=head_dropout
                )
            elif head_type == 'flatten':
                self.head = Flatten_Head(self.individual, self.n_vars, d_model * output_patch_num,
                                         target_window, head_dropout=head_dropout)
            elif head_type == 'quantile':
                self.head = Quantile_Head(self.n_vars, d_model, output_patch_num,
                                          target_window, num_quantiles=num_quantiles, dropout=head_dropout)
        
    
    def forward(self, z):                                       # z: [bs x nvars x seq_len]
        # norm
        if self.revin:
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0,2,1)

        # do patching
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)                   # z: [bs x nvars x patch_num x patch_len]
        z = z.permute(0,1,3,2)                                                              # z: [bs x nvars x patch_len x patch_num]

        # model - only request intermediate when contrastive=1 or use_patch_fusion=1
        need_intermediate = self.contrastive or self.use_patch_fusion
        if need_intermediate:
            z, zs = self.backbone(z, return_intermediate=True)                               # z: [bs x nvars x d_model x patch_num], zs: intermediate output
        else:
            # Original PatchTST: no intermediate output needed
            z, _ = self.backbone(z, return_intermediate=False)                              # z: [bs x nvars x d_model x patch_num]
            zs = None

        # Build prediction output based on configuration
        output = None
        zs_projected = None

        if self.use_patch_fusion and zs is not None:
            # Patch Fusion branch: encoder intermediate → patch_fusion_mlp → transformer_decoder → head
            # zs: (bs, nvars, d_model, patch_num) - from encoder_depth layer

            # Apply Patch_Fusion_MLP to get zs_fused
            # Output: (bs, nvars, d_model, output_patch_num)
            zs_fused = self.patch_fusion_mlp(zs)

            # For contrastive loss: project d_model → d_extractor
            bs, nvars, d_model, output_patch_num = zs_fused.shape
            if self.contrastive:
                zs_fused_for_proj = zs_fused.permute(0, 1, 3, 2).reshape(-1, d_model)  # (bs*nvars*output_patch_num, d_model)
                zs_projected_flat = self.alignment_mlp(zs_fused_for_proj)  # (bs*nvars*output_patch_num, d_extractor)
                zs_projected = zs_projected_flat.reshape(bs, nvars, output_patch_num, self.d_extractor)  # (bs, nvars, output_patch_num, d_extractor)

            # Apply Transformer Decoder
            # Input: (bs, nvars, d_model, output_patch_num) -> (bs*nvars, output_patch_num, d_model)
            zs_fused_flat = zs_fused.permute(0, 1, 3, 2).reshape(-1, output_patch_num, d_model)
            zs_fused_processed = self.transformer_decoder(zs_fused_flat)  # (bs*nvars, output_patch_num, d_model)

            # Reshape back: (bs*nvars, output_patch_num, d_model) -> (bs, nvars, d_model, output_patch_num)
            zs_fused_processed = zs_fused_processed.reshape(bs, nvars, output_patch_num, d_model)
            zs_fused_processed = zs_fused_processed.permute(0, 1, 3, 2)  # (bs, nvars, d_model, output_patch_num)

            output = self.head(zs_fused_processed)  # (bs, nvars, target_window)

        elif self.contrastive and not self.use_patch_fusion:
            # Original MLP Projector branch: encoder intermediate → projector → contrastive loss
            # Apply MLP projector to zs (no mean pooling - done in contrastive loss)
            bs, nvars, d_model, patch_num = zs.shape
            zs_flat = zs.permute(0, 1, 3, 2).reshape(-1, d_model)                          # [bs*nvars*patch_num x d_model]
            zs_projected = self.alignment_mlp(zs_flat)                                    # [bs*nvars*patch_num x projector_dim]
            zs_projected = zs_projected.reshape(bs, nvars, patch_num, self.alignment_mlp_dim)   # [bs x nvars x patch_num x projector_dim]

            # Encoder final output → Flatten Head
            output = self.head(z)  # z: [bs x nvars x target_window]

        else:
            # Original PatchTST: encoder final output → Flatten Head
            output = self.head(z)  # z: [bs x nvars x target_window] or [bs x nvars x num_quantiles x target_window]

        # Apply denorm
        if self.revin and output is not None:
            if self.head_type == 'quantile':
                # output: (bs, nvars, num_quantiles, pred_len)
                bs, nvars, num_quantiles, pred_len = output.shape
                output_per_quantile = []
                for q in range(num_quantiles):
                    out_q = output[:, :, q, :]  # (bs, nvars, pred_len)
                    out_q = out_q.permute(0, 2, 1)  # (bs, pred_len, nvars)
                    out_q = self.revin_layer(out_q, 'denorm')
                    out_q = out_q.permute(0, 2, 1)  # (bs, nvars, pred_len)
                    output_per_quantile.append(out_q)
                output = torch.stack(output_per_quantile, dim=2)  # (bs, nvars, num_quantiles, pred_len)
            else:
                output = output.permute(0, 2, 1)
                output = self.revin_layer(output, 'denorm')
                output = output.permute(0, 2, 1)

        # Return based on configuration
        if self.use_patch_fusion:
            # Return channel fusion output and zs_projected for contrastive loss
            return output, zs_projected
        elif self.contrastive:
            return output, zs_projected
        else:
            return output
    
    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout),
                    nn.Conv1d(head_nf, vars, 1)
                    )


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * patch_num]
                z = self.linears[i](z)                    # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)                 # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x


class Quantile_Head(nn.Module):
    """类似 Chronos2 的分位数预测头 - 每个 patch 独立预测

    Chronos2 方式:
    - 输入: (bs, nvars, d_model, input_patch_num) 来自encoder
    - 每个patch独立经过ResidualBlock: d_model -> d_ff -> num_quantiles * output_patch_size
    - 选择最后 output_patch_num 个patch的输出
    - Rearrange: (bs, nvars, output_patch_num, num_quantiles * output_patch_size) -> (bs, nvars, num_quantiles, pred_len)

    输出维度: (bs, nvars, num_quantiles, pred_len)
    """
    def __init__(self, n_vars, d_model, input_patch_num, pred_len, num_quantiles=20, dropout=0.0):
        super().__init__()
        self.n_vars = n_vars
        self.d_model = d_model
        self.input_patch_num = input_patch_num
        self.pred_len = pred_len
        self.num_quantiles = num_quantiles
        self.output_patch_size = 16  # 与 Chronos2 一致

        # 根据 pred_len 计算 output_patch_num
        # 例如 pred_len=96, output_patch_size=16 -> output_patch_num=6
        assert pred_len % self.output_patch_size == 0, f"pred_len must be divisible by output_patch_size, got pred_len={pred_len}"
        self.output_patch_num = pred_len // self.output_patch_size

        # 计算 d_ff，直接使用 head_nf 不压缩
        head_nf = d_model * input_patch_num
        d_ff = head_nf  # 不压缩，避免信息损失

        # ResidualBlock: (d_model * input_patch_num) -> d_ff -> (output_patch_num * num_quantiles * output_patch_size)
        self.hidden = nn.Linear(head_nf, d_ff)
        self.act = nn.GELU()
        self.output = nn.Linear(d_ff, self.output_patch_num * num_quantiles * self.output_patch_size)
        self.residual = nn.Linear(head_nf, self.output_patch_num * num_quantiles * self.output_patch_size)
        self.dropout = nn.Dropout(dropout)

        # 注册 quantiles buffer
        quantiles = torch.linspace(0.01, 0.99, num_quantiles)
        self.register_buffer('quantiles', quantiles)

    def forward(self, x):  # x: (bs, nvars, d_model, input_patch_num)
        # x: (bs, nvars, d_model, input_patch_num)
        # 拼接所有 patch 的信息: (bs, nvars, d_model * input_patch_num)
        x = x.reshape(x.size(0), x.size(1), -1)  # (bs, nvars, d_model * input_patch_num)

        # 一次性投影所有信息到输出空间
        # 输入: (bs, nvars, d_model * input_patch_num)
        # 输出: (bs, nvars, output_patch_num * num_quantiles * output_patch_size)
        hid = self.act(self.hidden(x))  # (bs, nvars, d_ff)
        out = self.dropout(self.output(hid))  # (bs, nvars, output_patch_num * num_quantiles * 16)
        res = self.residual(x)  # (bs, nvars, output_patch_num * num_quantiles * 16)
        out = out + res  # (bs, nvars, output_patch_num * num_quantiles * 16)

        # Rearrange: (bs, nvars, output_patch_num * num_quantiles * 16) -> (bs, nvars, num_quantiles, pred_len)
        bs, nvars, _ = out.shape
        out = out.reshape(bs, nvars, self.output_patch_num, self.num_quantiles, self.output_patch_size)
        # out shape: (bs, nvars, output_patch_num, num_quantiles, 16)
        out = out.permute(0, 1, 3, 2, 4)  # (bs, nvars, num_quantiles, output_patch_num, 16)
        out = out.reshape(bs, nvars, self.num_quantiles, self.output_patch_size * self.output_patch_num)
        # out shape: (bs, nvars, num_quantiles, pred_len)

        return out


class PatchwiseHead(nn.Module):
    """Patch-wise 预测头 - 每个 patch 独立预测，类似 Chronos2

    输入: (bs, nvars, d_model, output_patch_num) 来自 Patch Fusion
    每个 patch 经过 ResidualBlock: d_model -> d_ff -> output_patch_size
    残差连接
    Rearrange: (bs, nvars, output_patch_num, output_patch_size) -> (bs, nvars, pred_len)

    输出维度: (bs, nvars, pred_len)

    参数量: 比 Flatten_Head 更小 (d_model * output_patch_num -> d_model // 2 -> output_patch_size)
    """
    def __init__(self, n_vars, d_model, output_patch_num, output_patch_size=16, dropout=0.0):
        super().__init__()
        self.n_vars = n_vars
        self.d_model = d_model
        self.output_patch_num = output_patch_num
        self.output_patch_size = output_patch_size
        self.pred_len = output_patch_num * output_patch_size

        # ResidualBlock: d_model -> d_ff -> output_patch_size
        # 使用 d_model // 2 作为 hidden dim (平衡参数量和表示能力)
        # 对于 d_model=768: d_ff=384, 总参数量 ~450K (与 Flatten_Head 的 442K 接近)
        # 对于 d_model=128: d_ff=64, 总参数量 ~12K (远小于 Flatten_Head 的 258K)
        d_ff = max(d_model // 2, 64)  # 最小 64 避免过小

        self.hidden = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.output = nn.Linear(d_ff, output_patch_size)
        self.residual = nn.Linear(d_model, output_patch_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # x: (bs, nvars, d_model, output_patch_num)
        # x: (bs, nvars, d_model, output_patch_num)
        # 每个 patch 独立经过 ResidualBlock
        bs, nvars, d_model, output_patch_num = x.shape

        # Reshape: (bs, nvars, d_model, output_patch_num) -> (bs*nvars*output_patch_num, d_model)
        x = x.permute(0, 1, 3, 2)  # (bs, nvars, output_patch_num, d_model)
        x = x.reshape(-1, d_model)  # (bs*nvars*output_patch_num, d_model)

        # ResidualBlock
        hid = self.act(self.hidden(x))  # (bs*nvars*output_patch_num, d_ff)
        out = self.dropout(self.output(hid))  # (bs*nvars*output_patch_num, output_patch_size)
        res = self.residual(x)  # (bs*nvars*output_patch_num, output_patch_size)
        out = out + res  # (bs*nvars*output_patch_num, output_patch_size)

        # Reshape back: (bs*nvars*output_patch_num, output_patch_size) -> (bs, nvars, pred_len)
        out = out.reshape(bs, nvars, self.pred_len)

        return out



class TSTiEncoder(nn.Module):  #i means channel-independent
    def __init__(self, c_in, patch_num, patch_len, max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, encoder_depth=2, verbose=False, **kwargs):


        super().__init__()

        self.patch_num = patch_num
        self.patch_len = patch_len
        self.n_layers = n_layers
        self.encoder_depth = encoder_depth

        # Input encoding
        q_len = patch_num
        self.W_P = nn.Linear(patch_len, d_model)        # Eq 1: projection of feature vectors onto a d-dim vector space
        self.seq_len = q_len

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn)


    def forward(self, x, return_intermediate=False) -> Tensor:                   # x: [bs x nvars x patch_len x patch_num]

        n_vars = x.shape[1]
        # Input encoding
        x = x.permute(0,1,3,2)                                                   # x: [bs x nvars x patch_num x patch_len]
        x = self.W_P(x)                                                          # x: [bs x nvars x patch_num x d_model]

        u = torch.reshape(x, (x.shape[0]*x.shape[1],x.shape[2],x.shape[3]))      # u: [bs * nvars x patch_num x d_model]
        u = self.dropout(u + self.W_pos)                                         # u: [bs * nvars x patch_num x d_model]

        # Encoder - only return intermediate when needed
        if return_intermediate:
            z, intermediate = self.encoder(u, return_intermediate=True)           # z: [bs * nvars x patch_num x d_model]

            # Reshape intermediate outputs
            intermediate_reshaped = []
            for intermediate_z in intermediate:
                intermediate_z = torch.reshape(intermediate_z, (-1, n_vars, intermediate_z.shape[-2], intermediate_z.shape[-1]))
                intermediate_z = intermediate_z.permute(0, 1, 3, 2)              # [bs x nvars x d_model x patch_num]
                intermediate_reshaped.append(intermediate_z)

            # Get output at specified encoder_depth
            encoder_idx = self.encoder_depth - 1  # 0-indexed
            if encoder_idx < len(intermediate_reshaped):
                zs = intermediate_reshaped[encoder_idx]
            else:
                zs = z.permute(0,1,3,2)  # fallback to final output
        else:
            z = self.encoder(u)                                                  # z: [bs * nvars x patch_num x d_model]
            zs = None

        z = torch.reshape(z, (-1,n_vars,z.shape[-2],z.shape[-1]))                # z: [bs x nvars x patch_num x d_model]
        z = z.permute(0,1,3,2)                                                   # z: [bs x nvars x d_model x patch_num]

        return z, zs    
            
            
    
# Cell
class TSTEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None,
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList([TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src:Tensor, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None, return_intermediate:bool=False):
        output = src
        scores = None
        intermediate_outputs = []

        if self.res_attention:
            for mod in self.layers:
                output, scores = mod(output, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
                if return_intermediate:
                    intermediate_outputs.append(output)
            if return_intermediate:
                return output, intermediate_outputs
            return output
        else:
            for mod in self.layers:
                output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
                if return_intermediate:
                    intermediate_outputs.append(output)
            if return_intermediate:
                return output, intermediate_outputs
            return output



class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn


    def forward(self, src:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None) -> Tensor:

        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src




class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))


    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights

