__all__ = ['Model']

import torch
from torch import nn
import torch.nn.functional as F

# Try to import Chronos
try:
    from chronos import Chronos2Pipeline
    HAS_CHRONOS = True
except ImportError:
    HAS_CHRONOS = False

from layers.PatchTST_backbone import Flatten_Head, PatchwiseHead


class Model(nn.Module):
    """Chronos2 + prediction head model for time series forecasting.

    This model uses Chronos2 (frozen) to extract features, then uses a prediction head.

    Flow (use_future_patch=0):
        Input x: (bs, seq_len, nvars)
            ↓ Chronos2.embed(x) - frozen
        Feature: (bs, nvars, num_patches, 768)  [num_patches = seq_len/16]
            ↓ permute(0,1,3,2): (bs, nvars, 768, num_patches)
            ↓ Flatten_Head (individual=True)
            ↓ flatten: (bs*nvars, 768*num_patches)
            ↓ linear: (bs*nvars, pred_len)
        Output: (bs, nvars, pred_len) → permute → (bs, pred_len, nvars)

    Flow (use_future_patch=1):
        Input x: (bs, seq_len, nvars)
            ↓ Chronos2.model.encode(x, num_output_patches) - frozen
        Feature: (bs, nvars, num_context_patches + 1 + num_output_patches, 768)
            [num_context_patches = seq_len/16, num_output_patches = pred_len/16]
            ↓ extract ONLY future tokens: hidden_states[:, -num_output_patches:]
        Feature: (bs, nvars, num_output_patches, 768)
            ↓ PatchwiseHead (per-patch prediction, like Chronos2)
            ↓ ResidualBlock per patch: d_model -> d_ff -> output_patch_size
        Output: (bs, nvars, pred_len) → permute → (bs, pred_len, nvars)
    """

    def __init__(self, configs, **kwargs):
        super().__init__()

        if not HAS_CHRONOS:
            raise ImportError("chronos is not installed. Please install it with: pip install chronos-forecasting")

        # Load parameters
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.n_vars = configs.enc_in  # number of variables/channels
        self.patch_len = configs.patch_len  # Chronos uses patch_len=16
        self.stride = configs.stride

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Chronos parameters
        self.chronos_pretrained = getattr(configs, 'chronos_pretrained', './Chronos2')
        self.chronos_output_dim = 768  # Chronos2 output dimension
        self.use_future_patch = getattr(configs, 'use_future_patch', 0)  # Whether to use future tokens

        # Build Chronos2 (frozen)
        self.chronos = Chronos2Pipeline.from_pretrained(self.chronos_pretrained, device_map=self.device)
        # Register chronos.model as sub-module so its params are included in model.parameters()
        self.add_module('chronos_model', self.chronos.model)
        self.chronos.model.eval()
        for param in self.chronos.model.parameters():
            param.requires_grad = False

        # num_patches = past tokens = seq_len // patch_len
        self.num_patches = self.seq_len // self.patch_len  # e.g., 336/16 = 21

        # num_output_patches for future tokens (used when use_future_patch=1)
        self.num_output_patches = self.pred_len // self.patch_len  # e.g., 96/16 = 6

        self.configs = configs  # Store for later use in forward

        if self.use_future_patch:
            # Use PatchwiseHead (like Chronos2 - per-patch prediction)
            self.patchwise_head = PatchwiseHead(
                n_vars=self.n_vars,
                d_model=self.chronos_output_dim,
                output_patch_num=self.num_output_patches,
                output_patch_size=self.patch_len,
                dropout=getattr(configs, 'head_dropout', 0.0)
            )
            self.head_type = 'patch_wise'
        else:
            # Use Flatten_Head
            self.head_nf = self.chronos_output_dim * self.num_patches
            self.flatten_head = Flatten_Head(
                individual=getattr(configs, 'individual', 0),
                n_vars=self.n_vars,
                nf=self.head_nf,
                target_window=self.pred_len,
                head_dropout=getattr(configs, 'head_dropout', 0.0)
            )
            self.head_type = 'flatten'

    def forward(self, x):
        """Forward pass.

        Args:
            x: input sequence [Batch, seq_len, n_vars]

        Returns:
            output: [Batch, pred_len, n_vars]
        """
        # Chronos expects input in (B, C, T) format
        # x: (bs, seq_len, nvars) -> (bs, nvars, seq_len)
        x_perm = x.permute(0, 2, 1)

        if self.use_future_patch:
            # Use model.encode() to get embeddings, then extract ONLY future tokens
            chronos_model = self.chronos.model
            bs, nvars, seq_len = x_perm.shape

            # Flatten bs*nvars to process all at once: (bs, nvars, seq_len) -> (bs*nvars, seq_len)
            x_flat = x_perm.reshape(-1, seq_len)  # (bs*nvars, seq_len)

            # Batch encode all at once
            encoder_out, loc_scale, _, _ = chronos_model.encode(
                context=x_flat.float().to(self.device),
                num_output_patches=self.num_output_patches,
            )
            # encoder_out.last_hidden_state: (bs*nvars, num_context + 1 + num_output, d_model)
            # loc_scale: (2, bs*nvars) - tuple of (loc, scale) per series

            # Extract ONLY the last num_output_patches (future tokens) for each
            # Shape: (bs*nvars, num_output, d_model)
            future_embeds = encoder_out.last_hidden_state[:, -self.num_output_patches:, :]

            # Reshape back to (bs, nvars, num_output, d_model)
            embeddings = future_embeds.reshape(bs, nvars, self.num_output_patches, self.chronos_output_dim)

            # embeddings shape: (bs, nvars, num_output_patches, 768)
            # Permute to match PatchwiseHead expected format: (bs, nvars, d_model, output_patch_num)
            embeddings_perm = embeddings.permute(0, 1, 3, 2)  # (bs, nvars, 768, num_output_patches)

            # Apply PatchwiseHead
            # PatchwiseHead expects: (bs, nvars, d_model, output_patch_num)
            output = self.patchwise_head(embeddings_perm)  # (bs, nvars, pred_len)

            # Denormalize: output is in normalized space, loc_scale is (loc, scale) each (bs*nvars,)
            # Reshape loc_scale for broadcasting: loc (bs*nvars,) -> (bs, nvars)
            loc = loc_scale[0].reshape(bs, nvars)  # (bs, nvars)
            scale = loc_scale[1].reshape(bs, nvars)  # (bs, nvars)
            output = output * scale.unsqueeze(-1) + loc.unsqueeze(-1)  # (bs, nvars, pred_len)
        else:
            # Get Chronos2 embeddings using embed() method
            # Returns: list of (n_variates, num_patches+2, 768) - one per batch item
            # And loc_scales: list of (2, n_variates) - tuple of (loc, scale) per batch item
            embeddings_list, loc_scales = self.chronos.embed(x_perm.cpu())

            # Stack embeddings: (bs, nvars, num_patches+2, 768)
            embeddings = torch.stack(embeddings_list, dim=0).to(self.device)

            # Only take past tokens (excluding reg token and future token)
            embeddings = embeddings[:, :, :self.num_patches, :]  # (bs, nvars, past_tokens, 768)

            # embeddings shape: (bs, nvars, num_patches, 768)
            # Permute to match Flatten_Head expected format: (bs, nvars, d_model, patch_num)
            embeddings_perm = embeddings.permute(0, 1, 3, 2)  # (bs, nvars, 768, num_patches)

            # Apply Flatten_Head
            output = self.flatten_head(embeddings_perm)  # (bs, nvars, pred_len)

            # Denormalize: stack loc_scales and apply inverse
            # loc_scales: list of (2, nvars) tuples -> (bs, 2, nvars) after stack
            loc_scale_stacked = torch.stack([ls[0] for ls in loc_scales], dim=0).to(self.device)  # (bs, nvars)
            scale_scale_stacked = torch.stack([ls[1] for ls in loc_scales], dim=0).to(self.device)  # (bs, nvars)
            output = output * scale_scale_stacked.unsqueeze(-1) + loc_scale_stacked.unsqueeze(-1)  # (bs, nvars, pred_len)

        # Final permute: (bs, nvars, pred_len) -> (bs, pred_len, nvars)
        output = output.permute(0, 2, 1)

        return output
