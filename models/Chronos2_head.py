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

from layers.PatchTST_backbone import Flatten_Head


class Model(nn.Module):
    """Chronos2 + Flatten_Head model for time series forecasting.

    This model uses Chronos2 (frozen) to extract features from the input sequence,
    then uses Flatten_Head to make predictions.

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
            ↓ permute(0,1,3,2): (bs, nvars, 768, num_total_patches)
            ↓ Flatten_Head (individual=True)
            ↓ flatten: (bs*nvars, 768*num_total_patches)
            ↓ linear: (bs*nvars, pred_len)
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
        self.use_future_patch = getattr(configs, 'use_future_patch', 0)  # Whether to use future tokens from encode()

        # Build Chronos2 (frozen)
        self.chronos = Chronos2Pipeline.from_pretrained(self.chronos_pretrained, device_map=self.device)
        # Register chronos.model as sub-module so its params are included in model.parameters()
        self.add_module('chronos_model', self.chronos.model)
        self.chronos.model.eval()
        for param in self.chronos.model.parameters():
            param.requires_grad = False

        # num_patches = past tokens = seq_len // patch_len
        # Chronos2 embed output: past tokens + reg token + future token
        self.num_patches = self.seq_len // self.patch_len  # e.g., 336/16 = 21

        # num_output_patches for future tokens (used when use_future_patch=1)
        self.num_output_patches = self.pred_len // self.patch_len  # e.g., 96/16 = 6

        if self.use_future_patch:
            # Total patches: past + REG + future
            # Chronos2 embed output: num_context_patches + 1 (REG) + num_output_patches
            self.total_patches = self.num_patches + 1 + self.num_output_patches
            self.head_nf = self.chronos_output_dim * self.total_patches
        else:
            self.head_nf = self.chronos_output_dim * self.num_patches

        self.configs = configs  # Store for later use in forward

        # Flatten_Head
        self.flatten_head = Flatten_Head(
            individual=getattr(configs, 'individual', 0),
            n_vars=self.n_vars,
            nf=self.head_nf,
            target_window=self.pred_len,
            head_dropout=getattr(configs, 'head_dropout', 0.0)
        )

        self.head_type = 'flatten'  # Always use flatten head for this model

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
            # Use model.encode() to get embeddings including future tokens
            # Get Chronos model
            chronos_model = self.chronos.model

            # Prepare input: chronos model expects (batch, context_length)
            # x_perm: (bs, nvars, seq_len) -> we need to process each variate separately
            # OR treat each variate as a separate batch item
            bs, nvars, seq_len = x_perm.shape

            # For multivariate: process each variate independently or stack them
            # Chronos encode expects (batch, context) - we'll treat each (1, seq_len) as batch=1
            all_embeddings = []

            for b in range(bs):
                # Process each batch item
                # x_perm[b]: (nvars, seq_len) - treat each var as separate context
                var_embeddings = []
                for v in range(nvars):
                    context = x_perm[b, v:v+1, :]  # (1, seq_len)
                    # model.encode returns: (batch=1, num_patches + 1 + num_output_patches, d_model)
                    # For each variate, we get embeddings
                    encoder_out, (locs, scales) = chronos_model.encode(
                        context=context.float().cpu(),
                        num_output_patches=self.num_output_patches,
                    )
                    # encoder_out: (1, num_context_patches + 1 + num_output_patches, 768)
                    var_embeddings.append(encoder_out[0])  # (num_patches + 1 + num_output, 768)

                # Stack: (nvars, num_patches + 1 + num_output, 768)
                var_embeddings = torch.stack(var_embeddings, dim=0)
                all_embeddings.append(var_embeddings)

            # Stack: (bs, nvars, num_patches + 1 + num_output, 768)
            embeddings = torch.stack(all_embeddings, dim=0).to(self.device)
        else:
            # Get Chronos2 embeddings using embed() method
            # Returns: list of (n_variates, num_patches, 768) - one per batch item
            # Chronos2 embed output: past tokens + reg token + future token
            embeddings_list, loc_scales = self.chronos.embed(x_perm.cpu())

            # Stack embeddings: (bs, nvars, num_patches, 768)
            embeddings = torch.stack(embeddings_list, dim=0).to(self.device)

            # Only take past tokens (excluding reg token and future token)
            # past_tokens = seq_len // patch_len
            embeddings = embeddings[:, :, :self.num_patches, :]  # (bs, nvars, past_tokens, 768)

        # embeddings shape: (bs, nvars, num_patches, 768) or (bs, nvars, total_patches, 768)
        # Permute to match Flatten_Head expected format: (bs, nvars, d_model, patch_num)
        embeddings_perm = embeddings.permute(0, 1, 3, 2)  # (bs, nvars, 768, num_patches)

        # Apply Flatten_Head
        # Flatten_Head expects: (bs, nvars, d_model, patch_num)
        output = self.flatten_head(embeddings_perm)  # (bs, nvars, pred_len)

        # Final permute: (bs, nvars, pred_len) -> (bs, pred_len, nvars)
        output = output.permute(0, 2, 1)

        return output
