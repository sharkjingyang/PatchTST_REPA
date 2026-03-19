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

            all_embeddings = []

            for b in range(bs):
                var_embeddings = []
                for v in range(nvars):
                    # chronos model.encode expects context of shape (batch_size, context_length)
                    context = x_perm[b, v, :]  # (seq_len,) - single variate
                    # model.encode returns: (encoder_outputs, loc_scale, patched_future_covariates_mask, num_context_patches)
                    # encoder_outputs is Chronos2EncoderOutput with .last_hidden_state shape (batch, num_context+1+num_output, d_model)
                    encoder_out, _, _, _ = chronos_model.encode(
                        context=context.float().cpu().unsqueeze(0),  # (1, seq_len)
                        num_output_patches=self.num_output_patches,
                    )
                    # Extract ONLY the last num_output_patches (future tokens)
                    # last_hidden_state: (1, num_context + 1 + num_output, 768)
                    future_embeds = encoder_out.last_hidden_state[0, -self.num_output_patches:, :]  # (num_output, 768)
                    var_embeddings.append(future_embeds)

                # Stack: (nvars, num_output_patches, 768)
                var_embeddings = torch.stack(var_embeddings, dim=0)
                all_embeddings.append(var_embeddings)

            # Stack: (bs, nvars, num_output_patches, 768)
            embeddings = torch.stack(all_embeddings, dim=0).to(self.device)

            # embeddings shape: (bs, nvars, num_output_patches, 768)
            # Permute to match PatchwiseHead expected format: (bs, nvars, d_model, output_patch_num)
            embeddings_perm = embeddings.permute(0, 1, 3, 2)  # (bs, nvars, 768, num_output_patches)

            # Apply PatchwiseHead
            # PatchwiseHead expects: (bs, nvars, d_model, output_patch_num)
            output = self.patchwise_head(embeddings_perm)  # (bs, nvars, pred_len)
        else:
            # Get Chronos2 embeddings using embed() method
            # Returns: list of (n_variates, num_patches, 768) - one per batch item
            embeddings_list, loc_scales = self.chronos.embed(x_perm.cpu())

            # Stack embeddings: (bs, nvars, num_patches, 768)
            embeddings = torch.stack(embeddings_list, dim=0).to(self.device)

            # Only take past tokens (excluding reg token and future token)
            embeddings = embeddings[:, :, :self.num_patches, :]  # (bs, nvars, past_tokens, 768)

            # embeddings shape: (bs, nvars, num_patches, 768)
            # Permute to match Flatten_Head expected format: (bs, nvars, d_model, patch_num)
            embeddings_perm = embeddings.permute(0, 1, 3, 2)  # (bs, nvars, 768, num_patches)

            # Apply Flatten_Head
            output = self.flatten_head(embeddings_perm)  # (bs, nvars, pred_len)

        # Final permute: (bs, nvars, pred_len) -> (bs, pred_len, nvars)
        output = output.permute(0, 2, 1)

        return output
