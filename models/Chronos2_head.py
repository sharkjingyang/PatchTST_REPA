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

    Flow:
        Input x: (bs, seq_len, nvars)
            ↓ Chronos2.embed(x) - frozen
        Feature: (bs, nvars, num_patches, 768)  [num_patches = seq_len/16]
            ↓ permute(0,1,3,2): (bs, nvars, 768, num_patches)
            ↓ Flatten_Head (individual=True)
            ↓ flatten: (bs*nvars, 768*num_patches)
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
        self.head_nf = self.chronos_output_dim * self.num_patches  # e.g., 768 * 21 = 16128
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

        # Get Chronos2 embeddings
        # Returns: list of (n_variates, num_patches, 768) - one per batch item
        # Chronos2 embed output: past tokens + reg token + future token
        embeddings, loc_scales = self.chronos.embed(x_perm.cpu())

        # Stack embeddings: (bs, nvars, num_patches, 768)
        embeddings = torch.stack(embeddings, dim=0).to(self.device)

        # Only take past tokens (excluding reg token and future token)
        # past_tokens = seq_len // patch_len
        embeddings = embeddings[:, :, :self.num_patches, :]  # (bs, nvars, past_tokens, 768)

        # embeddings shape: (bs, nvars, num_patches, 768)
        # Permute to match Flatten_Head expected format: (bs, nvars, d_model, patch_num)
        embeddings_perm = embeddings.permute(0, 1, 3, 2)  # (bs, nvars, 768, num_patches)

        # Apply Flatten_Head
        # Flatten_Head expects: (bs, nvars, d_model, patch_num)
        output = self.flatten_head(embeddings_perm)  # (bs, nvars, pred_len)

        # Final permute: (bs, nvars, pred_len) -> (bs, pred_len, nvars)
        output = output.permute(0, 2, 1)

        return output
