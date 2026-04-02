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

    embed_type controls how Chronos2 is used:

    "past":
        chronos.embed(x) → past tokens (seq_len // patch_len) → Flatten_Head → output
        [same as old use_future_patch=0]

    "predict":
        chronos.model.encode(x, num_output_patches) → future tokens (pred_len // patch_len)
        → PatchwiseHead → output
        [same as old use_future_patch=1]

    "future":
        Training:  chronos.embed(future_seq) → future tokens (pred_len // patch_len)
                   → Flatten_Head → output  (teacher-forcing with ground-truth future)
        Inference: chronos.model.encode(x, num_output_patches) → last num_output_patches tokens
                   → same Flatten_Head → output  (no ground-truth available)
    """

    def __init__(self, configs, **kwargs):
        super().__init__()

        if not HAS_CHRONOS:
            raise ImportError("chronos is not installed. Please install it with: pip install chronos-forecasting")

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.n_vars = configs.enc_in
        self.patch_len = configs.patch_len  # Chronos uses patch_len=16

        self.device = getattr(configs, 'device', 'cuda:0')
        self.chronos_pretrained = getattr(configs, 'chronos_pretrained', './Chronos2')
        self.chronos_output_dim = 768
        self.embed_type = getattr(configs, 'chronos_embed_type', 'past')

        # Build Chronos2 (frozen)
        self.chronos = Chronos2Pipeline.from_pretrained(self.chronos_pretrained, device_map=self.device)
        self.add_module('chronos_model', self.chronos.model)
        self.chronos.model.eval()
        for param in self.chronos.model.parameters():
            param.requires_grad = False

        # num_patches: past tokens = seq_len // patch_len (e.g., 336/16 = 21)
        self.num_patches = self.seq_len // self.patch_len
        # num_output_patches: future tokens = pred_len // patch_len (e.g., 96/16 = 6)
        self.num_output_patches = self.pred_len // self.patch_len

        individual = getattr(configs, 'individual', 0)
        head_dropout = getattr(configs, 'head_dropout', 0.0)
        self.head_type = getattr(configs, 'head_type', 'flatten')

        # proj_down: Linear(768 → d_model) bottleneck before head (only for embed_type='future')
        self.proj_down = None
        if getattr(configs, 'proj_down', 0) and self.embed_type == 'future':
            d_model = configs.d_model
            self.proj_down = nn.Linear(self.chronos_output_dim, d_model)
            head_input_dim = d_model
        else:
            head_input_dim = self.chronos_output_dim

        if self.embed_type == 'predict':
            # PatchwiseHead: per-patch prediction (like Chronos2 native)
            self.patchwise_head = PatchwiseHead(
                n_vars=self.n_vars,
                d_model=self.chronos_output_dim,
                output_patch_num=self.num_output_patches,
                output_patch_size=self.patch_len,
                dropout=head_dropout
            )
        elif self.embed_type == 'future':
            if self.head_type == 'patch_wise':
                # PatchwiseHead on ground-truth future tokens
                self.patchwise_head = PatchwiseHead(
                    n_vars=self.n_vars,
                    d_model=head_input_dim,
                    output_patch_num=self.num_output_patches,
                    output_patch_size=self.patch_len,
                    dropout=head_dropout
                )
            else:  # flatten
                self.head_nf = head_input_dim * self.num_output_patches
                self.flatten_head = Flatten_Head(
                    individual=individual,
                    n_vars=self.n_vars,
                    nf=self.head_nf,
                    target_window=self.pred_len,
                    head_dropout=head_dropout
                )
        else:  # "past"
            # Flatten_Head on past tokens (num_patches tokens)
            self.head_nf = self.chronos_output_dim * self.num_patches
            self.flatten_head = Flatten_Head(
                individual=individual,
                n_vars=self.n_vars,
                nf=self.head_nf,
                target_window=self.pred_len,
                head_dropout=head_dropout
            )

    def forward(self, x, future_seq=None):
        """Forward pass.

        Args:
            x:          past input sequence, (bs, seq_len, n_vars)
            future_seq: ground-truth future sequence, (bs, pred_len, n_vars)
                        only used in embed_type="future" during training.
                        Ignored for "past" and "predict" modes.

        Returns:
            output: (bs, pred_len, n_vars)
        """
        # Chronos expects (B, C, T)
        x_perm = x.permute(0, 2, 1)  # (bs, nvars, seq_len)

        if self.embed_type == 'predict':
            # ---- "predict" mode: encode future tokens ----
            bs, nvars, seq_len = x_perm.shape
            x_flat = x_perm.reshape(-1, seq_len)  # (bs*nvars, seq_len)

            encoder_out, loc_scale, _, _ = self.chronos.model.encode(
                context=x_flat.float().to(self.device),
                num_output_patches=self.num_output_patches,
            )
            # last num_output_patches tokens
            future_embeds = encoder_out.last_hidden_state[:, -self.num_output_patches:, :]
            embeddings = future_embeds.reshape(bs, nvars, self.num_output_patches, self.chronos_output_dim)
            embeddings_perm = embeddings.permute(0, 1, 3, 2)  # (bs, nvars, 768, num_output_patches)

            output = self.patchwise_head(embeddings_perm)  # (bs, nvars, pred_len)

            loc = loc_scale[0].reshape(bs, nvars)
            scale = loc_scale[1].reshape(bs, nvars)
            output = output * scale.unsqueeze(-1) + loc.unsqueeze(-1)

        elif self.embed_type == 'future':
            # ---- "future" mode: always embed ground-truth future sequence ----
            # future_seq: (bs, pred_len, nvars) → (bs, nvars, pred_len)
            assert future_seq is not None, "embed_type='future' requires future_seq in forward()"
            future_perm = future_seq.permute(0, 2, 1)
            embeddings_list, loc_scales = self.chronos.embed(future_perm.cpu())

            # Stack: (bs, nvars, num_tokens+2, 768); take first num_output_patches
            embeddings = torch.stack(embeddings_list, dim=0).to(self.device)
            assert embeddings.shape[2] >= self.num_output_patches, (
                f"embed_type='future': chronos.embed() returned {embeddings.shape[2]} tokens "
                f"but need {self.num_output_patches} (pred_len={self.pred_len}, patch_len={self.patch_len})"
            )
            embeddings = embeddings[:, :, :self.num_output_patches, :]  # (bs, nvars, num_output_patches, 768)

            if self.proj_down is not None:
                # proj_down: (bs, nvars, num_output_patches, 768) → (bs, nvars, num_output_patches, d_model)
                embeddings = self.proj_down(embeddings)
            embeddings_perm = embeddings.permute(0, 1, 3, 2)  # (bs, nvars, d_model/768, num_output_patches)

            if self.head_type == 'patch_wise':
                output = self.patchwise_head(embeddings_perm)  # (bs, nvars, pred_len)
            else:
                output = self.flatten_head(embeddings_perm)    # (bs, nvars, pred_len)

            loc_scale_stacked = torch.stack([ls[0] for ls in loc_scales], dim=0).to(self.device)
            scale_stacked = torch.stack([ls[1] for ls in loc_scales], dim=0).to(self.device)
            loc = loc_scale_stacked.squeeze(-1)
            scale = scale_stacked.squeeze(-1)
            output = output * scale.unsqueeze(-1) + loc.unsqueeze(-1)

        else:  # "past"
            # ---- "past" mode: embed past tokens ----
            embeddings_list, loc_scales = self.chronos.embed(x_perm.cpu())

            embeddings = torch.stack(embeddings_list, dim=0).to(self.device)
            embeddings = embeddings[:, :, :self.num_patches, :]  # (bs, nvars, num_patches, 768)
            embeddings_perm = embeddings.permute(0, 1, 3, 2)  # (bs, nvars, 768, num_patches)

            output = self.flatten_head(embeddings_perm)  # (bs, nvars, pred_len)

            loc_scale_stacked = torch.stack([ls[0] for ls in loc_scales], dim=0).to(self.device)
            scale_stacked = torch.stack([ls[1] for ls in loc_scales], dim=0).to(self.device)
            loc = loc_scale_stacked.squeeze(-1)
            scale = scale_stacked.squeeze(-1)
            output = output * scale.unsqueeze(-1) + loc.unsqueeze(-1)

        # (bs, nvars, pred_len) → (bs, pred_len, nvars)
        return output.permute(0, 2, 1)
