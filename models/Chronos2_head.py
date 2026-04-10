__all__ = ['Model']

import torch
from torch import nn

try:
    from chronos import Chronos2Pipeline
    HAS_CHRONOS = True
except ImportError:
    HAS_CHRONOS = False

from layers.PatchTST_backbone import Flatten_Head, PatchwiseHead
from layers.RevIN import RevIN


class Model(nn.Module):
    """Chronos2 (frozen) + optional proj_down + prediction head.

    embed_type:
      "past"    — chronos.embed(x_past) past tokens → Flatten_Head
      "predict" — chronos.model.encode(x_past) future tokens → PatchwiseHead
      "future"  — chronos.embed(x_future) past tokens of future seq → Flatten_Head / PatchwiseHead
                  (teacher-forcing; requires future_seq in forward)

    --proj_down 1 inserts Linear(768→d_model) before the head for all embed_types.

    Forward shared path (all branches):
        embeddings: (bs, nvars, num_tokens, 768)
          → [proj_down] → (bs, nvars, num_tokens, d_model/768)
          → permute → (bs, nvars, d_model/768, num_tokens)
          → head → (bs, nvars, pred_len)
          → denorm → (bs, pred_len, nvars)
    """

    def __init__(self, configs, **kwargs):
        super().__init__()

        if not HAS_CHRONOS:
            raise ImportError("chronos is not installed. pip install chronos-forecasting")

        self.seq_len      = configs.seq_len
        self.pred_len     = configs.pred_len
        self.n_vars       = configs.enc_in
        self.patch_len    = configs.patch_len           # must be 16 for Chronos2
        self.device       = getattr(configs, 'device', 'cuda:0')
        self.embed_type   = getattr(configs, 'chronos_embed_type', 'past')
        self.head_type    = getattr(configs, 'head_type', 'flatten')

        self.num_patches        = self.seq_len  // self.patch_len   # e.g. 21
        self.num_output_patches = self.pred_len // self.patch_len   # e.g. 6

        # ---- Chronos2 (frozen) ----
        chronos_pretrained = getattr(configs, 'chronos_pretrained', './Chronos2')
        self.chronos = Chronos2Pipeline.from_pretrained(chronos_pretrained, device_map=self.device)
        self.add_module('chronos_model', self.chronos.model)
        self.chronos.model.eval()
        for p in self.chronos.model.parameters():
            p.requires_grad = False

        # ---- proj_down (optional, all embed_types) ----
        chronos_dim = 768
        if getattr(configs, 'proj_down', 0):
            d_model = configs.d_model
            self.proj_down = nn.Linear(chronos_dim, d_model)
            head_dim = d_model
        else:
            self.proj_down = None
            head_dim = chronos_dim

        # ---- RevIN for future mode ----
        self.revin_layer = RevIN(self.n_vars, affine=False) if self.embed_type == 'future' else None

        # ---- Head ----
        individual   = getattr(configs, 'individual', 0)
        head_dropout = getattr(configs, 'head_dropout', 0.0)

        # past → always Flatten_Head (patch-wise semantics don't hold for past→future)
        # predict → always PatchwiseHead (matches Chronos2 native design)
        # future → respects head_type
        if self.embed_type == 'past':
            self.head = Flatten_Head(
                individual=individual, n_vars=self.n_vars,
                nf=head_dim * self.num_patches,
                target_window=self.pred_len, head_dropout=head_dropout,
            )
        elif self.embed_type == 'predict':
            self.head = PatchwiseHead(
                n_vars=self.n_vars, d_model=head_dim,
                output_patch_num=self.num_output_patches,
                output_patch_size=self.patch_len, dropout=head_dropout,
            )
        else:  # 'future'
            if self.head_type == 'patch_wise':
                self.head = PatchwiseHead(
                    n_vars=self.n_vars, d_model=head_dim,
                    output_patch_num=self.num_output_patches,
                    output_patch_size=self.patch_len, dropout=head_dropout,
                )
            else:
                self.head = Flatten_Head(
                    individual=individual, n_vars=self.n_vars,
                    nf=head_dim * self.num_output_patches,
                    target_window=self.pred_len, head_dropout=head_dropout,
                )

    # ------------------------------------------------------------------
    # Embedding extraction (branch-specific) → (bs, nvars, num_tokens, 768)
    # Returns (embeddings, loc, scale); loc/scale=None for 'future' (uses RevIN)
    # ------------------------------------------------------------------
    def _extract(self, x, future_seq):
        x_perm = x.permute(0, 2, 1)  # (bs, nvars, seq_len)

        if self.embed_type == 'past':
            embeddings_list, loc_scales = self.chronos.embed(x_perm.cpu())
            embeddings = torch.stack(embeddings_list, dim=0).to(self.device)
            embeddings = embeddings[:, :, :self.num_patches, :]
            loc   = torch.stack([ls[0] for ls in loc_scales], dim=0).squeeze(-1).to(self.device)
            scale = torch.stack([ls[1] for ls in loc_scales], dim=0).squeeze(-1).to(self.device)
            return embeddings, loc, scale

        elif self.embed_type == 'predict':
            bs, nvars, seq_len = x_perm.shape
            x_flat = x_perm.reshape(-1, seq_len).float()
            encoder_out, loc_scale, _, _ = self.chronos.model.encode(
                context=x_flat.to(self.device),
                num_output_patches=self.num_output_patches,
            )
            embeds = encoder_out.last_hidden_state[:, -self.num_output_patches:, :]
            embeddings = embeds.reshape(bs, nvars, self.num_output_patches, 768)
            loc   = loc_scale[0].reshape(bs, nvars)
            scale = loc_scale[1].reshape(bs, nvars)
            return embeddings, loc, scale

        else:  # 'future'
            assert future_seq is not None, "embed_type='future' requires future_seq"
            self.revin_layer(x, 'norm')               # store x_past loc/scale in RevIN
            future_perm = future_seq.permute(0, 2, 1)
            embeddings_list, _ = self.chronos.embed(future_perm.cpu())
            embeddings = torch.stack(embeddings_list, dim=0).to(self.device)
            embeddings = embeddings[:, :, :self.num_output_patches, :]
            return embeddings, None, None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x, future_seq=None):
        """
        Args:
            x:          (bs, seq_len, n_vars)
            future_seq: (bs, pred_len, n_vars) — only for embed_type='future'
        Returns:
            (bs, pred_len, n_vars)
        """
        embeddings, loc, scale = self._extract(x, future_seq)
        # embeddings: (bs, nvars, num_tokens, 768)

        # proj_down (shared)
        if self.proj_down is not None:
            embeddings = self.proj_down(embeddings)

        # head (shared): needs (bs, nvars, dim, num_tokens)
        output = self.head(embeddings.permute(0, 1, 3, 2))  # (bs, nvars, pred_len)

        # denorm
        if self.embed_type == 'future':
            output = output.permute(0, 2, 1)            # (bs, pred_len, nvars)
            return self.revin_layer(output, 'denorm')
        else:
            output = output * scale.unsqueeze(-1) + loc.unsqueeze(-1)
            return output.permute(0, 2, 1)              # (bs, pred_len, nvars)
