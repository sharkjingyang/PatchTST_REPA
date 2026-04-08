__all__ = ['MiniChronos2_backbone']

import torch
import torch.nn as nn

from chronos.chronos_bolt import Patch, InstanceNorm
from layers.PatchTST_backbone import PatchwiseHead, Flatten_Head


class MiniChronos2Encoder(nn.Module):
    """Small Chronos2-style encoder (channel-independent, no GroupSelfAttention).

    Mimics Chronos2's encode(x_past, num_output_patches) logic:
      1. InstanceNorm x_past
      2. Patch(16, 16) → context patches
      3. Time encoding: context [-seq_len, ..., -1] / scale, future [0, ..., pred_len-1] / scale
      4. Concatenate [time_enc | values | mask] per patch → 48-dim features
      5. input_proj Linear(48 → d_model)
      6. Concatenate [context_emb | masked_future_emb] → TransformerEncoder
      7. Return last num_output_patches hidden states as future token representations

    The masked future tokens (all-zero values + all-zero mask) use positive time encoding,
    matching Chronos2's convention. The alignment between student and teacher representations
    is semantically sound: both encode "future predictions from past context".
    """

    PATCH_SIZE = 16

    def __init__(self, d_model: int = 128, n_layers: int = 3, n_heads: int = 8,
                 d_ff: int = 256, dropout: float = 0.0, time_encoding_scale: int = 336):
        super().__init__()
        self.d_model = d_model
        self.patch_size = self.PATCH_SIZE
        self.time_encoding_scale = time_encoding_scale

        self.instance_norm = InstanceNorm(use_arcsinh=False)
        self.patch = Patch(patch_size=self.PATCH_SIZE, patch_stride=self.PATCH_SIZE)

        # [time_enc(16) | values(16) | mask(16)] → d_model
        self.input_proj = nn.Linear(self.PATCH_SIZE * 3, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,   # pre-norm, same as Chronos2
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )

    def _prepare_context(self, x_flat: torch.Tensor):
        """Prepare context patches with time encoding.

        Args:
            x_flat: (N, seq_len)
        Returns:
            patched:   (N, num_ctx_patches, patch_size*3)
            attn_mask: (N, num_ctx_patches)  True = valid
            loc_scale: ((N,1), (N,1))
        """
        N = x_flat.shape[0]

        normalized, loc_scale = self.instance_norm(x_flat)   # normalize along seq dim

        patched_values = self.patch(normalized)               # (N, num_patches, 16)
        patched_mask   = torch.ones_like(patched_values)      # all valid (no NaN assumed)

        num_patches = patched_values.shape[1]
        final_len   = num_patches * self.patch_size

        # Time encoding: [-final_len, ..., -1] / scale → (1, num_patches, 16)
        time_enc = (
            torch.arange(-final_len, 0, dtype=torch.float32, device=x_flat.device)
            .div(self.time_encoding_scale)
            .reshape(1, num_patches, self.patch_size)
            .expand(N, -1, -1)
            .to(normalized.dtype)
        )

        patched = torch.cat([time_enc, patched_values, patched_mask], dim=-1)  # (N, P, 48)

        attn_mask = torch.ones(N, num_patches, dtype=torch.bool, device=x_flat.device)
        return patched, attn_mask, loc_scale

    def _prepare_future(self, N: int, num_output_patches: int,
                        device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Prepare masked future tokens with time encoding.

        Values and mask are all zeros (masked / unknown future).
        Time encoding is positive: [0, 1, ..., pred_len-1] / scale.

        Returns:
            (N, num_output_patches, patch_size*3)
        """
        final_len = num_output_patches * self.patch_size

        time_enc = (
            torch.arange(0, final_len, dtype=torch.float32, device=device)
            .div(self.time_encoding_scale)
            .reshape(1, num_output_patches, self.patch_size)
            .expand(N, -1, -1)
            .to(dtype)
        )
        zeros = torch.zeros(N, num_output_patches, self.patch_size, device=device, dtype=dtype)

        return torch.cat([time_enc, zeros, zeros], dim=-1)  # (N, num_output_patches, 48)

    def forward(self, x_flat: torch.Tensor, num_output_patches: int):
        """
        Args:
            x_flat:             (N, seq_len)  N = bs * nvars
            num_output_patches: int
        Returns:
            future_tokens: (N, num_output_patches, d_model)
            loc_scale:     ((N,1), (N,1))
        """
        N = x_flat.shape[0]

        ctx_raw, ctx_attn_mask, loc_scale = self._prepare_context(x_flat)
        fut_raw = self._prepare_future(N, num_output_patches, x_flat.device, x_flat.dtype)

        ctx_emb = self.input_proj(ctx_raw)   # (N, num_ctx_patches, d_model)
        fut_emb = self.input_proj(fut_raw)   # (N, num_output_patches, d_model)

        combined = torch.cat([ctx_emb, fut_emb], dim=1)  # (N, total_patches, d_model)

        # src_key_padding_mask: True = IGNORE (PyTorch convention, opposite of attn_mask)
        ctx_pad = ~ctx_attn_mask                                                    # (N, num_ctx_patches)
        fut_pad = torch.zeros(N, num_output_patches, dtype=torch.bool, device=x_flat.device)
        key_padding_mask = torch.cat([ctx_pad, fut_pad], dim=1)                     # (N, total)

        out = self.encoder(combined, src_key_padding_mask=key_padding_mask)         # (N, total, d_model)

        future_tokens = out[:, -num_output_patches:, :]  # (N, num_output_patches, d_model)
        return future_tokens, loc_scale


class MiniChronos2_backbone(nn.Module):
    """Backbone for PatchTST_decoder (MiniChronos2 architecture).

    Student path:
        x (bs, nvars, seq_len)
        → MiniChronos2Encoder (channel-independent, bs*nvars as batch)
        → z_student (bs, nvars, output_patch_num, d_model)
        → PatchwiseHead / Flatten_Head
        → InstanceNorm.inverse (uses x_past loc/scale)
        → pred (bs, nvars, pred_len)

    Teacher path (training only):
        z_chron (bs, nvars, output_patch_num, 768)   ← from frozen Chronos2.encode(x_past)
        → proj_down Linear(768 → d_model)
        → z_teacher (bs, nvars, output_patch_num, d_model)
        → teacher_head
        → InstanceNorm.inverse (same loc_scale as student — consistent denorm)
        → pred_teacher (bs, nvars, pred_len)
    """

    PATCH_SIZE = 16

    def __init__(self, c_in: int, context_window: int, target_window: int,
                 n_layers: int = 3, d_model: int = 128, n_heads: int = 8,
                 d_ff: int = 256, dropout: float = 0.0, head_dropout: float = 0.0,
                 head_type: str = 'patch_wise'):
        super().__init__()

        assert target_window % self.PATCH_SIZE == 0, (
            f"pred_len must be divisible by {self.PATCH_SIZE} (Chronos2 patch size), "
            f"got pred_len={target_window}"
        )

        self.output_patch_num = target_window // self.PATCH_SIZE
        self.patch_len = self.PATCH_SIZE     # alias for _print_parameter_stats compatibility
        self.patch_size = self.PATCH_SIZE
        self.head_type = head_type
        self.d_model = d_model
        self.time_encoding_scale = context_window

        self.mini_encoder = MiniChronos2Encoder(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            time_encoding_scale=context_window,
        )

        # Teacher path: project large Chronos2 (768-dim) to student space (d_model)
        self.proj_down = nn.Linear(768, d_model)

        def _build_head():
            if head_type == 'patch_wise':
                return PatchwiseHead(
                    n_vars=c_in,
                    d_model=d_model,
                    output_patch_num=self.output_patch_num,
                    output_patch_size=self.PATCH_SIZE,
                    dropout=head_dropout,
                )
            else:  # flatten
                return Flatten_Head(
                    individual=False,
                    n_vars=c_in,
                    nf=d_model * self.output_patch_num,
                    target_window=target_window,
                    head_dropout=head_dropout,
                )

        self.head = _build_head()
        self.teacher_head = _build_head()

        # Reference to instance_norm for inverse transform
        self.instance_norm = self.mini_encoder.instance_norm

    def _denorm(self, pred_norm: torch.Tensor,
                loc_scale: tuple, bs: int, nvars: int) -> torch.Tensor:
        """Apply InstanceNorm inverse.
        pred_norm: (bs, nvars, pred_len)
        loc_scale: ((N,1), (N,1)) where N = bs*nvars
        """
        loc, scale = loc_scale
        loc   = loc.reshape(bs, nvars, 1)
        scale = scale.reshape(bs, nvars, 1)
        return pred_norm * scale + loc

    def forward_student(self, x: torch.Tensor):
        """
        Args:
            x: (bs, nvars, seq_len)
        Returns:
            pred:      (bs, nvars, pred_len)   denormalized
            z_student: (bs, nvars, output_patch_num, d_model)
            loc_scale: tuple of (bs*nvars, 1) tensors
        """
        bs, nvars, seq_len = x.shape
        x_flat = x.reshape(bs * nvars, seq_len)

        future_tokens, loc_scale = self.mini_encoder(x_flat, self.output_patch_num)
        # future_tokens: (bs*nvars, output_patch_num, d_model)

        z_student = future_tokens.reshape(bs, nvars, self.output_patch_num, self.d_model)

        # Head expects (bs, nvars, d_model, output_patch_num)
        z_for_head = z_student.permute(0, 1, 3, 2)
        pred_norm = self.head(z_for_head)  # (bs, nvars, pred_len)

        pred = self._denorm(pred_norm, loc_scale, bs, nvars)
        return pred, z_student, loc_scale

    def forward_teacher(self, z_chron: torch.Tensor, loc_scale: tuple):
        """
        Args:
            z_chron:   (bs, nvars, output_patch_num, 768)  from Chronos2.encode(x_past)
            loc_scale: from student's InstanceNorm (x_past statistics)
        Returns:
            pred:      (bs, nvars, pred_len)   denormalized with x_past stats
            z_teacher: (bs, nvars, output_patch_num, d_model)
        """
        bs, nvars = z_chron.shape[:2]

        z_teacher = self.proj_down(z_chron)          # (bs, nvars, output_patch_num, d_model)

        z_for_head = z_teacher.permute(0, 1, 3, 2)  # (bs, nvars, d_model, output_patch_num)
        pred_norm = self.teacher_head(z_for_head)    # (bs, nvars, pred_len)

        pred = self._denorm(pred_norm, loc_scale, bs, nvars)
        return pred, z_teacher

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inference only (student path)."""
        pred, _, _ = self.forward_student(x)
        return pred
