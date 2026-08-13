# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Layer operation classes for CALM."""

from typing import Union

import torch
from transformers.models.qwen2 import modeling_qwen2


def freeze_model(model):
  """Freezes the model."""
  for param in model.parameters():
    param.requires_grad = False


def process_hook_args(
    model: torch.nn.Module,
    inp: Union[torch.Tensor, tuple[torch.Tensor, ...]],
    out: Union[torch.Tensor, tuple[torch.Tensor, ...]],
):
  """Extracts the main output tensor from a PyTorch hook output."""

  anchor_hidden_state = out[0] if isinstance(out, tuple) else out
  query = anchor_hidden_state

  return query, out


class CrossAttentionHook(torch.nn.Module):
  """Cross-attention bridge between augmenting and anchor models."""

  def __init__(
      self,
      anchor_hidden_dim: int,
      aug_hidden_dim: int,
      num_heads: int,
      rms_norm_eps: float = 1e-6,
  ):
    super().__init__()

    if anchor_hidden_dim % num_heads != 0:
      raise ValueError(
          f"anchor_hidden_dim ({anchor_hidden_dim}) must be divisible "
          f"by num_heads ({num_heads})."
      )

    self.embed_dim = anchor_hidden_dim
    self.num_heads = num_heads
    self.head_dim = anchor_hidden_dim // num_heads

    # Augmenting model -> K
    self.k_proj = torch.nn.Linear(
        aug_hidden_dim,
        anchor_hidden_dim,
    )

    # Augmenting model -> V
    self.v_proj = torch.nn.Linear(
        aug_hidden_dim,
        anchor_hidden_dim,
    )

    # Anchor model -> Q
    self.q_proj = torch.nn.Linear(
        anchor_hidden_dim,
        anchor_hidden_dim,
    )

    # Attention output projection.
    self.out_proj = torch.nn.Linear(
        anchor_hidden_dim,
        anchor_hidden_dim,
    )

    self.post_attention_layernorm = modeling_qwen2.Qwen2RMSNorm(
        self.embed_dim,
        eps=rms_norm_eps,
    )

    # Runtime state populated by CALM._forward_aug().
    self.aug_hidden_state = None
    self.aug_mask = None
    self.attn_weights = None

  def _reshape_heads(self, x):
    """
    Convert:

        [B, S, D]

    into:

        [B, H, S, Dh]
    """

    batch_size, seq_len, hidden_dim = x.shape

    if hidden_dim != self.embed_dim:
      raise RuntimeError(
          "Unexpected hidden dimension in cross-attention: "
          f"got {hidden_dim}, expected {self.embed_dim}"
      )

    x = x.view(
        batch_size,
        seq_len,
        self.num_heads,
        self.head_dim,
    )

    return x.transpose(1, 2)

  def forward(self, *hook_args):
    query, output = process_hook_args(*hook_args)

    if self.aug_hidden_state is None:
      raise RuntimeError(
          "CrossAttentionHook received no augmenting hidden state."
      )

    if self.aug_mask is None:
      raise RuntimeError(
          "CrossAttentionHook received no augmenting attention mask."
      )

    # ---------------------------------------------------------
    # Save the dtype expected by the frozen anchor transformer.
    # ---------------------------------------------------------

    anchor_dtype = query.dtype

    # Bridge calculations are performed in FP32.
    query_float = query.float()
    aug_hidden_float = self.aug_hidden_state.float()

    # ---------------------------------------------------------
    # Q from anchor.
    # K/V from augmenting model.
    # ---------------------------------------------------------

    q = self.q_proj(query_float)
    k = self.k_proj(aug_hidden_float)
    v = self.v_proj(aug_hidden_float)

    # ---------------------------------------------------------
    # Shape:
    #
    # Q: [B, H, S_anchor, Dh]
    # K: [B, H, S_aug,    Dh]
    # V: [B, H, S_aug,    Dh]
    # ---------------------------------------------------------

    q = self._reshape_heads(q)
    k = self._reshape_heads(k)
    v = self._reshape_heads(v)

    # ---------------------------------------------------------
    # Scaled dot-product attention.
    #
    # [B,H,S_anchor,Dh]
    # ×
    # [B,H,Dh,S_aug]
    #
    # =
    #
    # [B,H,S_anchor,S_aug]
    # ---------------------------------------------------------

    scale = self.head_dim ** -0.5

    attention_scores = torch.matmul(
        q,
        k.transpose(-2, -1),
    ) * scale

    # ---------------------------------------------------------
    # Apply augmenting-model padding mask.
    #
    # aug_mask:
    # [B, S_aug]
    #
    # becomes:
    # [B, 1, 1, S_aug]
    # ---------------------------------------------------------

    aug_mask = self.aug_mask.to(
        device=attention_scores.device,
        dtype=torch.bool,
    )

    if aug_mask.dim() != 2:
      raise RuntimeError(
          "Expected augmenting attention mask to have shape [B, S]. "
          f"Got {aug_mask.shape}."
      )

    if aug_mask.shape[0] != attention_scores.shape[0]:
      raise RuntimeError(
          "Cross-attention batch mismatch: "
          f"query batch={attention_scores.shape[0]}, "
          f"mask batch={aug_mask.shape[0]}."
      )

    if aug_mask.shape[1] != attention_scores.shape[-1]:
      raise RuntimeError(
          "Cross-attention sequence mismatch: "
          f"key sequence={attention_scores.shape[-1]}, "
          f"mask sequence={aug_mask.shape[1]}."
      )

    attention_scores = attention_scores.masked_fill(
        ~aug_mask[:, None, None, :],
        torch.finfo(attention_scores.dtype).min,
    )

    attention_weights = torch.softmax(
        attention_scores,
        dim=-1,
    )

    self.attn_weights = attention_weights.detach()

    # ---------------------------------------------------------
    # Weighted V.
    #
    # [B,H,S_anchor,S_aug]
    # ×
    # [B,H,S_aug,Dh]
    #
    # =
    #
    # [B,H,S_anchor,Dh]
    # ---------------------------------------------------------

    attn_output = torch.matmul(
        attention_weights,
        v,
    )

    # ---------------------------------------------------------
    # [B,H,S,Dh]
    # ->
    # [B,S,D]
    # ---------------------------------------------------------

    attn_output = (
        attn_output
        .transpose(1, 2)
        .contiguous()
    )

    batch_size, query_len, _, _ = attn_output.shape

    attn_output = attn_output.view(
        batch_size,
        query_len,
        self.embed_dim,
    )

    # ---------------------------------------------------------
    # Output projection.
    # ---------------------------------------------------------

    attn_output = self.out_proj(attn_output)

    # ---------------------------------------------------------
    # Convert back to anchor dtype before returning through
    # the anchor transformer.
    # ---------------------------------------------------------

    attn_output = attn_output.to(anchor_dtype)

    attn_output = self.post_attention_layernorm(
        attn_output
    )

    attn_output = attn_output.to(anchor_dtype)

    # ---------------------------------------------------------
    # Residual connection.
    #
    # Anchor hidden
    #       +
    # Cross-attention information
    # ---------------------------------------------------------

    output_fin = query + attn_output

    # Final safety cast.
    output_fin = output_fin.to(anchor_dtype)

    # Decoder layer outputs are tuples.
    new_output = (output_fin,) + output[1:]

    return new_output


class ExtractHiddenStateHook(torch.nn.Module):
  """Extract hidden state hook for CALM."""

  def __init__(self):
    super().__init__()

    self.hidden_state = None

  def forward(self, *hook_args):
    hidden_state, out = process_hook_args(
        *hook_args
    )

    self.hidden_state = hidden_state

    return out