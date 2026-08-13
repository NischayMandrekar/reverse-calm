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
  """Cross attention hook for CALM."""

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

    # Augmenting model -> Key / Value
    self.k_proj = torch.nn.Linear(
        aug_hidden_dim,
        anchor_hidden_dim,
    )
    self.v_proj = torch.nn.Linear(
        aug_hidden_dim,
        anchor_hidden_dim,
    )

    # Anchor model -> Query
    self.q_proj = torch.nn.Linear(
        anchor_hidden_dim,
        anchor_hidden_dim,
    )

    # Final attention output projection
    self.out_proj = torch.nn.Linear(
        anchor_hidden_dim,
        anchor_hidden_dim,
    )

    self.post_attention_layernorm = modeling_qwen2.Qwen2RMSNorm(
        self.embed_dim,
        eps=rms_norm_eps,
    )

    self.aug_hidden_state = None
    self.aug_mask = None
    self.attn_weights = None

  def _reshape_heads(self, x):
    """Convert [B, S, D] into [B, H, S, Dh]."""
    batch_size, seq_len, _ = x.shape

    x = x.view(
        batch_size,
        seq_len,
        self.num_heads,
        self.head_dim,
    )

    return x.transpose(1, 2)

  def forward(self, *hook_args):
    query, output = process_hook_args(*hook_args)

    assert self.aug_hidden_state is not None
    assert self.aug_mask is not None

    # ---------------------------------------------------------
    # Preserve the anchor model's dtype at the interface.
    # Bridge computation itself is performed in float32.
    # ---------------------------------------------------------

    anchor_dtype = query.dtype

    query_float = query.float()
    aug_hidden_float = self.aug_hidden_state.float()

    # ---------------------------------------------------------
    # Q from anchor
    # K/V from augmenting model
    # ---------------------------------------------------------

    q = self.q_proj(query_float)
    k = self.k_proj(aug_hidden_float)
    v = self.v_proj(aug_hidden_float)

    # ---------------------------------------------------------
    # [B, S, D] -> [B, H, S, Dh]
    # ---------------------------------------------------------

    q = self._reshape_heads(q)
    k = self._reshape_heads(k)
    v = self._reshape_heads(v)

    # ---------------------------------------------------------
    # Scaled dot-product attention
    # ---------------------------------------------------------

    scale = self.head_dim ** -0.5

    attention_scores = torch.matmul(
        q,
        k.transpose(-2, -1),
    ) * scale

    # ---------------------------------------------------------
    # Augmenting attention mask
    #
    # [B, S_aug]
    # ->
    # [B, 1, 1, S_aug]
    # ---------------------------------------------------------

    aug_mask = self.aug_mask.to(
        device=attention_scores.device,
        dtype=torch.bool,
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
    # Attention output
    # [B, H, S_anchor, Dh]
    # ->
    # [B, S_anchor, D]
    # ---------------------------------------------------------

    attn_output = torch.matmul(
        attention_weights,
        v,
    )

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
    # Output projection
    # ---------------------------------------------------------

    attn_output = self.out_proj(attn_output)

    # ---------------------------------------------------------
    # Return to anchor dtype BEFORE entering the frozen
    # Qwen transformer again.
    # ---------------------------------------------------------

    attn_output = attn_output.to(anchor_dtype)

    attn_output = self.post_attention_layernorm(
        attn_output
    )

    attn_output = attn_output.to(anchor_dtype)

    query = query.to(anchor_dtype)

    # ---------------------------------------------------------
    # Residual connection
    # ---------------------------------------------------------

    output_fin = attn_output + query

    # Final safety cast.
    output_fin = output_fin.to(anchor_dtype)

    new_output = (output_fin,) + output[1:]

    return new_output


class ExtractHiddenStateHook(torch.nn.Module):
  """Extract hidden state hook for CALM."""

  def __init__(self):
    super().__init__()
    self.hidden_state = None

  def forward(self, *hook_args):
    hidden_state, out = process_hook_args(*hook_args)
    self.hidden_state = hidden_state
    return out