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
  """Cross attention hook for CALM with dynamic dtype alignment."""

  def __init__(
      self,
      anchor_hidden_dim: int,
      aug_hidden_dim: int,
      num_heads: int,
      rms_norm_eps: float = 1e-6,
  ):
    super().__init__()
    self.proj = torch.nn.Linear(aug_hidden_dim, anchor_hidden_dim)
    self.embed_dim = anchor_hidden_dim
    self.num_heads = num_heads
    self.post_attention_layernorm = modeling_qwen2.Qwen2RMSNorm(
        self.embed_dim, eps=rms_norm_eps
    )
    self.cross_attention = torch.nn.MultiheadAttention(
        self.embed_dim,
        num_heads,
        kdim=self.embed_dim,
        vdim=self.embed_dim,
        batch_first=True,
    )
    self.aug_hidden_state = None
    self.aug_mask = None
    self.attn_weights = None

  def forward(self, *hook_args):
    query, output = process_hook_args(*hook_args)
    assert self.aug_hidden_state is not None
    assert self.aug_mask is not None

    # Dynamically cast hook layers to match input dtypes (fixes Half vs Float mismatch permanently)
    if self.proj.weight.dtype != self.aug_hidden_state.dtype:
      self.proj.to(self.aug_hidden_state.dtype)
    if self.post_attention_layernorm.weight.dtype != query.dtype:
      self.post_attention_layernorm.to(query.dtype)
    if hasattr(self.cross_attention, 'in_proj_weight') and self.cross_attention.in_proj_weight is not None:
      if self.cross_attention.in_proj_weight.dtype != query.dtype:
        self.cross_attention.to(query.dtype)

    key = self.proj(self.aug_hidden_state)
    value = self.proj(self.aug_hidden_state)

    self.aug_mask = self.aug_mask.float()
    attn_output, attn_weights = self.cross_attention(
        query, key, value, need_weights=True
    )
    self.attn_weights = attn_weights

    attn_output = self.post_attention_layernorm(attn_output)
    output_fin = attn_output + query
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