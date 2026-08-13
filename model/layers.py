# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""CALM bridge layers for Qwen2 anchor + DeepSeek augmenting model."""

from typing import Union

import torch
from transformers.models.qwen2 import modeling_qwen2


def freeze_model(model):
  for param in model.parameters():
    param.requires_grad = False


def process_hook_args(
    model: torch.nn.Module,
    inp: Union[torch.Tensor, tuple[torch.Tensor, ...]],
    out: Union[torch.Tensor, tuple[torch.Tensor, ...]],
):
  del model, inp
  hidden = out[0] if isinstance(out, tuple) else out
  return hidden, out


def _make_anchor_norm(hidden_dim, eps, anchor_config):
  model_type = getattr(anchor_config, "model_type", "")
  if model_type == "qwen2":
    return modeling_qwen2.Qwen2RMSNorm(hidden_dim, eps=eps)
  return torch.nn.RMSNorm(hidden_dim, eps=eps)


class CrossAttentionHook(torch.nn.Module):
  """Trainable bridge: DeepSeek K/V -> Qwen2 anchor query."""

  def __init__(
      self,
      anchor_hidden_dim,
      aug_hidden_dim,
      num_heads=1,
      rms_norm_eps=1e-6,
      anchor_config=None,
  ):
    super().__init__()

    if anchor_hidden_dim % num_heads != 0:
      raise ValueError(
          f"anchor_hidden_dim={anchor_hidden_dim} is not divisible by "
          f"num_heads={num_heads}"
      )

    self.embed_dim = anchor_hidden_dim
    self.num_heads = num_heads

    # DeepSeek hidden dimension (3584) -> Konkani-Qwen hidden dimension (1536).
    self.proj = torch.nn.Linear(aug_hidden_dim, anchor_hidden_dim)

    self.cross_attention = torch.nn.MultiheadAttention(
        embed_dim=anchor_hidden_dim,
        num_heads=num_heads,
        kdim=anchor_hidden_dim,
        vdim=anchor_hidden_dim,
        batch_first=True,
    )

    self.post_attention_layernorm = _make_anchor_norm(
        anchor_hidden_dim,
        rms_norm_eps,
        anchor_config,
    )

    # Bridge parameters are intentionally kept in FP32 for stable training.
    # CALM moves the resulting residual back to the frozen anchor dtype.
    self.proj.float()
    self.cross_attention.float()
    self.post_attention_layernorm.float()

    self.aug_hidden_state = None
    self.aug_mask = None
    self.attn_weights = None

  def set_state(self, hidden_state, attention_mask):
    # Clone references to the CURRENT DeepSeek pass only.
    self.aug_hidden_state = hidden_state
    self.aug_mask = attention_mask

  def clear_state(self):
    self.aug_hidden_state = None
    self.aug_mask = None
    self.attn_weights = None

  def forward(self, *hook_args):
    query, output = process_hook_args(*hook_args)

    if self.aug_hidden_state is None:
      raise RuntimeError(
          "CALM bridge has no DeepSeek hidden state. "
          "DeepSeek must run before the anchor."
      )

    if self.aug_mask is None:
      raise RuntimeError("CALM bridge has no DeepSeek attention mask.")

    # Expected:
    # query            [B, S_anchor, 1536]
    # aug_hidden_state [B, S_aug,    3584]
    if query.shape[0] != self.aug_hidden_state.shape[0]:
      raise RuntimeError(
          "CALM bridge batch mismatch: "
          f"anchor={tuple(query.shape)}, "
          f"deepseek={tuple(self.aug_hidden_state.shape)}"
      )

    # Keep the trainable bridge in FP32. This is important because the anchor
    # is FP16 while DeepSeek is 4-bit/FP16. We do not mutate bridge parameter
    # dtype inside forward (doing so would invalidate an optimizer's references).
    device = query.device

    query_fp32 = query.float()
    aug_hidden_fp32 = self.aug_hidden_state.to(
        device=device,
        dtype=torch.float32,
    )

    # One projection is intentionally shared for K/V, matching original CALM.
    key_value = self.proj.to(device=device)(aug_hidden_fp32)

    key_padding_mask = ~self.aug_mask.to(device=device).bool()

    attn_output, attn_weights = self.cross_attention.to(device=device)(
        query=query_fp32,
        key=key_value,
        value=key_value,
        key_padding_mask=key_padding_mask,
        need_weights=True,
    )

    self.attn_weights = attn_weights.detach()

    attn_output = self.post_attention_layernorm.to(device=device)(
        attn_output
    )

    # Return exactly the dtype expected by the frozen Qwen2 anchor block.
    output_fin = (query_fp32 + attn_output).to(query.dtype)

    return (output_fin,) + output[1:]


class ExtractHiddenStateHook(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.hidden_state = None

  def forward(self, *hook_args):
    hidden_state, out = process_hook_args(*hook_args)
    self.hidden_state = hidden_state
    return out