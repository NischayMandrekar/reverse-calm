# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""CALM composition for a Konkani Qwen2 anchor + DeepSeek augmenting model.

Project setup:
  anchor_model = nischay185/konkani-qwen2-1.5b
  aug_model    = deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

The anchor receives native Konkani. The augmenting model receives the
translated input. The augmenting model only supplies intermediate hidden
states; the anchor remains the final language model/output head.

Base models are frozen. Only CALM bridge modules are trainable.
"""

import os
from typing import Callable, Optional, Union

import torch
import transformers

from model import layers
from model import utils


class CALMConfig(transformers.PretrainedConfig):
  model_type = "calm"

  def __init__(
      self,
      anchor_model: str = "nischay185/konkani-qwen2-1.5b",
      aug_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
      anchor_config=None,
      aug_config=None,
      connections=None,
      num_connections=None,
      num_heads=1,
      **kwargs,
  ):
    super().__init__(**kwargs)

    if (connections is None) == (num_connections is None):
      raise ValueError(
          "Provide exactly one of `connections` or `num_connections`."
      )

    if num_heads < 1:
      raise ValueError("num_heads must be >= 1.")

    self.anchor_model = anchor_model
    self.aug_model = aug_model
    self.anchor_config = anchor_config
    self.aug_config = aug_config
    self.connections = connections
    self.num_connections = num_connections
    self.num_heads = num_heads


class CALM(transformers.PreTrainedModel, transformers.GenerationMixin):
  config_class = CALMConfig
  base_model_prefix = "anchor_model"

  @property
  def lm_head(self):
    return self.anchor_model.lm_head

  def __init__(
      self,
      config: CALMConfig,
      anchor_model_instance=None,
      aug_model_instance=None,
  ):
    super().__init__(config)

    # IMPORTANT:
    # The notebook already loads these models, including the 4-bit DeepSeek
    # quantization. Reuse those instances instead of loading them again.
    if anchor_model_instance is not None:
      self.anchor_model = anchor_model_instance
    else:
      if config.anchor_config is None:
        config.anchor_config = transformers.AutoConfig.from_pretrained(
            config.anchor_model
        )
      self.anchor_model = transformers.AutoModelForCausalLM.from_pretrained(
          config.anchor_model,
          config=config.anchor_config,
          torch_dtype=torch.float16,
      )

    if aug_model_instance is not None:
      self.aug_model = aug_model_instance
    else:
      if config.aug_config is None:
        config.aug_config = transformers.AutoConfig.from_pretrained(
            config.aug_model
        )
      self.aug_model = transformers.AutoModelForCausalLM.from_pretrained(
          config.aug_model,
          config=config.aug_config,
      )

    self.vocab_size = self.anchor_model.config.vocab_size

    self.num_anchor_layers = len(self.anchor_model.model.layers)
    self.num_aug_layers = len(self.aug_model.model.layers)

    if config.connections is not None:
      self.connections = [tuple(x) for x in config.connections]
      if not utils.check_connections(
          self.connections,
          self.num_anchor_layers,
          self.num_aug_layers,
      ):
        raise ValueError("Invalid CALM layer connections.")
      self.num_connections = len(self.connections)
    else:
      self.num_connections = config.num_connections
      self.connections = utils.get_connections(
          self.num_connections,
          self.num_anchor_layers,
          self.num_aug_layers,
      )

    # Capture hidden states from DeepSeek.
    self.extract_hidden_state_hooks = {}
    for connection in self.connections:
      aug_layer_idx = connection[1]
      hook = layers.ExtractHiddenStateHook()
      self.extract_hidden_state_hooks[connection] = hook
      self.aug_model.model.layers[aug_layer_idx].register_forward_hook(hook)

    # Build trainable bridges.
    self.cross_attention_hooks = torch.nn.ModuleList()
    for connection in self.connections:
      anchor_dim, aug_dim = utils.get_hidden_dims(
          self.anchor_model,
          self.aug_model,
          connection,
      )
      bridge = layers.CrossAttentionHook(
          anchor_hidden_dim=anchor_dim,
          aug_hidden_dim=aug_dim,
          num_heads=config.num_heads,
          rms_norm_eps=getattr(
              self.anchor_model.config, "rms_norm_eps", 1e-6
          ),
          anchor_config=self.anchor_model.config,
      )

      # Put trainable bridge parameters on the anchor GPU in FP32.
      anchor_param = next(self.anchor_model.parameters())
      bridge.to(device=anchor_param.device, dtype=torch.float32)
      self.cross_attention_hooks.append(bridge)

    # Freeze the two pretrained models. The bridge stays trainable.
    layers.freeze_model(self.anchor_model)
    layers.freeze_model(self.aug_model)

    # Inject each bridge into its corresponding anchor layer.
    for i, connection in enumerate(self.connections):
      anchor_layer_idx = connection[0]
      self.anchor_model.model.layers[anchor_layer_idx].register_forward_hook(
          self.cross_attention_hooks[i]
      )

  def train(self, mode=True):
    # Keep both frozen pretrained models in eval mode. Trainer.train() must
    # not accidentally turn dropout on inside them.
    super().train(mode)
    self.anchor_model.eval()
    self.aug_model.eval()
    self.cross_attention_hooks.train(mode)
    return self

  def release_memory(self):
    for hook in self.cross_attention_hooks:
      hook.clear_state()
    for hook in self.extract_hidden_state_hooks.values():
      hook.hidden_state = None

  def _forward_aug(
      self,
      aug_input_ids=None,
      aug_attention_mask=None,
      aug_position_ids=None,
      aug_inputs_embeds=None,
  ):
    if aug_input_ids is None and aug_inputs_embeds is None:
      raise ValueError(
          "The augmenting model requires `aug_input_ids` (translated input) "
          "or `aug_inputs_embeds`."
      )

    # Clear all old states BEFORE DeepSeek runs. This prevents stale-batch
    # hidden states from ever reaching the anchor.
    self.release_memory()

    if aug_attention_mask is None:
      if aug_input_ids is not None:
        aug_attention_mask = torch.ones_like(aug_input_ids)
      else:
        raise ValueError("aug_attention_mask is required with aug_inputs_embeds.")

    if aug_position_ids is None:
      aug_position_ids = aug_attention_mask.long().cumsum(-1) - 1
      aug_position_ids.masked_fill_(aug_attention_mask == 0, 1)

    with torch.no_grad():
      self.aug_model.eval()
      self.aug_model(
          input_ids=aug_input_ids,
          attention_mask=aug_attention_mask,
          position_ids=aug_position_ids,
          inputs_embeds=aug_inputs_embeds,
          use_cache=False,
          output_attentions=False,
          output_hidden_states=False,
          return_dict=True,
      )

    # The hooks have just been populated by THIS DeepSeek forward.
    for i, connection in enumerate(self.connections):
      hidden = self.extract_hidden_state_hooks[connection].hidden_state

      if hidden is None:
        raise RuntimeError(
            f"DeepSeek hidden state was not captured at layer "
            f"{connection[1]} for connection {connection}."
        )

      if hidden.shape[0] != aug_attention_mask.shape[0]:
        raise RuntimeError(
            "Augmenting batch mismatch: "
            f"hidden={tuple(hidden.shape)}, "
            f"aug_mask={tuple(aug_attention_mask.shape)}"
        )

      self.cross_attention_hooks[i].set_state(
          hidden,
          aug_attention_mask,
      )

  def forward(
      self,
      input_ids=None,
      attention_mask=None,
      position_ids=None,
      past_key_values=None,
      inputs_embeds=None,
      labels=None,
      use_cache=False,
      output_attentions=None,
      output_hidden_states=None,
      return_dict=True,
      cache_position=None,
      aug_input_ids=None,
      aug_attention_mask=None,
      aug_position_ids=None,
      aug_inputs_embeds=None,
      **kwargs,
  ):
    if input_ids is None and inputs_embeds is None:
      raise ValueError("Anchor requires input_ids or inputs_embeds.")

    if attention_mask is None and input_ids is not None:
      attention_mask = torch.ones_like(input_ids)

    # DeepSeek translated stream FIRST.
    self._forward_aug(
        aug_input_ids=aug_input_ids,
        aug_attention_mask=aug_attention_mask,
        aug_position_ids=aug_position_ids,
        aug_inputs_embeds=aug_inputs_embeds,
    )

    try:
      # Konkani anchor SECOND. Its intermediate layers consume the bridges.
      return self.anchor_model(
          input_ids=input_ids,
          attention_mask=attention_mask,
          position_ids=position_ids,
          past_key_values=past_key_values,
          inputs_embeds=inputs_embeds,
          labels=labels,
          use_cache=use_cache,
          output_attentions=output_attentions,
          output_hidden_states=output_hidden_states,
          return_dict=return_dict,
          cache_position=cache_position,
          **kwargs,
      )
    finally:
      self.release_memory()

  def prepare_inputs_for_generation(
      self,
      input_ids,
      past_key_values=None,
      attention_mask=None,
      inputs_embeds=None,
      cache_position=None,
      use_cache=True,
      **kwargs,
  ):
    # This is the critical generation piece: keep the translated DeepSeek
    # sequence available on EVERY decoding step.
    past_length = 0

    if past_key_values is not None:
      if isinstance(past_key_values, transformers.Cache):
        past_length = (
            cache_position[0]
            if cache_position is not None
            else past_key_values.get_seq_length()
        )
      else:
        past_length = past_key_values[0][0].shape[2]

      if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
        input_ids = input_ids[:, -(attention_mask.shape[1] - past_length):]
      elif past_length < input_ids.shape[1]:
        input_ids = input_ids[:, past_length:]

    if attention_mask is not None:
      position_ids = attention_mask.long().cumsum(-1) - 1
      position_ids.masked_fill_(attention_mask == 0, 1)
      position_ids = position_ids[:, -input_ids.shape[1]:]
    else:
      position_ids = None

    model_inputs = {
        "input_ids": input_ids.contiguous(),
        "position_ids": position_ids,
        "past_key_values": past_key_values,
        "use_cache": use_cache,
        "attention_mask": attention_mask,
    }

    # Preserve the full translated input. Do NOT truncate it with the anchor
    # decoding cache; DeepSeek is the fixed context provider.
    model_inputs["aug_input_ids"] = kwargs.get("aug_input_ids")
    model_inputs["aug_attention_mask"] = kwargs.get("aug_attention_mask")
    model_inputs["aug_position_ids"] = kwargs.get("aug_position_ids")

    if cache_position is None:
      start = past_length
      cache_position = torch.arange(
          start,
          start + input_ids.shape[1],
          device=input_ids.device,
      )
    model_inputs["cache_position"] = cache_position[-input_ids.shape[1]:]

    return model_inputs

  def save_pretrained(self, save_directory, **kwargs):
    # Save the CALM configuration + bridge state. The two pretrained base
    # models are intentionally not duplicated into the CALM checkpoint.
    os.makedirs(save_directory, exist_ok=True)
    bridge_state = self.cross_attention_hooks.state_dict()
    torch.save(
        bridge_state,
        os.path.join(save_directory, "calm_bridge.pt"),
    )
    self.config.save_pretrained(save_directory)