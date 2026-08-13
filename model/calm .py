import os
from typing import Callable, List, Optional, Tuple, Union
from model import layers
from model import utils
import torch
import transformers

class CALMConfig(transformers.PretrainedConfig):
  model_type = "calm"
  def __init__(
      self,
      anchor_model: str = "google/gemma-2b",
      aug_model: str = "google/gemma-2b",
      anchor_config: Optional[transformers.AutoConfig] = None,
      aug_config: Optional[transformers.AutoConfig] = None,
      connections: list[Tuple[int, int]] = None,
      num_connections: int = None,
      num_heads: int = 1,
      **kwargs,
  ):
    self.anchor_model = anchor_model
    self.aug_model = aug_model
    self.connections = connections
    self.num_connections = num_connections
    self.num_heads = num_heads
    self.anchor_config = anchor_config
    self.aug_config = aug_config
    super().__init__(**kwargs)

class CALM(transformers.PreTrainedModel):
  config_class = CALMConfig

  @property
  def lm_head(self):
    return self.anchor_model.lm_head

  def __init__(self, config: CALMConfig, anchor_model_instance=None, aug_model_instance=None):
    super().__init__(config)
    if config.anchor_config is None:
      config.anchor_config = transformers.AutoConfig.from_pretrained(config.anchor_model)
    if config.aug_config is None:
      config.aug_config = transformers.AutoConfig.from_pretrained(config.aug_model)

    if anchor_model_instance is not None:
      self.anchor_model = anchor_model_instance
    else:
      self.anchor_model = transformers.AutoModelForCausalLM.from_pretrained(config.anchor_model, config=config.anchor_config)
    
    if aug_model_instance is not None:
      self.aug_model = aug_model_instance
    else:
      self.aug_model = transformers.AutoModelForCausalLM.from_pretrained(config.aug_model, config=config.aug_config)

    self.vocab_size = self.anchor_model.config.vocab_size
    self.config = config
    self.num_anchor_layers = len(self.anchor_model.model.layers)
    self.num_aug_layers = len(self.aug_model.model.layers)

    assert (config.connections is None) ^ (config.num_connections is None)

    if config.connections is not None:
      self.connections = config.connections
      self.num_connections = len(config.connections)
    else:
      self.num_connections = config.num_connections
      self.connections = utils.get_connections(config.num_connections, self.num_anchor_layers, self.num_aug_layers)

    self.extract_hidden_state_hooks = {}
    for connection in self.connections:
      aug_connection_idx = connection[1]
      hook = layers.ExtractHiddenStateHook()
      self.extract_hidden_state_hooks[tuple(connection)] = hook
      self.aug_model.model.layers[aug_connection_idx].register_forward_hook(hook)

    self.connection_hidden_dims = []
    for connection in self.connections:
      anchor_hidden_dim, aug_hidden_dim = utils.get_hidden_dims(self.anchor_model, self.aug_model, tuple(connection))
      self.connection_hidden_dims.append((anchor_hidden_dim, aug_hidden_dim))

    self.cross_attention_hooks = torch.nn.ModuleList([])
    for _, connection_hidden_dim in zip(self.connections, self.connection_hidden_dims):
      self.cross_attention_hooks.append(
    layers.CrossAttentionHook(
        anchor_hidden_dim=connection_hidden_dim[0],
        aug_hidden_dim=connection_hidden_dim[1],
        num_heads=config.num_heads,
        rms_norm_eps=self.anchor_model.config.rms_norm_eps,
    )
)

    layers.freeze_model(self.anchor_model)
    layers.freeze_model(self.aug_model)

    for connection_idx, connection in enumerate(self.connections):
      connection_anchor_layer_idx = connection[0]
      layer = self.anchor_model.model.layers[connection_anchor_layer_idx]
      layer.register_forward_hook(self.cross_attention_hooks[connection_idx])

  def release_memory(self):
    for cross_attention_hook in self.cross_attention_hooks:
      cross_attention_hook.aug_hidden_state = None
      cross_attention_hook.aug_mask = None
      cross_attention_hook.attn_weights = None
    for extract_hidden_state_hook in self.extract_hidden_state_hooks.values():
      extract_hidden_state_hook.hidden_state = None

  def _forward_aug(self, input_ids=None, attention_mask=None, aug_input_ids=None, aug_attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None, labels=None, use_cache=True, output_attentions=None, output_hidden_states=None, return_dict=None, cache_position=None):
    with torch.no_grad():
      self.aug_model.eval()
      _aug_input_ids = aug_input_ids if aug_input_ids is not None else input_ids
      _aug_attention_mask = aug_attention_mask if aug_attention_mask is not None else attention_mask
      _position_ids = position_ids
      if aug_input_ids is not None and aug_input_ids.shape[1] != input_ids.shape[1]:
          _position_ids = _aug_attention_mask.long().cumsum(-1) - 1
          _position_ids.masked_fill_(_aug_attention_mask == 0, 1)

      output = self.aug_model(
          input_ids=_aug_input_ids,
          attention_mask=_aug_attention_mask,
          position_ids=_position_ids,
          use_cache=False,
      )
      for connection_idx, connection in enumerate(self.connections):
        aug_hidden_state = self.extract_hidden_state_hooks[tuple(connection)].hidden_state
        self.cross_attention_hooks[connection_idx].aug_hidden_state = aug_hidden_state
        self.cross_attention_hooks[connection_idx].aug_mask = _aug_attention_mask
        del aug_hidden_state
    return output

  def forward(self, input_ids=None, attention_mask=None, aug_input_ids=None, aug_attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None, labels=None, use_cache=True, output_attentions=None, output_hidden_states=None, return_dict=None, cache_position=None):
    aug_output = self._forward_aug(
        input_ids=input_ids,
        attention_mask=attention_mask,
        aug_input_ids=aug_input_ids,
        aug_attention_mask=aug_attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        labels=labels,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )
    del aug_output

    output = self.anchor_model(
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
    )
    return output

  def save_pretrained(self, save_directory, **kwargs):
    super().save_pretrained(save_directory, safe_serialization=False, **kwargs)

  def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, cache_position=None, use_cache=True, **kwargs):
    past_length = 0
    if past_key_values is not None:
      if isinstance(past_key_values, transformers.Cache):
        past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
      else:
        past_length = past_key_values[0][0].shape[2]
      if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
        input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
      elif past_length < input_ids.shape[1]:
        input_ids = input_ids[:, past_length:]

    position_ids = kwargs.get("position_ids", None)
    if attention_mask is not None and position_ids is None:
      position_ids = attention_mask.long().cumsum(-1) - 1
      position_ids.masked_fill_(attention_mask == 0, 1)

    model_inputs = {"input_ids": input_ids.contiguous()}
    input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
    if cache_position is None:
      cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device)
    elif use_cache:
      cache_position = cache_position[-input_length:]

    model_inputs.update({
        "position_ids": position_ids,
        "cache_position": cache_position,
        "past_key_values": past_key_values,
        "use_cache": use_cache,
        "attention_mask": attention_mask,
        "aug_input_ids": kwargs.get("aug_input_ids", None),
        "aug_attention_mask": kwargs.get("aug_attention_mask", None),
    })
    return model_inputs