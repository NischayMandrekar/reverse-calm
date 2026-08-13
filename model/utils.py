# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Utilities for CALM layer connections."""

import numpy as np


def check_connections(connections, num_anchor_layers, num_aug_layers):
  for connection in connections:
    if len(connection) != 2:
      return False
    a, b = connection
    if not 0 <= a < num_anchor_layers:
      return False
    if not 0 <= b < num_aug_layers:
      return False
  return True


def get_connections(num_connections, num_anchor_layers, num_aug_layers):
  if num_connections < 1:
    raise ValueError("num_connections must be >= 1")

  if num_connections > min(num_anchor_layers, num_aug_layers):
    raise ValueError(
        f"num_connections={num_connections} exceeds the smaller model "
        f"depth: anchor={num_anchor_layers}, augmenting={num_aug_layers}"
    )

  anchor_layers = np.linspace(
      0, num_anchor_layers - 1, num_connections, dtype=int
  )
  aug_layers = np.linspace(
      0, num_aug_layers - 1, num_connections, dtype=int
  )

  return list(zip(anchor_layers.tolist(), aug_layers.tolist()))


def get_hidden_dims(anchor_model, aug_model, connection):
  anchor_layer, aug_layer = connection

  anchor_layer_module = anchor_model.model.layers[anchor_layer]
  aug_layer_module = aug_model.model.layers[aug_layer]

  anchor_hidden_dim = getattr(
      anchor_layer_module,
      "hidden_size",
      anchor_model.config.hidden_size,
  )
  aug_hidden_dim = getattr(
      aug_layer_module,
      "hidden_size",
      aug_model.config.hidden_size,
  )

  return int(anchor_hidden_dim), int(aug_hidden_dim)