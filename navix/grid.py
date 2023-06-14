# Copyright 2023 The Navix Authors.

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


from __future__ import annotations


from typing import Dict, Tuple
import jax
import jax.numpy as jnp
from jax.random import KeyArray
from jax.typing import ArrayLike
from chex import Array


def room(width: int, height: int):
    grid = jnp.zeros((width, height), dtype=jnp.int32)
    return jnp.pad(grid, 1, mode="constant", constant_values=-1)


def coordinates_to_idx(grid: Array, coordinates: ArrayLike):
    return coordinates[0] * grid.shape[0] + coordinates[1]


def idx_to_coordinates(grid: Array, idx: int):
    return jnp.stack(jnp.divmod(idx, grid.shape[0]))


def coordinate_to_mask(grid: Array, coordinates: ArrayLike) -> Array:
    raise NotImplementedError()


def entity_coordinates(grid: Array, entity_id: int) -> Tuple[int, int]:
    idx = mask_entity(grid, entity_id).reshape(
        -1,
    )
    idx = jnp.asarray(idx, dtype=jnp.int32)
    coordinates = tuple(idx_to_coordinates(grid, idx))
    return coordinates


def mask_entity(grid: Array, entity_id: int) -> Array:
    return jnp.asarray(grid == entity_id, dtype=jnp.float32)


def place_entity(grid: Array, entity_id: int, coordinates: ArrayLike) -> Array:
    return grid.at[coordinates].set(entity_id)


def spawn_entity(grid: Array, entity_id: int, key: KeyArray) -> Array:
    assert entity_id > 0, f"Reserved id {entity_id}, please specify an id > 0"
    mask = mask_entity(grid, 0)
    idx = jax.random.categorical(
        key,
        jnp.log(
            mask.reshape(
                -1,
            )
        ),
    )
    coordinates = tuple(idx_to_coordinates(grid, idx))
    grid = place_entity(grid, entity_id, coordinates)
    return grid


def remove_entity(grid: Array, entity_id: int, replacement: int = 0) -> Array:
    mask = mask_entity(grid, entity_id)
    return jnp.where(mask, replacement, grid)


def from_ascii(ascii_map: str, mapping: Dict[str, int] = {}) -> Array:
    mapping = {**{"#": -1, ".": 0}, **mapping}

    ascii_map = ascii_map.strip()
    max_width = max(len(line.strip()) for line in ascii_map.splitlines())
    grid = []
    for line in ascii_map.splitlines():
        line = line.strip()
        assert len(line) == max_width, "All lines must be the same length"
        row = [int(mapping.get(character, character)) for character in line]
        grid.append(row)

    return jnp.asarray(grid, dtype=jnp.int32)
