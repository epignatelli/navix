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


from typing import Callable, Dict, Tuple
import jax
import jax.numpy as jnp
from jax.random import KeyArray
from jax import Array


Coordinates = Tuple[Array, Array]


def mask_by_address(grid: Array, address: Coordinates, comparison_fn: Callable[[Array, Array], Array] = jnp.greater_equal) -> Array:
    mesh = jnp.mgrid[0:grid.shape[0], 0:grid.shape[1]]
    cond_1 = comparison_fn(mesh[0], address[0])
    cond_2 = comparison_fn(mesh[1], address[1])
    mask = jnp.asarray(jnp.logical_and(cond_1, cond_2), dtype=jnp.int32)
    return mask


def room(width: int, height: int):
    grid = jnp.zeros((height, width), dtype=jnp.int32)
    return jnp.pad(grid, 1, mode="constant", constant_values=-1)


def two_rooms(width: int, height: int, key: KeyArray) -> Array:
    """Two rooms separated by a vertical wall at `width // 2`"""
    grid = jnp.zeros((height - 2, width - 2), dtype=jnp.int32)

    wall_at = jax.random.randint(key, (), 2, width - 2)

    grid = jnp.pad(grid, 1, mode="constant", constant_values=-1)
    grid = grid.at[1:-1, wall_at].set(-1)
    return grid


def random_positions(key: KeyArray, grid: Array, n=1) -> Array:
    mask = jnp.where(grid, 0, 1)  # all floor tiles
    probs = jnp.log(mask).reshape((-1,))
    idx = jax.random.categorical(key, probs, shape=(n,))

    positions = jnp.stack(jnp.divmod(idx, grid.shape[1])).T
    return positions.squeeze()


def random_directions(key: KeyArray, n=1) -> Array:
    return jax.random.randint(key, (n,), 0, 4).squeeze()


def from_ascii_map(ascii_map: str, mapping: Dict[str, int] = {}) -> Array:
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
