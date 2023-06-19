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


def idx_from_coordinates(grid: Array, coordinates: Array):
    """Converts a 2D coordinate (col, row) into a flat index"""
    idx = coordinates[0] * grid.shape[1] + coordinates[1]
    return jnp.asarray(idx, dtype=jnp.int32)


def coordinates_from_idx(grid: Array, idx: Array):
    """Converts a flat index into a 2D coordinate (col, row)"""
    col, row = jnp.divmod(idx, grid.shape[1])
    coords = jnp.stack([col, row])
    return jnp.asarray(coords, dtype=jnp.int32)


def mask_by_coordinates(
    grid: Array,
    address: Coordinates,
    comparison_fn: Callable[[Array, Array], Array] = jnp.greater_equal,
) -> Array:
    """Returns a mask of the same shape as `grid` where the value is 1 if the
    corresponding element in `grid` satisfies the `comparison_fn` with the
    corresponding element in `address` (col, row) and 0 otherwise."""
    mesh = jnp.mgrid[0 : grid.shape[0], 0 : grid.shape[1]]
    cond_1 = comparison_fn(mesh[0], address[0])
    cond_2 = comparison_fn(mesh[1], address[1])
    mask = jnp.asarray(jnp.logical_and(cond_1, cond_2), dtype=jnp.int32)
    return mask


def room(height: int, width: int):
    """A grid of ids of size `width` x `height`"""
    grid = jnp.zeros((height, width), dtype=jnp.int32)
    return jnp.pad(grid, 1, mode="constant", constant_values=-1)


def two_rooms(height: int, width: int, key: KeyArray) -> Tuple[Array, Array]:
    """Two rooms separated by a vertical wall at `width // 2`"""
    # create room
    grid = jnp.zeros((height - 2, width - 2), dtype=jnp.int32)
    grid = jnp.pad(grid, 1, mode="constant", constant_values=-1)

    # add separation wall
    wall_at = jax.random.randint(key, (), 2, width - 2)
    grid = grid.at[1:-1, wall_at].set(-1)
    return grid, wall_at


def random_positions(
    key: KeyArray, grid: Array, n=1, exclude: Array = jnp.asarray([(-1, -1)])
) -> Array:
    # all floor tiles
    # TODO(epignatelli): switch to all walkable tiles
    mask = jnp.where(grid, 0, 1).reshape((-1,))  # all floor tiles

    # temporarily set excluded positions to 0 probability
    mask = jnp.min(
        jax.vmap(
            lambda position: mask.at[idx_from_coordinates(grid, position)].set(0)
        )(exclude),
        axis=0,
    )

    log_probs = jnp.log(mask)
    idx = jax.random.categorical(key, log_probs, shape=(n,))

    positions = jax.vmap(coordinates_from_idx, in_axes=(None, 0))(grid, idx)
    return positions.T.squeeze()


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
