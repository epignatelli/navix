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
from jax import Array


Coordinates = Tuple[Array, Array]


def coordinates(grid: Array) -> Coordinates:
    return tuple(jnp.mgrid[0 : grid.shape[0], 0 : grid.shape[1]])  # type: ignore


def idx_from_coordinates(grid: Array, coordinates: Array):
    """Converts a batch of 2D coordinates [(col, row), ...] into a flat index"""
    coordinates = coordinates.T
    assert coordinates.shape[0] == 2, coordinates.shape

    idx = coordinates[0] * grid.shape[1] + coordinates[1]
    return jnp.asarray(idx, dtype=jnp.int32)


def coordinates_from_idx(grid: Array, idx: Array):
    """Converts a flat index into a 2D coordinate (col, row)"""
    coords = jnp.divmod(idx, grid.shape[1])
    return jnp.asarray(coords, dtype=jnp.int32).T


def mask_by_coordinates(
    grid: Array,
    address: Coordinates,
    comparison_fn: Callable[[Array, Array], Array] = jnp.greater_equal,
) -> Array:
    """This is a workaround to compute dynamicly-sized masks in XLA,
    which would not be possible otherwise.
    Returns a mask of the same shape as `grid` where the value is 1 if the
    corresponding element in `grid` satisfies the `comparison_fn` with the
    corresponding element in `address` (col, row) and 0 otherwise.
    """
    mesh = jnp.mgrid[0 : grid.shape[0], 0 : grid.shape[1]]
    cond_1 = comparison_fn(mesh[0], address[0])
    cond_2 = comparison_fn(mesh[1], address[1])
    mask = jnp.asarray(jnp.logical_and(cond_1, cond_2), dtype=jnp.int32)
    return mask


def translate(
    position: Array, direction: Array, modulus: Array = jnp.asarray(1)
) -> Array:
    moves = (
        lambda position: position + jnp.asarray((0, modulus)),  # east
        lambda position: position + jnp.asarray((modulus, 0)),  # south
        lambda position: position + jnp.asarray((0, -modulus)),  # west
        lambda position: position + jnp.asarray((-modulus, 0)),  # north
    )
    return jax.lax.switch(direction, moves, position)


def translate_forward(position: Array, forward_direction: Array, modulus: Array):
    return translate(position, forward_direction, modulus)


def translate_left(position: Array, forward_direction: Array, modulus: Array):
    return translate(position, (forward_direction + 3) % 4, modulus)


def translate_right(position: Array, forward_direction: Array, modulus: Array):
    return translate(position, (forward_direction + 1) % 4, modulus)


def rotate(direction: Array, spin: int) -> Array:
    return (direction + spin) % 4


def align(patch: Array, current_direction: Array, desired_direction: Array) -> Array:
    return jax.lax.switch(
        desired_direction - current_direction,
        (
            lambda x: jnp.rot90(x, 1),  # 0 = transpose, 1 = flip
            lambda x: jnp.rot90(x, 2),  # 0 = flip, 1 = flip
            lambda x: jnp.rot90(x, 3),  # 0 = flip, 1 = transpose
            lambda x: x,
        ),
        patch,
    )


def random_positions(
    key: Array, grid: Array, n: int = 1, exclude: Array = jnp.asarray((-1, -1))
) -> Array:
    probs = grid.reshape(-1)
    indices = idx_from_coordinates(grid, exclude)
    probs = probs.at[indices].set(-1) + 1.0
    idx = jax.random.categorical(key, jnp.log(probs), shape=(n,))
    position = coordinates_from_idx(grid, idx)
    return position.squeeze()


def random_directions(key: Array, n=1) -> Array:
    return jax.random.randint(key, (n,), 0, 4).squeeze()


def positions_equal(a: Array, b: Array) -> Array:
    if b.ndim == 1:
        b = b[None]
    if a.ndim == 1:
        a = a[None]
    assert a.ndim == b.ndim == 2, (a.shape, b.shape)
    is_equal = jnp.all(jnp.equal(a, b), axis=-1)
    assert is_equal.shape == (max(b.shape[0], a.shape[0]),)
    return is_equal


def room(height: int, width: int):
    """A grid of ids of size `width` x `height`"""
    grid = jnp.zeros((height - 2, width - 2), dtype=jnp.int32)
    return jnp.pad(grid, 1, mode="constant", constant_values=-1)


def two_rooms(height: int, width: int, key: Array) -> Tuple[Array, Array]:
    """Two rooms separated by a vertical wall at `width // 2`"""
    # create room
    grid = jnp.zeros((height - 2, width - 2), dtype=jnp.int32)
    grid = jnp.pad(grid, 1, mode="constant", constant_values=-1)

    # add separation wall
    wall_at = jax.random.randint(key, (), 2, width - 2)
    grid = grid.at[1:-1, wall_at].set(-1)
    return grid, wall_at


def crop(grid: Array, origin: Array, direction: Array, radius: int) -> Array:
    input_shape = grid.shape

    # pad with radius
    padding = [(radius, radius), (radius, radius)]
    for _ in range(len(input_shape) - 2):
        padding.append((0, 0))

    padded = jnp.pad(grid, padding, constant_values=0)

    # translate the grid such that the agent is `radius` away from the top and left edges
    translated = jnp.roll(padded, -jnp.asarray(origin), axis=(0, 1))

    # crop such that the agent is in the centre of the grid
    cropped = translated[: 2 * radius + 1, : 2 * radius + 1]

    # rotate such that the agent is facing north
    rotated = jax.lax.switch(
        direction,
        (
            lambda x: jnp.rot90(x, 1),  # 0 = transpose, 1 = flip
            lambda x: jnp.rot90(x, 2),  # 0 = flip, 1 = flip
            lambda x: jnp.rot90(x, 3),  # 0 = flip, 1 = transpose
            lambda x: x,
        ),
        cropped,
    )

    cropped = rotated[: radius + 1]
    return jnp.asarray(cropped, dtype=grid.dtype)


def view_cone(transparency_map: Array, origin: Array, radius: int) -> Array:
    # transparency_map is a boolean map of transparent (1) and opaque (0) tiles

    def fin_diff(array, _):
        array = jnp.roll(array, -1, axis=0) + array + jnp.roll(array, +1, axis=0)
        array = jnp.roll(array, -1, axis=1) + array + jnp.roll(array, +1, axis=1)
        return array * transparency_map, ()

    mask = jnp.zeros_like(transparency_map).at[tuple(origin)].set(1)

    view = jax.lax.scan(fin_diff, mask, None, radius)[0]

    # we now set a hard threshold > 0, but we can also think in the future
    # to use a cutoof at a different value to mimic the effect of a torch
    # (or eyesight for what matters)
    view = jnp.where(view > 0, 1, 0)

    # we add back the opaque tiles
    view = jnp.where(transparency_map == 0, 1, view)

    return view


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
