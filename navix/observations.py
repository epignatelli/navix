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

from jax import Array
import jax
import jax.numpy as jnp

from .components import State
from .graphics import (
    triangle_east_tile,
    diamond_tile,
    door_tile,
    key_tile,
    floor_tile,
    wall_tile,
    mosaic,
    TILE_SIZE
)


def third_person_view(state: State, radius: int) -> Array:
    raise NotImplementedError()


def first_person_view(state: State, radius: int) -> Array:
    raise NotImplementedError()


def categorical(state: State) -> Array:
    # updates are in reverse order of display
    # place keys (keys are represented with opposite sign of the door they open)
    grid = jnp.max(
        jax.vmap(lambda key: state.grid.at[tuple(key.position)].set(-key.tag))(
            state.keys
        ),
        axis=0,
    )
    # place doors
    grid = jnp.max(
        jax.vmap(lambda door: grid.at[tuple(door.position)].set(door.tag))(
            state.doors
        ),
        axis=0,
    )
    # place goals
    grid = jnp.max(
        jax.vmap(lambda goal: grid.at[tuple(goal.position)].set(goal.tag))(state.goals), axis=0
    )
    # place player last, always on top
    grid = grid.at[tuple(state.player.position)].set(state.player.tag)
    return grid


def rgb(state: State) -> Array:
    positions = jnp.stack([
        *state.keys.position,
        *state.doors.position,
        *state.goals.position,
        state.player.position,
    ])

    tiles = jnp.stack([
        *([key_tile()] * len(state.keys.position)),
        *([door_tile()] * len(state.doors.position)),
        *([diamond_tile()] * len(state.goals.position)),
        triangle_east_tile(),
    ])

    def draw(carry, x):
        image = carry
        mask, tile = x
        mask = jax.image.resize(mask,(mask.shape[0] * TILE_SIZE, mask.shape[1] * TILE_SIZE), method='nearest')
        mask = jnp.stack([mask] * tile.shape[-1], axis=-1)
        tiled = mosaic(state.grid, tile)
        image = jnp.where(mask, tiled, image)
        return (image), ()

    def body_fun(carry, x):
        position, tile = x
        mask = jnp.zeros_like(state.grid).at[tuple(position)].set(1)
        return draw(carry, (mask, tile))

    background = jnp.zeros((state.grid.shape[0] * TILE_SIZE, state.grid.shape[1] * TILE_SIZE, 3), dtype=jnp.uint8)

    # add floor
    floor_mask = jnp.where(state.grid == 0, 1, 0)
    floor = floor_tile()
    background, _ = draw(background, (floor_mask, floor))

    # add walls
    wall_mask = jnp.where(state.grid == -1, 1, 0)
    wall = wall_tile()
    background, _ = draw(background, (wall_mask, wall))

    # add entities
    image, _ = jax.lax.scan(body_fun, background, (positions, tiles))
    return image  # type: ignore