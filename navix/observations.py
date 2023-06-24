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
from jax import Array

from . import graphics
from .components import State
from .grid import idx_from_coordinates


# def _view_extremes(state: State, radius: Array) -> Tuple[Array, Array]:
#     pos_left = translate_left(state.player.position, state.player.direction, radius)
#     pos_forward_left = translate_forward(pos_left, state.player.direction, radius)
#     pos_right = translate_right(state.player.position, state.player.direction, radius)
#     pos_forward_right = translate_forward(pos_right, state.player.direction, radius)

#     all_pos = jnp.stack([pos_left, pos_forward_left, pos_right, pos_forward_right], axis=0)

#     north_west = jnp.min(all_pos, axis=0)
#     south_east = jnp.max(all_pos, axis=0)

#     return north_west, south_east


# def _view_mask(state: State, radius: Array) -> Array:
#     north_west, south_east = _view_extremes(state, radius)
#     mask = state.grid.at[north_west[0]:south_east[0] + 1, north_west[1]:south_east[1] + 1].set(1)
#     return mask


# def _first_person_crop(state: State, radius: Array) -> Array:
#     north_west, south_east = _view_extremes(state, radius)
#     view = state.grid[north_west[0]:south_east[0] + 1, north_west[1]:south_east[1] + 1]
#     view = jnp.rot90(view, k=state.player.direction + 1)
#     return view


def none(
    state: State,
    # cache: graphics.RenderingCache,
    tiles_registry: Dict[str, Array] = graphics.TILES_REGISTRY,
) -> Array:
    return jnp.asarray(())


def categorical(
    state: State,
    # cache: graphics.RenderingCache,
    tiles_registry: Dict[str, Array] = graphics.TILES_REGISTRY,
) -> Array:
    # get idx of entity on the set of patches
    indices = idx_from_coordinates(state.grid, state.get_positions(axis=0))
    # get tags corresponding to the entities
    tags = state.get_tags(axis=0)
    # set tags on the flat set of patches
    shape = state.grid.shape
    grid = state.grid.reshape(-1).at[indices].set(tags)
    # unflatten patches to reconstruct the grid
    return grid.reshape(shape)


def rgb(
    state: State,
    # cache: graphics.RenderingCache,
    tiles_registry: Dict[str, Array] = graphics.TILES_REGISTRY,
) -> Array:
    # TODO(epignatelli): we can simplify this by indexing from tuples directly
    # e.g., background[tuple(positions.T)]
    # this allows to remove:
    # the idx_from_coordinates function, the
    # graphics.unflatten_patches function (replace it with a simpler reshape), and the
    # graphics.flatten_patches function from `env.reset`

    # get idx of entity on the flat set of patches
    indices = idx_from_coordinates(state.grid, state.get_positions(axis=0))
    # get tiles corresponding to the entities
    tiles = state.get_tiles(tiles_registry, axis=0)
    # set tiles on the flat set of patches
    patches = state.cache.patches.at[indices].set(tiles)
    # unflatten patches to reconstruct the image
    image_size = (state.grid.shape[0] * graphics.TILE_SIZE, state.grid.shape[1] * graphics.TILE_SIZE)
    image = graphics.unflatten_patches(patches, image_size)
    return image
