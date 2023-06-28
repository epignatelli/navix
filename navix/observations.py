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
import jax

import jax.numpy as jnp
from jax import Array

from . import graphics
from .components import DISCARD_PILE_IDX
from .entities import State
from .grid import idx_from_coordinates, crop, view_cone


def none(
    state: State,
    sprites_registry: Array = graphics.SPRITES_REGISTRY,
) -> Array:
    return jnp.asarray(())


def categorical(
    state: State,
    sprites_registry: Array = graphics.SPRITES_REGISTRY,
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


def categorical_first_person(
    state: State,
    sprites_registry: Array = graphics.SPRITES_REGISTRY,
    radius: int = 3,
) -> Array:
    # get transparency map
    transparency_map = jnp.where(state.grid == 0, 1, 0)
    positions = state.get_positions(axis=0)
    transparent = state.get_transparents(axis=0)
    transparency_map = transparency_map.at[tuple(positions.T)].set(~transparent)

    # apply view mask
    view = view_cone(transparency_map, state.players.position, radius)

    # get categorical representation
    tags = state.get_tags(axis=0)
    obs = state.grid.at[tuple(positions.T)].set(tags) * view

    # crop grid to agent's view
    obs = crop(obs, state.players.position, state.players.direction, radius)

    return obs


def rgb(
    state: State,
    sprites_registry: Array = graphics.SPRITES_REGISTRY,
) -> Array:
    # for 1-d vs 2-d indexing benchamarks
    # see https://github.com/epignatelli/navix/tree/observation/2dindexing

    # get idx of entity on the flat set of patches
    indices = idx_from_coordinates(state.grid, state.get_positions(axis=0))
    # get tiles corresponding to the entities
    tiles = state.get_sprites(sprites_registry, axis=0)
    # set tiles on the flat set of patches
    patches = state.cache.patches.at[indices].set(tiles)
    # remove discard pile
    patches = patches[:DISCARD_PILE_IDX]
    # unflatten patches to reconstruct the image
    image_size = (
        state.grid.shape[0] * graphics.TILE_SIZE,
        state.grid.shape[1] * graphics.TILE_SIZE,
    )
    image = graphics.unflatten_patches(patches, image_size)
    return image
