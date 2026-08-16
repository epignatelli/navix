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

from .rendering.cache import TILE_SIZE, unflatten_patches
from .components import DISCARD_PILE_IDX, Directional, HasColour, Openable
from .states import State
from .grid import (
    apply_minigrid_opacity,
    align,
    idx_from_coordinates,
    crop,
    view_cone,
)
from .entities import EntityIds


RADIUS = 3


def set_radius(radius: int):
    global RADIUS
    RADIUS = radius


def none(state: State) -> Array:
    """An empty observation represented as an array of shape f32[0].
    Useful for testing purposes.

    Args:
        state (State): The current state of the game.

    Returns:
        Array: A 0-shaped array `f32[0]`."""
    return jnp.asarray(())


def categorical(state: State) -> Array:
    """Fully observable grid with a categorical state representation.
    Each entity is represented by its unique integer tag.
    
    Args:
        state (State): The current state of the game.
    
    Returns:
        Array: A grid of integers, where each integer represents an entity, \
        represented as an array of shape `i32[H, W]`, where `H` and `W` are the height \
        and width of the grid."""
    # get idx of entity on the set of patches
    indices = idx_from_coordinates(state.grid, state.get_positions())
    # get tags corresponding to the entities
    tags = state.get_tags()
    # set tags on the flat set of patches
    shape = state.grid.shape
    num_cells = shape[0] * shape[1]
    # a picked-up entity's position (DISCARD_PILE_COORDS) maps to flat
    # index -1 - .at[].set()'s default mode wraps negative indices
    # around (numpy semantics) rather than dropping them, silently
    # overwriting a real cell. Push negative indices to be explicitly
    # out of bounds first, so mode="drop" discards those writes instead.
    indices = jnp.where(indices < 0, num_cells, indices)
    grid = state.grid.reshape(-1).at[indices].set(tags, mode="drop")
    # unflatten patches to reconstruct the grid
    return grid.reshape(shape)


def categorical_first_person(state: State) -> Array:
    """Categorical state representation, but cropped to the agent's view, and aligned \
    with the agent's direction, such that the agent always points upwards.
    
    Args:
        state (State): The current state of the game.

    Returns:
        Array: A grid of integers, where each integer represents an entity, \
        represented as an array of shape `i32[2 * RADIUS + 1, 2 * RADIUS + 1]`."""
    # get transparency map
    transparency_map = jnp.where(state.grid == 0, 1, 0)
    positions = state.get_positions()
    transparent = state.get_transparency()
    # a picked-up entity's position (DISCARD_PILE_COORDS = (0, -1)) is
    # off-grid - .at[].set()'s default mode wraps negative components
    # around (numpy semantics) rather than dropping them, silently
    # overwriting a real cell (see categorical()/symbolic() for the
    # same bug, fixed the same way). Push off-grid positions to be
    # explicitly out of bounds first, so mode="drop" discards those
    # writes instead.
    H, W = state.grid.shape
    row, col = positions[..., 0], positions[..., 1]
    on_grid = (row >= 0) & (row < H) & (col >= 0) & (col < W)
    row = jnp.where(on_grid, row, H)
    col = jnp.where(on_grid, col, W)
    transparency_map = transparency_map.at[row, col].set(transparent, mode="drop")

    # apply view mask. crop() places the agent at the *bottom* row of
    # the 2*RADIUS+1 view, so the far row is 2*RADIUS cells forward of
    # the agent, not RADIUS - view_cone's diffusion needs to reach that
    # far, or the forward half of the view is permanently marked
    # unseen regardless of whether real walls are there.
    player = state.get_player()
    view = view_cone(transparency_map, player.position, RADIUS * 2)

    # get categorical representation
    tags = state.get_tags()
    obs = state.grid.at[row, col].set(tags, mode="drop") * view

    # crop grid to agent's view
    obs = crop(obs, player.position, player.direction, RADIUS)

    return obs


def symbolic(state: State) -> Array:
    """Fully observable grid with a symbolic state representation as originally \
    proposed in the MiniGrid environment.
    The symbol is a triple of (OBJECT_TAG, COLOUR_IDX, OPEN/CLOSED/LOCKED). The
    last layer might also contain the direction of the entity, for example, the
    direction of the agent.
    
    Args:
        state (State): The current state of the game.
        
    Returns:
        Array: A grid of integers, where each integer represents an entity, \
        represented as an array of shape `u8[H, W, 3]`, where `H` and `W` are the height \
        and width of the grid."""
    # initialise as all floors
    H, W = state.grid.shape
    obs = jnp.zeros((H, W, 3), dtype=jnp.uint8)
    wall_symbol = jnp.array([EntityIds.WALL, 5, 0], dtype=jnp.uint8)
    floor_symbol = jnp.array([EntityIds.FLOOR, 0, 0], dtype=jnp.uint8)
    obs = jnp.where(state.grid[..., None] == -1, wall_symbol, floor_symbol)

    # place entities
    for entity_class in state.entities:
        entity = state.entities[entity_class]
        # 1. tag layer
        tag = entity.tag
        # 2. colour layer
        if isinstance(entity, HasColour):
            colour = entity.colour
        else:
            colour = jnp.zeros(entity.shape)
        # 3. state layer
        entity_state = entity.symbolic_state

        # collate
        entity_symbol = jnp.stack([tag, colour, entity_state], axis=-1, dtype=jnp.uint8)
        # a picked-up entity's position (DISCARD_PILE_COORDS = (0, -1))
        # is off-grid - .at[].set()'s default mode wraps negative
        # components around (numpy semantics) rather than dropping
        # them, silently overwriting a real cell. Push off-grid
        # positions to be explicitly out of bounds first, so
        # mode="drop" discards those writes instead.
        row, col = entity.position[..., 0], entity.position[..., 1]
        on_grid = (row >= 0) & (row < H) & (col >= 0) & (col < W)
        row = jnp.where(on_grid, row, H)
        col = jnp.where(on_grid, col, W)
        obs = obs.at[row, col].set(entity_symbol, mode="drop")
    return obs


def symbolic_first_person(state: State) -> Array:
    """First person view with a symbolic state representation, but cropped to the \
    agent's view, and aligned with the agent's direction, such that the agent always \
    points upwards. See `symbolic` for more details.
    
    Args:
        state (State): The current state of the game.
    
    Returns:
        Array: A grid of integers, where each integer represents an entity, \
        represented as an array of shape `u8[2 * RADIUS + 1, 2 * RADIUS + 1, 3]`."""
    # get transparency map
    obs = symbolic(state)

    # replace player with pocket to show them what they are carrying
    player = state.get_player()
    obs = obs.at[tuple(player.position.T)].set(
        jnp.asarray([EntityIds.FLOOR, 0, 0], dtype=jnp.uint8)
    )

    # crop to first person view
    obs = crop(
        obs,
        player.position,
        player.direction,
        RADIUS,
        padding_value=255,
    )
    # replace padding symbol with walls
    wall_symbol = jnp.array([EntityIds.WALL, 5, 0], dtype=jnp.uint8)
    obs = jnp.where(obs == 255, wall_symbol, obs)
    return obs


def rgb(state: State) -> Array:
    """Fully observable grid with an RGB state representation.
    Each entity is represented by its unique RGB sprite. The RGB sprites are \
    stored in a cache, and the entities are placed on the grid according to their \
    positions.
    
    Args:
        state (State): The current state of the game.
    
    Returns:
        Array: An RGB image of the grid, represented as an array of shape \
        `u8[H * S, W * S, 3]`, where `H` and `W` are the height and width of the grid,
        and `S` is the size of the tile."""
    # get idx of entity on the flat set of patches
    indices = idx_from_coordinates(state.grid, state.get_positions())
    # get tiles corresponding to the entities
    tiles = state.get_sprites()
    # set tiles on the flat set of patches
    patches = state.cache.patches.at[indices].set(tiles)
    # remove discard pile
    patches = patches[:DISCARD_PILE_IDX]
    # unflatten patches to reconstruct the image
    image_size = (
        state.grid.shape[0] * TILE_SIZE,
        state.grid.shape[1] * TILE_SIZE,
    )
    image = unflatten_patches(patches, image_size)
    return image


def rgb_first_person(state: State) -> Array:
    """First person view with an RGB state representation.
    The image is cropped to the agent's view, and aligned with the agent's direction, \
    such that the agent always points upwards. See `rgb` for more details.
    See `rgb` for more details.

    Args:
        state (State): The current state of the game.
    
    Returns:
        Array: An RGB image of the agent's view, represented as an array of shape \
        `u8[(2 * RADIUS + 1) * S, (2 * RADIUS + 1) * S, 3]`, where 
        `S` is the size of the tile."""
    # get the player
    player = state.get_player()

    # get sprites aligned to player's direction
    sprites = state.get_sprites_first_person()  # (n_sprites, TILE_SIZE, TILE_SIZE, 3)
    # sprites = jax.vmap(lambda x: align(x, jnp.asarray(0), alignment_direction))(sprites)

    # Grid lines are drawn once on the floor tile in
    # rendering/cache.py's render_background(), not here - `sprites` is
    # per-entity (player, keys, balls, doors, ...), and MiniGrid's own
    # grid lines are drawn under objects, not over them (Wall.render()
    # fully covers its tile, so grid lines never show through walls
    # either). Drawing lines directly on these entity sprites would
    # incorrectly overlay a line across their artwork instead.

    # update current patchwork
    indices = idx_from_coordinates(state.grid, state.get_positions())
    patches = state.cache.patches.at[indices].set(
        sprites
    )  # ( H * W + 1, TILE_SIZE, TILE_SIZE, 3)

    # remove discard pile
    patches = patches[:DISCARD_PILE_IDX]  # ( H * W, TILE_SIZE, TILE_SIZE, 3)
    # rearrange the sprites in a grid
    patchwork = patches.reshape(
        *state.grid.shape, *patches.shape[1:]
    )  # (H, W, TILE_SIZE, TILE_SIZE, 3)

    # apply minigrid opacity
    patchwork = apply_minigrid_opacity(patchwork)

    # apply fov. Unseen/out-of-map tiles use the *opacity-adjusted*
    # wall grey, not the raw (100, 100, 100) constant: every other
    # cell in `patchwork` already went through apply_minigrid_opacity
    # above, but dark_cell_colour is inserted as a flat literal after
    # that, bypassing it - using the raw constant here made real,
    # visible walls (opacity-adjusted, ~146) visually inconsistent
    # with the unseen/padding fill (100) in the same image, a seam
    # that isn't in MiniGrid's own rendering. A scalar still works for
    # both the jnp.where fill below and crop()'s padding_value, since
    # grey has equal R/G/B and both broadcast it across the full
    # (..., 3) tile.
    dark_cell_colour = apply_minigrid_opacity(jnp.asarray(100, dtype=jnp.uint8))
    transparency_map = jnp.where(state.grid == 0, 1, 0)  # (H, W)
    positions = state.get_positions()
    transparent = state.get_transparency()
    transparency_map = transparency_map.at[tuple(positions.T)].set(transparent)
    # crop() places the agent at the *bottom* row of the 2*RADIUS+1
    # view, so the far row is 2*RADIUS cells forward of the agent, not
    # RADIUS - view_cone's diffusion needs to reach that far, or the
    # forward half of the view is permanently marked unseen regardless
    # of whether real walls are there.
    view = view_cone(transparency_map, player.position, RADIUS * 2)  # (H, W)
    view = jnp.asarray(view, dtype=jnp.bool)
    patchwork = jnp.where(view[..., None, None, None], patchwork, dark_cell_colour)

    # crop grid to agent's view
    patchwork = crop(
        patchwork, player.position, player.direction, RADIUS, dark_cell_colour
    )  # (RADIUS * 2 + 1, RADIUS * 2 + 1, TILE_SIZE, TILE_SIZE, 3)

    # reconstruct image
    obs = jnp.swapaxes(patchwork, 1, 2)
    shape = obs.shape
    obs = obs.reshape(shape[0] * shape[1], shape[2] * shape[3], *shape[4:])
    return obs
