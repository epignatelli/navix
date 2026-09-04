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


"""Observation functions: how a `State` is turned into what the agent sees.

Pick one and pass it as `observation_fn` to `navix.make` / `Environment.create`.
Two families:

- **Fully observable** (`categorical`, `symbolic`, `rgb`) - the whole
  `height x width` grid, always the same orientation.
- **First person / POMDP** (`categorical_first_person`,
  `symbolic_first_person`, `rgb_first_person`) - cropped to a
  `(2 * RADIUS + 1)` square with the player at the bottom-centre facing
  *up*, so the observation is egocentric and rotation-invariant. Cells
  the player cannot see (behind a wall, outside the view cone) are masked
  to a "not seen" fill.

Three encodings, shared by both families:

- **categorical** - one integer per cell, the entity's tag (see
  `entities.EntityIds`); shape `(H, W)`.
- **symbolic** - three integers per cell `(tag, colour, state)` as in
  MiniGrid; shape `(H, W, 3)`, `uint8`.
- **rgb** - a rendered image, `uint8`, each cell a `TILE_SIZE x TILE_SIZE`
  sprite; shape `(H * TILE_SIZE, W * TILE_SIZE, 3)`.

`Environment` infers the matching `observation_space` for these built-in
functions; a custom `observation_fn` needs `observation_space` passed
explicitly.
"""

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
"""Half-size of the first-person view: those observations are
`(2 * RADIUS + 1)` cells on a side (default `3` -> a 7x7 window). Change
it with `set_radius` *before* building an environment - `Environment`
reads it when it computes `observation_space`."""


def set_radius(radius: int):
    """Sets the module-global `RADIUS` used by every `*_first_person`
    observation. Call it before `navix.make` so the environment's
    `observation_space` picks up the new size.

    Args:
        radius (int): the new half-window size; the view becomes
            `(2 * radius + 1)` cells square."""
    global RADIUS
    RADIUS = radius


def none(state: State) -> Array:
    """The empty observation - shape `f32[0]`. Use it when the agent
    should learn from `state`/`reward` directly (e.g. debugging, or a
    hand-coded policy) and never looks at `observation`.

    Args:
        state (State): the current state (ignored).

    Returns:
        Array: an empty `f32[0]` array."""
    return jnp.asarray(())


def categorical(state: State) -> Array:
    """The whole grid as one integer per cell: the tag of whatever entity
    occupies it (`0` for empty floor, `-1`-marked walls become their tag
    via `entities.EntityIds`), fully observable.

    Args:
        state (State): the current state.

    Returns:
        Array: `i32[H, W]` (`H = env.height`, `W = env.width`). Entities
        that have been picked up (off-grid) do not appear."""
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
    """The egocentric version of `categorical`: one tag per cell, cropped
    to a `(2 * RADIUS + 1)` square around the player and rotated so the
    player sits at the bottom-centre facing up. Cells outside the view
    cone or occluded by a wall are set to `0` (not seen).

    Args:
        state (State): the current state.

    Returns:
        Array: `i32[2 * RADIUS + 1, 2 * RADIUS + 1]`."""
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
    """MiniGrid's symbolic encoding: three integers per cell,
    `(object_tag, colour_index, state)`, fully observable. `object_tag` is
    the entity id (empty floor and walls have their own tags);
    `colour_index` indexes the palette (`0` when the entity has no
    colour); the third channel is the entity's own discrete state - a
    door's open/closed/locked, or the player's facing direction.

    Args:
        state (State): the current state.

    Returns:
        Array: `u8[H, W, 3]` (`H = env.height`, `W = env.width`)."""
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
    """The egocentric version of `symbolic`: the `(tag, colour, state)`
    triple per cell, cropped to a `(2 * RADIUS + 1)` square around the
    player and rotated so the player faces up. Out-of-view / occluded
    cells are filled with the wall symbol; the player's own cell shows
    what it is carrying.

    Args:
        state (State): the current state.

    Returns:
        Array: `u8[2 * RADIUS + 1, 2 * RADIUS + 1, 3]`."""
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
    """The whole grid rendered as an RGB image, fully observable. Each
    cell is a `TILE_SIZE x TILE_SIZE` sprite (walls, floor grid lines,
    entities) drawn from `state.cache`.

    Args:
        state (State): the current state.

    Returns:
        Array: `u8[H * TILE_SIZE, W * TILE_SIZE, 3]` (`H = env.height`,
        `W = env.width`)."""
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
    """The egocentric version of `rgb`: the rendered image cropped to a
    `(2 * RADIUS + 1)`-tile square around the player and rotated so the
    player faces up. Out-of-view / occluded tiles are filled with the
    dimmed "unseen" grey.

    Args:
        state (State): the current state.

    Returns:
        Array: `u8[(2 * RADIUS + 1) * TILE_SIZE, (2 * RADIUS + 1) * TILE_SIZE, 3]`."""
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
