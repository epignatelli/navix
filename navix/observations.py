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
- **rgb** - a rendered image, `uint8`, each cell MiniGrid's `TILE_SIZE x TILE_SIZE`
  tile; shape `(H * TILE_SIZE, W * TILE_SIZE, 3)`.

`Environment` infers the matching `observation_space` for these built-in
functions; a custom `observation_fn` needs `observation_space` passed
explicitly.
"""

from __future__ import annotations
import jax

import jax.numpy as jnp
from jax import Array

from .rendering import minigrid_tiles
from .rendering.cache import TILE_SIZE
from .components import (
    EMPTY_POCKET_ID,
    Directional,
    HasColour,
    Openable,
    Pickable,
)
from .states import State
from .grid import (
    align,
    idx_from_coordinates,
    crop,
    process_vis,
)
from .entities import Entities, EntityIds


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
    occupies it (`0` for empty floor, `EntityIds.WALL` for the walls
    `state.grid` marks `-1`), fully observable.

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
    grid = grid_tags(state).reshape(-1).at[indices].set(tags, mode="drop")
    # unflatten patches to reconstruct the grid
    return grid.reshape(shape)


def grid_tags(state: State) -> Array:
    """`state.grid` with its walls (`-1`) written as `EntityIds.WALL` and
    free cells left at `0`, the base layer the categorical observations
    write entity tags over.

    Args:
        state (State): the current state.

    Returns:
        Array: `i32[H, W]`."""
    wall = EntityIds.WALL.astype(state.grid.dtype)
    return jnp.where(state.grid == -1, wall, state.grid)


def first_person_vis(state: State) -> Array:
    """Which cells of the first-person window the player can see.

    The transparency map is built in world coordinates - free cells and
    transparent entities let sight through, everything else blocks it -
    then cropped to the egocentric window, which is the frame MiniGrid's
    `Grid.process_vis` is defined in. `padding_value=0` reads off-map as
    opaque, so sight cannot leave the map and come back.

    Args:
        state (State): the current state.

    Returns:
        Array: `bool[2 * RADIUS + 1, 2 * RADIUS + 1]`, the visibility
        mask over the cropped window."""
    transparency_map = jnp.where(state.grid == 0, 1, 0)
    positions = state.get_positions()
    transparent = state.get_transparency()
    # a picked-up entity's position (DISCARD_PILE_COORDS = (0, -1)) is
    # off-grid - .at[].set()'s default mode wraps negative components
    # around (numpy semantics) rather than dropping them, silently
    # marking a real cell with the carried item's transparency. Push
    # off-grid positions to be explicitly out of bounds first, so
    # mode="drop" discards those writes instead.
    H, W = state.grid.shape
    row, col = positions[..., 0], positions[..., 1]
    on_grid = (row >= 0) & (row < H) & (col >= 0) & (col < W)
    row = jnp.where(on_grid, row, H)
    col = jnp.where(on_grid, col, W)
    transparency_map = transparency_map.at[row, col].set(transparent, mode="drop")

    player = state.get_player()
    window = crop(
        transparency_map, player.position, player.direction, RADIUS, padding_value=0
    )
    return process_vis(window > 0)


def pocket_symbol(state: State) -> Array:
    """The `(tag, colour, state)` triple for the player's own cell.

    MiniGrid's `gen_obs_grid` writes whatever the player is carrying into
    its own cell of the observation, and an empty cell when it carries
    nothing.

    Args:
        state (State): the current state.

    Returns:
        Array: `u8[3]`, the carried entity's symbol, or the floor
        symbol when the pocket is empty."""
    pocket = state.get_player().pocket
    symbol = jnp.asarray([EntityIds.FLOOR, 0, 0], dtype=jnp.uint8)
    for entity_class in state.entities:
        entity = state.entities[entity_class]
        if not isinstance(entity, Pickable):
            continue
        held = (entity.id == pocket) & (pocket != EMPTY_POCKET_ID)
        if isinstance(entity, HasColour):
            colour = jnp.asarray(entity.colour, dtype=jnp.int32)
        else:
            colour = jnp.zeros_like(entity.id)
        # a carried object is never a door, so its state channel is 0,
        # exactly as MiniGrid's `WorldObj.encode` leaves it.
        carried = jnp.stack(
            [
                jnp.asarray(entity.tag, dtype=jnp.int32),
                colour,
                jnp.zeros_like(colour),
            ],
            axis=-1,
        )
        # only one entity can be in the pocket, so the max over instances
        # is that entity's symbol, and zeros when this class holds none.
        chosen = jnp.max(jnp.where(held[..., None], carried, 0), axis=0)
        symbol = jnp.where(jnp.any(held), chosen.astype(jnp.uint8), symbol)
    return symbol


def categorical_first_person(state: State) -> Array:
    """The egocentric version of `categorical`: one tag per cell, cropped
    to a `(2 * RADIUS + 1)` square around the player and rotated so the
    player sits at the bottom-centre facing up. Cells occluded by a wall or
    outside MiniGrid's visibility rule are set to `0` (`EntityIds.UNKNOWN`,
    not seen); off-map cells in sight read as walls. The player's own cell reports what it
    is carrying (`pocket_symbol`), as MiniGrid's `gen_obs_grid` does, and
    falls back to `PLAYER` when the pocket is empty - a free cell reads
    `0` in this encoding, the same value as "not seen", so writing
    MiniGrid's empty cell there would hide the player's own position
    instead of marking it.

    Args:
        state (State): the current state.

    Returns:
        Array: `i32[2 * RADIUS + 1, 2 * RADIUS + 1]`."""
    view = first_person_vis(state)

    # a picked-up entity's position is off-grid; push it out of bounds so
    # mode="drop" discards the write rather than wrapping it onto a real
    # cell (see first_person_vis).
    H, W = state.grid.shape
    positions = state.get_positions()
    row, col = positions[..., 0], positions[..., 1]
    on_grid = (row >= 0) & (row < H) & (col >= 0) & (col < W)
    row = jnp.where(on_grid, row, H)
    col = jnp.where(on_grid, col, W)
    player = state.get_player()

    # get categorical representation
    tags = state.get_tags()
    obs = grid_tags(state).at[row, col].set(tags, mode="drop")

    # the player's own cell reports the pocket, PLAYER when it is empty -
    # written outright, since an open door under the player shares its
    # cell and the scatter above keeps either tag
    pocket = pocket_symbol(state)[0].astype(obs.dtype)
    carrying = state.get_player().pocket != EMPTY_POCKET_ID
    obs = obs.at[tuple(player.position.T)].set(
        jnp.where(carrying, pocket, EntityIds.PLAYER.astype(obs.dtype))
    )

    # off-map cells read as walls, as in MiniGrid's padded slice; the
    # mask then hides whatever is out of sight.
    obs = crop(obs, player.position, player.direction, RADIUS, int(EntityIds.WALL))
    obs = obs * view

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
    return encode_grid(state, with_player=True)


def encode_grid(state: State, with_player: bool) -> Array:
    """`symbolic`'s encoding, optionally leaving the player out, so the
    cell under it reads what it stands on - which is what MiniGrid draws
    the agent over in `rgb`."""
    # initialise as all floors
    H, W = state.grid.shape
    obs = jnp.zeros((H, W, 3), dtype=jnp.uint8)
    wall_symbol = jnp.array([EntityIds.WALL, 5, 0], dtype=jnp.uint8)
    floor_symbol = jnp.array([EntityIds.FLOOR, 0, 0], dtype=jnp.uint8)
    obs = jnp.where(state.grid[..., None] == -1, wall_symbol, floor_symbol)

    # place entities
    for entity_class in state.entities:
        if entity_class == Entities.PLAYER and not with_player:
            continue
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
    player and rotated so the player faces up. Cells occluded by a wall
    or outside MiniGrid's visibility rule read `(UNKNOWN, 0, 0)`, the
    "not seen" symbol MiniGrid's `Grid.encode` writes; off-map cells that
    are in sight read as walls, as MiniGrid's padded slice does. The
    player's own cell shows what it is carrying (`pocket_symbol`).

    Args:
        state (State): the current state.

    Returns:
        Array: `u8[2 * RADIUS + 1, 2 * RADIUS + 1, 3]`."""
    obs = symbolic(state)

    # the player's own cell reports the pocket, as in MiniGrid
    player = state.get_player()
    obs = obs.at[tuple(player.position.T)].set(pocket_symbol(state))

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

    # mask after the crop so off-map walls out of sight read as unseen
    view = first_person_vis(state)
    unknown_symbol = jnp.zeros(3, dtype=jnp.uint8)
    return jnp.where(view[..., None], obs, unknown_symbol)


def rgb(state: State) -> Array:
    """The whole grid rendered as an RGB image, fully observable - the
    frame MiniGrid's `RGBImgObsWrapper` returns. Each cell is MiniGrid's
    own tile for its symbolic encoding (`rendering.minigrid_tiles`), the
    player drawn in its real direction over what it stands on, and the
    cells `rgb_first_person` would show highlighted.

    Args:
        state (State): the current state.

    Returns:
        Array: `u8[H * TILE_SIZE, W * TILE_SIZE, 3]` (`H = env.height`,
        `W = env.width`)."""
    symbols = encode_grid(state, with_player=False).astype(jnp.int32)
    player = state.get_player()
    agent = jnp.zeros(symbols.shape[:2], dtype=jnp.int32)
    agent = agent.at[tuple(player.position)].set(1 + player.direction)
    table = jnp.asarray(minigrid_tiles.tiles(TILE_SIZE))
    highlight = world_vis(state).astype(jnp.int32)
    return tile_image(
        table[agent, highlight, symbols[..., 0], symbols[..., 1], symbols[..., 2]]
    )


def world_vis(state: State) -> Array:
    """`first_person_vis` carried back to world coordinates: which cells
    of the grid the player can see. Each cell's own `(row, col)` is
    cropped with the same window, so the scatter needs no inverse of the
    rotation.

    Args:
        state (State): the current state.

    Returns:
        Array: `bool[H, W]`."""
    H, W = state.grid.shape
    rows, cols = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing="ij")
    player = state.get_player()
    coords = crop(
        jnp.stack([rows, cols], axis=-1),
        player.position,
        player.direction,
        RADIUS,
        padding_value=-1,
    )
    # off-map and unseen cells are pushed out of bounds, so mode="drop"
    # discards them
    seen = first_person_vis(state) & (coords[..., 0] >= 0)
    row = jnp.where(seen, coords[..., 0], H)
    col = jnp.where(seen, coords[..., 1], W)
    return jnp.zeros((H, W), dtype=jnp.bool_).at[row, col].set(True, mode="drop")


def tile_image(patchwork: Array) -> Array:
    """`(rows, cols, TILE_SIZE, TILE_SIZE, 3)` tiles laid out as one image."""
    obs = jnp.swapaxes(patchwork, 1, 2)
    shape = obs.shape
    return obs.reshape(shape[0] * shape[1], shape[2] * shape[3], *shape[4:])


def rgb_first_person(state: State) -> Array:
    """The egocentric RGB view MiniGrid's `RGBImgPartialObsWrapper` returns:
    `symbolic_first_person` drawn with MiniGrid's own tiles
    (`rendering.minigrid_tiles`). Visible cells are highlighted, unseen
    ones are the dark empty tile, and the player is drawn facing up over
    whatever it carries.

    Args:
        state (State): the current state.

    Returns:
        Array: `u8[(2 * RADIUS + 1) * TILE_SIZE, (2 * RADIUS + 1) * TILE_SIZE, 3]`."""
    symbols = symbolic_first_person(state).astype(jnp.int32)
    # the player faces up (MiniGrid's direction 3) at the bottom centre
    agent = jnp.zeros(symbols.shape[:2], dtype=jnp.int32).at[2 * RADIUS, RADIUS].set(4)
    # an unseen cell is MiniGrid's untinted empty tile
    seen = symbols[..., 0] != int(EntityIds.UNKNOWN)
    kind = jnp.where(seen, symbols[..., 0], int(EntityIds.FLOOR))
    table = jnp.asarray(minigrid_tiles.tiles(TILE_SIZE))
    return tile_image(
        table[agent, seen.astype(jnp.int32), kind, symbols[..., 1], symbols[..., 2]]
    )
