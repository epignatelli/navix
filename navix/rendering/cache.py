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


"""The rendering cache and the patch <-> image plumbing behind `rgb`
observations.

An `rgb` observation is the grid drawn as `TILE_SIZE x TILE_SIZE`
sprites. The static parts (walls, floor, grid lines) never change during
an episode, so they are rendered once into a flat list of tile
`patches`, stored on `State.cache`, and only the cells holding moving
entities are re-drawn each step.
"""

from __future__ import annotations

from typing import Dict, Tuple

import jax
from jax import Array
import jax.numpy as jnp
from flax import struct

from ..grid import draw_grid_lines
from .registry import TILE_SIZE, SPRITES_REGISTRY


class RenderingCache(struct.PyTreeNode):
    """The pre-rendered background, carried on `State.cache` so `rgb`
    observations are cheap. Build one with `RenderingCache.init`; it is
    valid for any state on the same-shaped grid."""

    patches: Array
    """`u8[H * W + 1, TILE_SIZE, TILE_SIZE, 3]` - one RGB tile per grid
    cell (row-major), plus a trailing all-zero "discard pile" tile that
    off-grid entities render into and that `rgb` slices off."""

    @classmethod
    def init(cls, grid: Array) -> RenderingCache:
        """Renders `grid`'s static background (walls / floor / grid lines)
        once into the flat patch list.

        Args:
            grid (Array): `i32[H, W]` base map (`0` floor, `-1` wall).

        Returns:
            RenderingCache: the cache for that grid shape."""
        background = render_background(grid)
        patches = flatten_patches(background)

        # add discard pile
        patches = jnp.concatenate(
            [
                patches,
                jnp.zeros((1, TILE_SIZE, TILE_SIZE, 3), dtype=jnp.uint8),
            ],
            axis=0,
        )
        return cls(patches=patches)


def render_background(
    grid: Array, sprites_registry: Dict[str, Array] = SPRITES_REGISTRY
) -> Array:
    """Draws the static layer of an `rgb` frame: a wall sprite on every
    `-1` cell, a floor sprite (with MiniGrid-matched grid lines) on every
    `0` cell. No entities.

    Args:
        grid (Array): `i32[H, W]` base map.
        sprites_registry (dict): sprite lookup; defaults to the global
            `SPRITES_REGISTRY`.

    Returns:
        Array: `u8[H * TILE_SIZE, W * TILE_SIZE, 3]`."""
    image_width = grid.shape[0] * TILE_SIZE
    image_height = grid.shape[1] * TILE_SIZE
    n_channels = 3

    background = jnp.zeros((image_height, image_width, n_channels), dtype=jnp.uint8)
    grid_resized = jax.image.resize(
        grid, (grid.shape[0] * TILE_SIZE, grid.shape[1] * TILE_SIZE), method="nearest"
    )

    mask = jnp.asarray(grid_resized, dtype=bool)  # 0 = floor, 1 = wall
    # index by [entity_type, direction, open/closed, y, x, channel]
    wall_tile = tile_grid(grid, sprites_registry["wall"])
    # draw grid lines on the floor tile only, matching MiniGrid: its
    # Wall.render() fills the whole tile opaquely, completely covering
    # whatever grid line was drawn underneath, so walls never actually
    # show one - only empty/floor cells do. floor.png's own baked-in
    # border (a different colour, from before this fix) gets
    # overwritten by this rather than drawn alongside it. luminosity=32
    # (not draw_grid_lines' default of 100, MiniGrid's raw grey
    # constant) and corner_luminosity=54 empirically match a real
    # MiniGrid render - see draw_grid_lines' docstring for why the raw
    # constant renders too bold, and why the corner needs its own,
    # brighter value.
    floor_tile = tile_grid(
        grid,
        draw_grid_lines(
            sprites_registry["floor"],
            luminosity=jnp.asarray(32),
            corner_luminosity=jnp.asarray(54),
        ),
    )
    background = jnp.where(mask[..., None], wall_tile, floor_tile)
    return background


def tile_grid(grid: Array, tile: Array) -> Array:
    """Tiles a grid (H, W) with equal tiles `tiles` (w, h, 3) to get a final array
    of shape (H * h, W * w, 3) and dtype `jnp.uint8`"""
    tiled = jnp.tile(tile, (*grid.shape, 1))
    return jnp.asarray(tiled, dtype=jnp.uint8)


def flatten_patches(
    image: Array, patch_size: Tuple[int, int] = (TILE_SIZE, TILE_SIZE)
) -> Array:
    """Splits an image into a row-major list of fixed-size tiles - the
    inverse of `unflatten_patches`.

    Args:
        image (Array): `(H, W, C)`, with `H`/`W` multiples of
            `patch_size`.
        patch_size (tuple[int, int]): tile `(height, width)`; defaults to
            `(TILE_SIZE, TILE_SIZE)`.

    Returns:
        Array: `(H//ph * W//pw, ph, pw, C)`."""
    height = image.shape[0] // patch_size[0]
    width = image.shape[1] // patch_size[1]
    n_channels = image.shape[2]

    grid = image.reshape(height, patch_size[0], width, patch_size[1], n_channels)

    # Swap the first and second axes of the grid to revert the stacking order
    grid = jnp.swapaxes(grid, 1, 2)

    # Reshape the grid of tiles into the original list of tiles
    patches = grid.reshape(height * width, patch_size[0], patch_size[1], n_channels)

    return patches


def unflatten_patches(patches: Array, image_size: Tuple[int, int]) -> Array:
    """Reassembles a row-major list of tiles into an image - the inverse
    of `flatten_patches`.

    Args:
        patches (Array): `(n_tiles, ph, pw, C)`.
        image_size (tuple[int, int]): the target `(H, W)`;
            `H * W == n_tiles * ph * pw`.

    Returns:
        Array: `(H, W, C)`."""
    image_height = image_size[0]
    image_width = image_size[1]
    patch_height = patches.shape[1]
    patch_width = patches.shape[2]
    n_channels = patches.shape[3]

    # Reshape the list of tiles into a 2D grid
    grid = patches.reshape(
        image_height // patch_height,
        image_width // patch_width,
        patch_height,
        patch_width,
        n_channels,
    )

    # Swap the first and second axes of the grid to change the order of stacking
    grid = jnp.swapaxes(grid, 1, 2)

    # Reshape and stack the grid tiles horizontally and vertically to form the final image
    image = grid.reshape(image_height, image_width, n_channels)

    return image
