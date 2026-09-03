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


"""Array helpers for grid geometry - the low-level toolkit the
environments, `navix.actions` and `navix.observations` are built from.

Everything here is a pure JAX function over plain arrays; nothing knows
about `State` or `Entity`. Groups:

- coordinate <-> flat-index conversion (`coordinates`,
  `idx_from_coordinates`, ...);
- movement and rotation of a `(row, col)` position or a direction
  (`translate`, `rotate`, `translate_forward/left/right`);
- rotating an image patch to align it with a direction (`align`,
  `rotate_tile`);
- random placement (`random_positions`, `random_distinct_positions`,
  `random_position_far_from`, `random_directions`, `random_colour`);
- building maps (`room`, `two_rooms`, `vertical_wall`, `horizontal_wall`,
  `from_ascii_map`) and the multi-room grid helpers (`room_grid*`,
  `room_*`, `RoomsGrid`);
- cropping and first-person rendering (`crop`, `view_cone`,
  `draw_grid_lines`, `apply_minigrid_opacity`).

Convention: positions are `(row, col)`, directions are `0` east, `1`
south, `2` west, `3` north, and a "grid" is `i32[H, W]` with `0` = floor
and `-1` = wall.
"""

from __future__ import annotations
from functools import partial
import math


from typing import Callable, Dict, List, Tuple
import jax
import jax.numpy as jnp
from jax import Array
from flax import struct
from navix.rendering.registry import TILE_SIZE


Coordinates = Tuple[Array, Array]
"""A `(rows, cols)` pair of index arrays, as returned by `coordinates` -
the shape `jnp.where` / advanced indexing expect."""


def coordinates(grid: Array) -> Coordinates:
    """Returns a tuple of 2D coordinates [(col, row), ...] for each cell in the grid.
    A grid array of shape `i32[height, width]` will return a tuple of length (height * width),
    containing two arrays, each of shape `i32[2]`.

    Args:
        grid (Array): A 2D grid of shape (height, width).
    
    Returns:
        Tuple[Array, Array]: A tuple of two arrays containing the 2D coordinates of \
        each cell in the grid.
    """
    return tuple(jnp.mgrid[0 : grid.shape[0], 0 : grid.shape[1]])  # type: ignore


def idx_from_coordinates(grid: Array, coordinates: Array) -> Array:
    """Converts a batch of 2D coordinates [(col, row), ...] into a flat index

    Args:
        grid (Array): A 2D grid of shape (height, width).
        coordinates (Array): A batch of 2D coordinates of shape (batch_size, 2).

    Returns:
        Array: A flat index of shape `i32[batch_size]` for each coordinate in the batch.
    """
    coordinates = coordinates.T
    assert coordinates.shape[0] == 2, coordinates.shape

    idx = coordinates[0] * grid.shape[1] + coordinates[1]
    return jnp.asarray(idx, dtype=jnp.int32)


def coordinates_from_idx(grid: Array, idx: Array) -> Array:
    """Converts a flat index of shape `i32[]` into a 2D coordinate `i32[2]` containing \
    (col, row) data. The index is calculated as `idx = row * width + col`.
    
    Args:
        grid (Array): A 2D grid of shape (height, width).
        idx (Array): A flat index of shape `i32[]`.
        
    Returns:
        Array: A 2D coordinate of shape `i32[2]` containing the (col, row) data."""
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

    Args:
        grid (Array): A 2D grid of shape (height, width).
        address (Coordinates): A tuple of 2D coordinates (col, row).
        comparison_fn (Callable[[Array, Array], Array], optional): A comparison function. \
        Defaults to `jnp.greater_equal`.

    Returns:
        Array: A boolean mask of the same shape as `grid`.

    """
    mesh = jnp.mgrid[0 : grid.shape[0], 0 : grid.shape[1]]
    cond_1 = comparison_fn(mesh[0], address[0])
    cond_2 = comparison_fn(mesh[1], address[1])
    mask = jnp.asarray(jnp.logical_and(cond_1, cond_2), dtype=jnp.int32)
    return mask


def translate(
    position: Array, direction: Array, modulus: Array = jnp.asarray(1)
) -> Array:
    """Translates a point in a grid by a given direction and modulus.
    
    Args:
        position (Array): A 2D coordinate of shape `i32[2]` containing the (col, row) data.
        direction (Array): A direction in the range [0, 1, 2, 3] representing the \
        cardinal directions [east, south, west, north].
        modulus (Array, optional): The modulus of the translation. Defaults to jnp.asarray(1).

    Returns:
        Array: A 2D coordinate of shape `i32[2]` containing the (col, row) data.
        """
    moves = (
        lambda position: position + jnp.asarray((0, modulus)),  # east
        lambda position: position + jnp.asarray((modulus, 0)),  # south
        lambda position: position + jnp.asarray((0, -modulus)),  # west
        lambda position: position + jnp.asarray((-modulus, 0)),  # north
    )
    return jax.lax.switch(direction, moves, position)


def translate_forward(position: Array, forward_direction: Array, modulus: Array):
    """Translates a point in a grid by a given forward direction and modulus.
    
    Args:
        position (Array): A 2D coordinate of shape `i32[2]` containing the (col, row) data.
        forward_direction (Array): A direction in the range [0, 1, 2, 3] representing the \
        cardinal directions [east, south, west, north].
        modulus (Array): The modulus of the translation.
    
    Returns:
        Array: A 2D coordinate of shape `i32[2]` containing the (col, row) data."""
    return translate(position, forward_direction, modulus)


def translate_left(position: Array, forward_direction: Array, modulus: Array):
    """Translates a point in a grid by a given left direction and modulus.
    
    Args:
        position (Array): A 2D coordinate of shape `i32[2]` containing the (col, row) data.
        forward_direction (Array): A direction in the range [0, 1, 2, 3] representing the \
        cardinal directions [east, south, west, north].
        modulus (Array): The modulus of the translation.
    
    Returns:
        Array: A 2D coordinate of shape `i32[2]` containing the (col, row) data."""
    return translate(position, (forward_direction + 3) % 4, modulus)


def translate_right(position: Array, forward_direction: Array, modulus: Array):
    """Translates a point in a grid by a given right direction and modulus.
    
    Args:
        position (Array): A 2D coordinate of shape `i32[2]` containing the (col, row) data.
        forward_direction (Array): A direction in the range [0, 1, 2, 3] representing the \
        cardinal directions [east, south, west, north].
        modulus (Array): The modulus of the translation.
        
    Returns:
        Array: A 2D coordinate of shape `i32[2]` containing the (col, row) data."""
    return translate(position, (forward_direction + 1) % 4, modulus)


def rotate(direction: Array, spin: int) -> Array:
    """Changes a direction vectory by a given number of spins.
    
    Args:
        direction (Array): A direction vector of shape `i32[]` in the range [0, 3] \
        representing the cardinal directions [east, south, west, north].
        spin (int): The number of spins to apply.
        
    Returns:
        Array: A direction vector of shape `i32[]` in the range [0, 3] representing \
        the cardinal directions [east, south, west, north]."""
    return (direction + spin) % 4


def align(patch: Array, current_direction: Array, desired_direction: Array) -> Array:
    """Aligns a patch of the grid from the current direction to the desired direction.
    
    Args:
        patch (Array): A patch of the grid.
        current_direction (Array): The current direction in the range [0, 1, 2, 3] \
        representing the cardinal directions [east, south, west, north].
        desired_direction (Array): The desired direction in the range [0, 1, 2, 3] \
        representing the cardinal directions [east, south, west, north].
        
    Returns:
        Array: A patch of the grid aligned to the desired direction."""
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


def rotate_tile(patch: Array, num_times_90: Array) -> Array:
    """Rotates a patch of the grid by a given number of 90-degree rotations.

    Args:
        patch (Array): A patch of the grid.
        num_times_90 (int): The number of 90-degree rotations to apply.

    Returns:
        Array: A patch of the grid rotated by the given number of 90-degree rotations.
    """
    return jax.lax.switch(
        num_times_90,
        (
            lambda x: jnp.flip(jnp.swapaxes(x, 0, 1), axis=0),  # rot90
            lambda x: jnp.flip(jnp.flip(x, axis=0), axis=1),  # rot180
            lambda x: jnp.flip(jnp.swapaxes(x, 0, 1), axis=1),  # rot270
            lambda x: x,  # rot0
        ),
        patch,
    )


def random_positions(
    key: Array, grid: Array, n: int = 1, exclude: Array = jnp.asarray((-1, -1))
) -> Array:
    """Generates `n` random positions in the grid, excluding the `exclude` position.

    Args:
        key (Array): A random key.
        grid (Array): A 2D grid of shape (height, width).
        n (int, optional): The number of random positions to generate. Defaults to 1.
        exclude (Array, optional): The position to exclude. Defaults to jnp.asarray((-1, -1)).

    Returns:
        Array: A batch of random positions of shape `i32[n, 2]`."""
    probs = grid.reshape(-1)
    indices = idx_from_coordinates(grid, exclude)
    probs = probs.at[indices].set(-1) + 1.0
    idx = jax.random.categorical(key, jnp.log(probs), shape=(n,))
    position = coordinates_from_idx(grid, idx)
    return position.squeeze()


def random_distinct_positions(
    key: Array, grid: Array, n: int, exclude: Array = jnp.asarray((-1, -1))
) -> Array:
    """Generates `n` *mutually distinct* random positions in the grid,
    each also excluding `exclude` - unlike `random_positions(..., n=n)`,
    whose `n` draws are i.i.d. (`jax.random.categorical` samples with
    replacement) and so can collide with each other (see issue #172's
    PR: GoToObject/Fetch/PutNear all need genuinely distinct object
    positions to track a mission by position alone). Draws sequentially,
    each excluding every position drawn so far in addition to `exclude` -
    `n` is a small static Python int in every current caller, so the
    unrolled loop is fine under jit (same pattern as `jax.random.split`
    being called with a static count elsewhere in this codebase).

    Args:
        key (Array): A random key.
        grid (Array): A 2D grid of shape (height, width).
        n (int): The number of distinct positions to generate.
        exclude (Array, optional): Position(s) to also exclude, shape
            `(2,)` or `(k, 2)`. Defaults to `jnp.asarray((-1, -1))`.

    Returns:
        Array: `n` mutually distinct positions of shape `i32[n, 2]`."""
    exclude = jnp.atleast_2d(exclude)
    keys = jax.random.split(key, n)
    positions = []
    for i in range(n):
        excluded_so_far = jnp.concatenate([exclude, *positions], axis=0)
        pos = random_positions(keys[i], grid, n=1, exclude=excluded_so_far)
        positions.append(jnp.reshape(pos, (1, 2)))
    return jnp.concatenate(positions, axis=0)


def random_position_far_from(
    key: Array,
    grid: Array,
    reference: Array,
    min_distance: int = 2,
    exclude: Array = jnp.asarray((-1, -1)),
) -> Array:
    """Generates one random position at Chebyshev distance `>=
    min_distance` from `reference`, also excluding `exclude`.

    For `PutNear`-style tasks: quantified directly (500 seeds each),
    `random_distinct_positions` alone let 36% of `Navix-PutNear-6x6-N2-v0`
    episodes spawn with the "move" object already within Chebyshev
    distance 1 of the "drop near" target - trivially "solved" with no
    real navigation needed, unlike real MiniGrid's `PutNearEnv`, which
    explicitly rejects that via `reject_fn=near_obj` (see PR #191
    review's "New risks" section).

    Args:
        key (Array): A random key.
        grid (Array): A 2D grid of shape (height, width).
        reference (Array): The `(row, col)` position to stay away from.
        min_distance (int, optional): Minimum Chebyshev distance from
            `reference`. Defaults to 2 (i.e. not orthogonally/
            diagonally adjacent, and not the same cell).
        exclude (Array, optional): Position(s) to also exclude, shape
            `(2,)` or `(k, 2)`. Defaults to `jnp.asarray((-1, -1))`.

    Returns:
        Array: A position of shape `i32[2]`."""
    mesh_row, mesh_col = jnp.mgrid[0 : grid.shape[0], 0 : grid.shape[1]]
    chebyshev = jnp.maximum(jnp.abs(mesh_row - reference[0]), jnp.abs(mesh_col - reference[1]))
    too_close = (chebyshev < min_distance).reshape(-1)

    probs = grid.reshape(-1)
    exclude = jnp.atleast_2d(exclude)
    exclude_idx = idx_from_coordinates(grid, exclude)
    probs = probs.at[exclude_idx].set(-1)
    probs = jnp.where(too_close, -1, probs) + 1.0
    idx = jax.random.categorical(key, jnp.log(probs), shape=(1,))
    return coordinates_from_idx(grid, idx).squeeze()


def random_directions(key: Array, n=1) -> Array:
    """Generates `n` random directions in the range [0, 1, 2, 3] representing the \
        cardinal directions [east, south, west, north].
        
    Args:
        key (Array): A random key.
        n (int, optional): The number of random directions to generate. Defaults to 1.
        
    Returns:
        Array: A batch of random directions of shape `i32[n]`."""
    return jax.random.randint(key, (n,), 0, 4).squeeze()


def random_colour(key: Array, n=1) -> Array:
    """Generates `n` random colours in the range [0, 1, 2, 3, 4, 5].

    Args:
        key (Array): A random key.
        n (int, optional): The number of random colours to generate. Defaults to 1.

    Returns:
        Array: A batch of random colours of shape `u8[n]`."""
    return jax.random.randint(key, (n,), 0, 6).squeeze()


def positions_equal(a: Array, b: Array) -> Array:
    """Checks if two points are equal.

    Args:
        a (Array): A 2D coordinate of shape `i32[2]` containing the (col, row) data.
        b (Array): A 2D coordinate of shape `i32[2]` containing the (col, row) data.

    Returns:

    """
    if b.ndim == 1:
        b = b[None]
    if a.ndim == 1:
        a = a[None]
    assert a.ndim == b.ndim == 2, (a.shape, b.shape)
    is_equal = jnp.all(jnp.equal(a, b), axis=-1)
    assert is_equal.shape == (max(b.shape[0], a.shape[0]),)
    return is_equal


def room(height: int, width: int) -> Array:
    """Creates an array representing a room of size `height` x `width`, including
    a set of walls around the room. The room is represented as a 2D grid of shape
    `(height, width)`, including walls, with walls set to -1 and empty tiles set to 0.

    Args:
        height (int): The height of the room.
        width (int): The width of the room.

    Returns:
        Array: A 2D grid of shape `(height, width)` representing a room."""
    grid = jnp.zeros((height - 2, width - 2), dtype=jnp.int32)
    return jnp.pad(grid, 1, mode="constant", constant_values=-1)


def two_rooms(height: int, width: int, key: Array) -> Tuple[Array, Array]:
    """Creates a 2D grid representing two rooms of size `height` x `width`, separated
    by a wall. The rooms are represented as a 2D grid of shape `(height, width)`, \
    including walls, with walls set to -1 and empty tiles set to 0.

    Args:
        height (int): The height of the rooms.
        width (int): The width of the rooms.
        key (Array): A random key, determining the position of the wall separating the rooms.
    
    Returns:
        Tuple[Array, Array]: A tuple containing the 2D grid representing the rooms \
        and the column index of the wall separating the rooms."""
    # create room
    grid = jnp.zeros((height - 2, width - 2), dtype=jnp.int32)
    grid = jnp.pad(grid, 1, mode="constant", constant_values=-1)

    # add separation wall
    wall_at = jax.random.randint(key, (), 2, width - 2)
    grid = grid.at[1:-1, wall_at].set(-1)
    return grid, wall_at


def vertical_wall(
    grid: Array, row_idx: int, opening_col_idx: Array | None = None
) -> Array:
    """Creates a vertical wall in the grid at the given row index, with an opening at the \
        given column index.
    
    Args:
        grid (Array): A 2D grid of shape `(height, width)`.
        row_idx (int): The row index where the wall is placed.
        opening_col_idx (Array, optional): The column index where the opening is placed. \
        Defaults to None.
    
    Returns:
        Array: A 2D grid of shape `(height, width)` with a vertical wall."""
    rows = jnp.arange(1, grid.shape[0] - 1)
    cols = jnp.asarray([row_idx] * (grid.shape[0] - 2))
    positions = jnp.stack((rows, cols), axis=1)
    if opening_col_idx is not None:
        positions = jnp.delete(
            positions, opening_col_idx - 1, axis=0, assume_unique_indices=True
        )
    return positions


def horizontal_wall(
    grid: Array, col_idx: int, opening_row_idx: Array | None = None
) -> Array:
    """Creates a horizontal wall in the grid at the given column index, with an opening at the \
        given row index.
        
    Args:
        grid (Array): A 2D grid of shape `(height, width)`.
        col_idx (int): The column index where the wall is placed.
        opening_row_idx (Array, optional): The row index where the opening is placed. \
        Defaults to None.
    
    Returns:
        Array: A 2D grid of shape `(height, width)` with a horizontal wall."""
    rows = jnp.asarray([col_idx] * (grid.shape[1] - 2))
    cols = jnp.arange(1, grid.shape[1] - 1)
    positions = jnp.stack((rows, cols), axis=1)
    if opening_row_idx is not None:
        positions = jnp.delete(
            positions, opening_row_idx - 1, axis=0, assume_unique_indices=True
        )
    return positions


def room_grid_dims(room_size: int, num_rows: int, num_cols: int) -> Tuple[int, int]:
    """The `(height, width)` of a `room_grid` layout - adjacent rooms
    share a single-cell-thick dividing wall (verified against
    MiniGrid's actual `RoomGrid.__init__`), so the grid is smaller than
    `num_rows * room_size` naively.

    Args:
        room_size (int): The size (both height and width) of one room,
            including its own walls. Must be >= 3 (at least a 1-cell
            interior).
        num_rows (int): Number of rooms stacked vertically.
        num_cols (int): Number of rooms stacked horizontally.

    Returns:
        Tuple[int, int]: The `(height, width)` of the full grid."""
    height = (room_size - 1) * num_rows + 1
    width = (room_size - 1) * num_cols + 1
    return height, width


def room_grid(room_size: int, num_rows: int, num_cols: int) -> Array:
    """Creates the base occupancy grid for a `num_rows` x `num_cols`
    layout of `room_size` x `room_size` rooms: the outer border plus
    every internal room-dividing wall, with no doors punched through
    yet (every room is fully sealed off from its neighbours) - callers
    open specific cells with `open_wall` to place doors. `room_size`,
    `num_rows`, `num_cols` are always static (environment-registration-
    time) values, never per-episode traced ones, so this is plain
    Python control flow, not vectorised.

    Args:
        room_size (int): The size of one room, including its own walls.
        num_rows (int): Number of rooms stacked vertically.
        num_cols (int): Number of rooms stacked horizontally.

    Returns:
        Array: A 2D grid of shape `(height, width)`."""
    height, width = room_grid_dims(room_size, num_rows, num_cols)
    grid = jnp.zeros((height, width), dtype=jnp.int32)
    grid = grid.at[jnp.asarray([0, height - 1])].set(-1)
    grid = grid.at[:, jnp.asarray([0, width - 1])].set(-1)
    for i in range(1, num_rows):
        grid = grid.at[i * (room_size - 1)].set(-1)
    for j in range(1, num_cols):
        grid = grid.at[:, j * (room_size - 1)].set(-1)
    return grid


def room_top_left(room_size: int, i: int, j: int) -> Tuple[int, int]:
    """The `(row, col)` of room `(i, j)`'s top-left corner (its own
    wall, not its interior) in a `room_grid` layout.

    Args:
        room_size (int): The size of one room, including its own walls.
        i (int): The room's row index.
        j (int): The room's column index.

    Returns:
        Tuple[int, int]: The `(row, col)` of the room's top-left corner."""
    return i * (room_size - 1), j * (room_size - 1)


def room_interior_bounds(room_size: int, i: int, j: int) -> Tuple[int, int, int, int]:
    """The inclusive `(row_min, col_min, row_max, col_max)` interior
    bounds of room `(i, j)` in a `room_grid` layout - excludes the
    room's own walls, the range a position can be sampled from.

    Args:
        room_size (int): The size of one room, including its own walls.
        i (int): The room's row index.
        j (int): The room's column index.

    Returns:
        Tuple[int, int, int, int]: The inclusive interior bounds."""
    top_row, top_col = room_top_left(room_size, i, j)
    return top_row + 1, top_col + 1, top_row + room_size - 2, top_col + room_size - 2


def room_grid_door_position(
    key: Array, room_size: int, i: int, j: int, side: int
) -> Array:
    """A random position along room `(i, j)`'s shared wall on the given
    `side` - where `add_door` would place a door in MiniGrid's actual
    `RoomGrid`. Verified against MiniGrid's own door placement: a
    uniform random offset along the wall, excluding the corners.

    Args:
        key (Array): A random key.
        room_size (int): The size of one room, including its own walls.
        i (int): The room's row index.
        j (int): The room's column index.
        side (int): The wall side, using navix's own direction
            convention (`entities.Directions`): 0=east, 1=south,
            2=west, 3=north. Always a static (registration-time) value,
            never a per-episode traced one, so this is a plain `if`,
            not `jax.lax.switch`.

    Returns:
        Array: An `i32[2]` `(row, col)` position on the wall."""
    top_row, top_col = room_top_left(room_size, i, j)
    offset = jax.random.randint(key, (), minval=1, maxval=room_size - 1)
    if side == 0:  # east
        return jnp.asarray([top_row + offset, top_col + room_size - 1])
    elif side == 1:  # south
        return jnp.asarray([top_row + room_size - 1, top_col + offset])
    elif side == 2:  # west
        return jnp.asarray([top_row + offset, top_col])
    else:  # north
        return jnp.asarray([top_row, top_col + offset])


def room_mask(grid: Array, room_size: int, i: int, j: int) -> Array:
    """A boolean mask of `grid`'s shape, `True` only within room `(i,
    j)`'s interior (its own walls excluded) - unlike `unlock.py`'s
    `mask_by_coordinates` (which only tests "before a single row/col",
    fine for a 1-row-of-rooms layout corner-anchored at the origin),
    rooms in a general `room_grid` need all four bounds, since they
    aren't corner-anchored. Useful for scoping `random_positions`'
    sampling to one room.

    Args:
        grid (Array): A 2D grid of shape `(height, width)`.
        room_size (int): The size of one room, including its own walls.
        i (int): The room's row index.
        j (int): The room's column index.

    Returns:
        Array: A boolean mask of shape `(height, width)`."""
    row_min, col_min, row_max, col_max = room_interior_bounds(room_size, i, j)
    rows, cols = coordinates(grid)
    return (
        (rows >= row_min) & (rows <= row_max) & (cols >= col_min) & (cols <= col_max)
    )


def open_wall(grid: Array, position: Array) -> Array:
    """Removes a single wall cell (e.g. to place a door through it) by
    setting it to floor (`0`) - the door's own entity (not the grid)
    then controls whether the player can actually pass through.

    Args:
        grid (Array): A 2D grid of shape `(height, width)`.
        position (Array): The `(row, col)` position to open.

    Returns:
        Array: The updated grid."""
    return grid.at[position[0], position[1]].set(0)


def crop(
    grid: Array, origin: Array, direction: Array, radius: int, padding_value: int = 100
) -> Array:
    """Crops a grid around a given origin, facing a given direction, with a given radius.

    Args:
        grid (Array): A 2D grid of shape `(height, width)`.
        origin (Array): The origin of the crop.
        direction (Array): The direction the crop is facing.
        radius (int): The radius of the crop.
        padding_value (int, optional): The padding value. Defaults to 0.

    Returns:
        Array: A cropped grid."""
    diameter = radius * 2

    # pad with radius
    padding = [(diameter, diameter), (diameter, diameter)]
    for _ in range(len(grid.shape) - 2):
        padding.append((0, 0))

    padded = jnp.pad(grid, padding, constant_values=padding_value)

    # translate the grid such that the agent is `radius` away from the top and left edges
    translated = jnp.roll(padded, -jnp.asarray(origin), axis=(0, 1))

    # crop such that the agent is in the centre of the grid
    cropped = translated[: 2 * diameter + 1, : 2 * diameter + 1]

    # rotate such that the agent is facing north
    rotated = rotate_tile(cropped, direction)

    # if radius is 6
    cropped = rotated.at[: diameter + 1, radius : diameter * 2 - radius + 1].get(
        fill_value=padding_value
    )
    return jnp.asarray(cropped, dtype=grid.dtype)


def apply_minigrid_opacity(image: Array, opacity: Array = jnp.asarray(0.7)) -> Array:
    """Applies minigrid opacity to the given image, used in
    `minigrid.wrappers.RGBImgPartialObsWrapper`. The default MiniGrid opacity is 0.7.

    Args:
        image (Array): The input image to which opacity is applied.
        opacity (Array, optional): The opacity value to apply. Defaults to 0.7.

    Returns:
        Array: The input image with applied opacity.
    """
    return jax.numpy.asarray(255 - opacity * (255 - image), dtype=jax.numpy.uint8)


def draw_grid_lines(
    tile: Array,
    luminosity: Array = jnp.asarray(100),
    corner_luminosity: Array | None = None,
) -> Array:
    """Draws grid lines on the given tile.

    `luminosity` defaults to MiniGrid's own raw grey constant
    (`COLORS["grey"]`), used as-is for e.g. walls. But MiniGrid's grid
    lines specifically are drawn as a sub-pixel-width, anti-aliased
    strip - at typical tile sizes only *partially* covered by that raw
    colour, blending it with the background - whereas this function
    does a hard, fully-opaque fill of `line_thickness` whole pixels
    with no antialiasing. Using the raw `100` here renders visibly
    bolder/brighter than MiniGrid's actual output; pass a lower value
    (e.g. `~32`, empirically matched against a real MiniGrid render -
    see `rendering/cache.py::render_background`) to compensate.

    `corner_luminosity` fills just the `line_thickness` x
    `line_thickness` corner block with a separate value, defaulting to
    `luminosity` (i.e. no distinct corner treatment) when omitted. In
    MiniGrid, the top and left line strips are drawn independently and
    both get anti-aliased, so the corner - covered by both - ends up
    brighter than either strip alone; a single flat `luminosity` for
    the whole line can't reproduce that without also being tuned to a
    higher value at just the corner.

    Args:
        tile (Array): The input tile to which grid lines are drawn.

    Returns:
        Array: The tile with drawn grid lines.
    """
    # Draw lines (top and left edges) at 3.1% of the tile size as per
    # minigrid.core.Grid.render_tile. TILE_SIZE is a static Python int
    # (not a runtime/traced value), so line_thickness must be a plain
    # Python int too - jnp.ceil() returns a float-dtype array, which
    # can't be used as a slice bound (`tile.at[:line_thickness, :]`
    # raises `TypeError: Only integer scalar arrays can be converted
    # to a scalar index`).
    line_thickness = math.ceil(TILE_SIZE * 0.031)
    corner_luminosity = luminosity if corner_luminosity is None else corner_luminosity
    tile = tile.at[:line_thickness, :].set(luminosity)
    tile = tile.at[:, :line_thickness].set(luminosity)
    tile = tile.at[:line_thickness, :line_thickness].set(corner_luminosity)
    return tile


def view_cone(transparency_map: Array, origin: Array, radius: int) -> Array:
    """Computes the view cone of a given origin in a grid with a given radius.
    The view cone is a boolean map of transparent (1) and opaque (0) tiles, indicating
    whether a tile is visible from the origin or not.

    Args:
        transparency_map (Array): A boolean map of transparent (1) and opaque (0) tiles.
        origin (Array): The origin of the view cone.
        radius (int): The radius of the view cone.

    Returns:
        Array: The view cone of the given origin in the grid with the given radius."""

    def fin_diff(array, _):
        array = jnp.roll(array, -1, axis=0) + array + jnp.roll(array, +1, axis=0)
        array = jnp.roll(array, -1, axis=1) + array + jnp.roll(array, +1, axis=1)
        # this accumulates path *counts*, not just reachability - each
        # step lets every unit of mass split into up to 9 copies of
        # itself (a 3x3 neighbourhood sum), so the total grows ~9x per
        # step and only the downstream `view > 0` threshold is ever
        # read. In an open area, that overflows int32 (~2.1e9) by
        # cone radius 12 - beyond that, a wrapped-negative cell is
        # indistinguishable from a genuinely unreachable one, so
        # overflow silently *removes* visibility rather than erroring.
        # Clamping to a boolean flood (min with 1) after every step is
        # behaviour-preserving below the overflow threshold (verified:
        # identical `> 0` masks at every radius that doesn't overflow)
        # and immune to it above, since a boolean value can never
        # overflow regardless of radius.
        array = jnp.minimum(array, 1)
        return array * transparency_map, ()

    # initialise the field to all zeros, except at the source (agent's position)
    mask = jnp.zeros_like(transparency_map).at[tuple(origin)].set(1)

    # start the diffusion process using finite differences
    # if radius is small, it should be fast enough to compile
    MIN_SCAN_RADIUS = 10
    if radius <= MIN_SCAN_RADIUS:
        view = mask
        for _ in range(radius):
            view = fin_diff(view, None)[0]
    else:
        view = jax.lax.scan(fin_diff, mask, None, radius)[0]

    # view has anything that is visible > 0
    # we now set a hard threshold > 0, but we can also think in the future
    # to use a cutoff at a different value to mimic the effect of a torch
    vis_free = view > 0

    # add frontier obstacles
    # frontier obstacles = opaque cells neighbouring any visible-free cell (8-neighbourhood)
    opaque = transparency_map == 0
    nb = (
        vis_free
        | jnp.roll(vis_free, +1, 0)
        | jnp.roll(vis_free, -1, 0)
        | jnp.roll(vis_free, +1, 1)
        | jnp.roll(vis_free, -1, 1)
        | jnp.roll(jnp.roll(vis_free, +1, 0), +1, 1)
        | jnp.roll(jnp.roll(vis_free, +1, 0), -1, 1)
        | jnp.roll(jnp.roll(vis_free, -1, 0), +1, 1)
        | jnp.roll(jnp.roll(vis_free, -1, 0), -1, 1)
    )
    frontier = nb & opaque

    # final visible = transparent region plus blocking frontier
    visible = vis_free | frontier
    visible = visible.at[tuple(origin)].set(True)

    return visible.astype(transparency_map.dtype)


def from_ascii_map(ascii_map: str, mapping: Dict[str, int] = {}) -> Array:
    """Converts an ASCII map into a 2D grid. The ASCII map is a string where each character
    represents a tile in the grid. The mapping dictionary can be used to map ASCII characters
    to integer values. By default, the mapping is as follows:
    - `#` is mapped to -1
    - `.` is mapped to 0
    
    Args:
        ascii_map (str): The ASCII map.
        mapping (Dict[str, int], optional): A dictionary mapping ASCII characters to integer \
        values. Defaults to {}.
    
    Returns:
        Array: A 2D grid representing the ASCII map."""
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


class RoomsGrid(struct.PyTreeNode):
    """A grid of rooms. Each room is represented as a 2D grid of shape `(room_height, room_width)`,
    with walls set to -1 and empty tiles set to 0. The grid of rooms is represented as a 2D grid of
    shape `(rows * (room_height + 1), cols * (room_width + 1))`, with walls set to -1 and empty tiles
    set to 0. The grid of rooms is represented as a 2D grid of shape `(rows * (room_height + 1), cols * (room_width + 1))`,
    with walls set to -1 and empty tiles set to 0."""

    room_starts: Array  # shape (rows, cols)
    room_size: Tuple[int, int]

    @classmethod
    def create(
        cls, num_rows: int, num_cols: int, room_size: Tuple[int, int]
    ) -> RoomsGrid:
        """Creates a grid of rooms with the given number of rows and columns, and the given room size.

        Args:
            num_rows (int): The number of rows.
            num_cols (int): The number of columns.
            room_size (Tuple[int, int]): The size of each room `(height, width)`.

        Returns:
            RoomsGrid: A grid of rooms."""
        # generate rooms grid
        height = num_rows * (room_size[0] + 1)
        width = num_cols * (room_size[1] + 1)
        starts = jnp.mgrid[
            : height : room_size[0] + 1,
            : width : room_size[1] + 1,
        ].transpose(1, 2, 0)
        starts = jnp.asarray(starts, dtype=jnp.int32)
        sizes = jnp.ones((num_rows, num_cols, 2)) * jnp.asarray([[[room_size]]])
        sizes = jnp.asarray(sizes, dtype=jnp.int32)
        return cls(starts, room_size)

    def get_grid(self, occupied_positions: Array | None = None) -> Array:
        """Computes the array representation of the grid of rooms, with walls set to \
        -1 and empty tiles set to 0.
        
        Args:
            occupied_positions (Array, optional): A batch of extra occupied positions \
            of shape `(n, 2)`. Defaults to None.
        
        Returns:
            Array: A 2D grid of shape `(rows * (room_height + 1), cols * (room_width + 1))`."""
        room_size = self.room_size
        num_rows, num_cols = self.room_starts.shape[:2]
        grid = jnp.zeros(
            (1 + num_rows * (room_size[0] + 1), 1 + num_cols * (room_size[1] + 1))
        )
        grid = grid.at[jnp.arange(num_rows + 1) * (room_size[0] + 1)].set(-1)
        grid = grid.at[:, jnp.arange(num_cols + 1) * (room_size[1] + 1)].set(-1)

        if occupied_positions is not None:
            grid = grid.at[tuple(occupied_positions.T)].set(0)
        return grid

    def position_in_room(self, row: Array, col: Array, *, key: Array) -> Array:
        """Generates a random position in a given room.

        Args:
            row (Array): The row index of the room.
            col (Array): The column index of the room.
            key (Array): A random key.

        Returns:
            Array: A random position in the given room."""
        k1, k2 = jax.random.split(key)
        local_row = jax.random.randint(k1, (), minval=1, maxval=self.room_size[0])
        local_col = jax.random.randint(k2, (), minval=1, maxval=self.room_size[1])
        return jnp.asarray([local_row, local_col]) + self.room_starts[row, col]

    @partial(jax.jit, static_argnums=3)
    def position_on_border(
        self, row: Array, col: Array, side: int, *, key: Array
    ) -> Array:
        """Generates a random position on the border of a given room.
        Side is 0: west, 1: east, 2: north, 3: south (like padding)

        Args:
            row (Array): The row index of the room.
            col (Array): The column index of the room.
            side (int): The side of the room.
            key (Array): A random key.

        Returns:
            Array: A random position on the border of the given room."""
        starts = self.room_starts[row, col]
        room_size = self.room_size
        if side == 0:
            idx = jax.random.randint(key, (), minval=1, maxval=room_size[0] + 1)
            pos = (starts[0] + idx, starts[1])
        elif side == 1:
            idx = jax.random.randint(key, (), minval=1, maxval=room_size[0] + 1)
            pos = (starts[0] + idx, starts[1] + room_size[1] + 1)
        elif side == 2:
            idx = jax.random.randint(key, (), minval=1, maxval=room_size[1] + 1)
            pos = (starts[0], starts[1] + idx)
        elif side == 3:
            idx = jax.random.randint(key, (), minval=1, maxval=room_size[1] + 1)
            pos = (starts[0] + room_size[0] + 1, starts[1] + idx)
        else:
            raise ValueError("Side should be less than 4 and greater than -1")
        return jnp.asarray(pos)
