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
from functools import partial


from typing import Callable, Dict, List, Tuple
import jax
import jax.numpy as jnp
from jax import Array
import jax.tree_util as jtu
from flax import struct


Coordinates = Tuple[Array, Array]


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


def crop(
    grid: Array, origin: Array, direction: Array, radius: int, padding_value: int = 0
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
    input_shape = grid.shape
    # assert radius % 2, "Radius must be an odd number"
    # mid = jnp.asarray([g // 2 for g in grid.shape[:2]])
    # translated = jnp.roll(grid, mid - origin, axis=(0, 1))

    # # crop such that the agent is in the centre of the grid
    # cropped = translated.at[: 2 * radius + 1, : 2 * radius + 1].get(
    #     fill_value=padding_value
    # )

    # pad with radius
    padding = [(radius, radius), (radius, radius)]
    for _ in range(len(input_shape) - 2):
        padding.append((0, 0))

    padded = jnp.pad(grid, padding, constant_values=padding_value)

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

    cropped = rotated.at[: radius + 1].get(fill_value=padding_value)
    return jnp.asarray(cropped, dtype=grid.dtype)


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
        return jnp.asarray(local_row, local_col) + self.room_starts[row, col]

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
