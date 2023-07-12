from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from flax import struct
from jax import Array


TILE_SIZE = 32
RED = jnp.asarray([255, 0, 0], dtype=jnp.uint8)
GREEN = jnp.asarray([0, 255, 0], dtype=jnp.uint8)
BLUE = jnp.asarray([0, 0, 255], dtype=jnp.uint8)
BLACK = jnp.asarray([0, 0, 0], dtype=jnp.uint8)
WHITE = jnp.asarray([255, 255, 255], dtype=jnp.uint8)
YELLOW = jnp.asarray([255, 255, 0], dtype=jnp.uint8)
PURPLE = jnp.asarray([255, 0, 255], dtype=jnp.uint8)
CYAN = jnp.asarray([0, 255, 255], dtype=jnp.uint8)
ORANGE = jnp.asarray([255, 128, 0], dtype=jnp.uint8)
PINK = jnp.asarray([255, 0, 128], dtype=jnp.uint8)
BROWN = jnp.asarray([128, 64, 0], dtype=jnp.uint8)
GRAY_10 = jnp.asarray([230, 230, 230], dtype=jnp.uint8)
GRAY_20 = jnp.asarray([205, 205, 205], dtype=jnp.uint8)
GRAY_40 = jnp.asarray([153, 153, 153], dtype=jnp.uint8)
GRAY_50 = jnp.asarray([128, 128, 128], dtype=jnp.uint8)
GRAY_70 = jnp.asarray([77, 77, 77], dtype=jnp.uint8)
GRAY_80 = jnp.asarray([51, 51, 51], dtype=jnp.uint8)
GRAY_90 = jnp.asarray([25, 25, 25], dtype=jnp.uint8)
GOLD = jnp.asarray([255, 215, 0], dtype=jnp.uint8)
SILVER = jnp.asarray([192, 192, 192], dtype=jnp.uint8)
BRONZE = jnp.asarray([205, 127, 50], dtype=jnp.uint8)
MAROON = jnp.asarray([128, 0, 0], dtype=jnp.uint8)
NAVY = jnp.asarray([0, 0, 128], dtype=jnp.uint8)
TEAL = jnp.asarray([0, 128, 128], dtype=jnp.uint8)
OLIVE = jnp.asarray([128, 128, 0], dtype=jnp.uint8)
LIME = jnp.asarray([0, 255, 0], dtype=jnp.uint8)
AQUA = jnp.asarray([0, 255, 255], dtype=jnp.uint8)
FUCHSIA = jnp.asarray([255, 0, 255], dtype=jnp.uint8)
SALMON = jnp.asarray([250, 128, 114], dtype=jnp.uint8)
TURQUOISE = jnp.asarray([64, 224, 208], dtype=jnp.uint8)
VIOLET = jnp.asarray([238, 130, 238], dtype=jnp.uint8)
INDIGO = jnp.asarray([75, 0, 130], dtype=jnp.uint8)
BEIGE = jnp.asarray([245, 245, 220], dtype=jnp.uint8)
MINT = jnp.asarray([189, 252, 201], dtype=jnp.uint8)
LAVENDER = jnp.asarray([230, 230, 250], dtype=jnp.uint8)
APRICOT = jnp.asarray([251, 206, 177], dtype=jnp.uint8)
MAUVE = jnp.asarray([224, 176, 255], dtype=jnp.uint8)
LILAC = jnp.asarray([200, 162, 200], dtype=jnp.uint8)
TAN = jnp.asarray([210, 180, 140], dtype=jnp.uint8)


def colour_chart(size: int = TILE_SIZE) -> Array:
    colours = [
        RED,
        GREEN,
        BLUE,
        BLACK,
        WHITE,
        YELLOW,
        PURPLE,
        CYAN,
        ORANGE,
        PINK,
        BROWN,
        GRAY_20,
        GRAY_50,
        GRAY_70,
        GRAY_90,
        GOLD,
        SILVER,
        BRONZE,
        MAROON,
        NAVY,
        TEAL,
        OLIVE,
        LIME,
        AQUA,
        FUCHSIA,
        SALMON,
        TURQUOISE,
        VIOLET,
        INDIGO,
        BEIGE,
        MINT,
        LAVENDER,
        APRICOT,
        MAUVE,
        LILAC,
        TAN,
    ]
    grid = jnp.zeros((size * len(colours), size * len(colours), 3), dtype=jnp.uint8)
    for i, colour in enumerate(colours):
        for j, colour in enumerate(colours):
            grid = grid.at[i * size : (i + 1) * size, j * size : (j + 1) * size].set(
                colour
            )
    return grid


class RenderingCache(struct.PyTreeNode):
    patches: Array
    """A flat set of patches representing the RGB values of each tile in the base map"""

    @classmethod
    def init(cls, grid: Array) -> RenderingCache:
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


def colorise_tile(tile: Array, colour: Array, background: Array = WHITE) -> Array:
    assert tile.shape == (
        TILE_SIZE,
        TILE_SIZE,
    ), "Tile must be of size TILE_SIZE, TILE_SIZE, 3, got {}".format(tile.shape)
    tile = jnp.stack([tile] * colour.shape[0], axis=-1)
    tile = jnp.where(tile, colour, background)
    return tile


def render_rectangle(size: int = TILE_SIZE, colour: Array = BLACK) -> Array:
    rectangle = jnp.ones((size - 2, size - 2), dtype=jnp.int32)
    rectangle = jnp.pad(rectangle, 1, "constant", constant_values=0)
    return colorise_tile(rectangle, colour)


def render_triangle_east(size: int = TILE_SIZE, colour: Array = RED) -> Array:
    triangle = jnp.ones((size, size), dtype=jnp.int32)
    triangle = jnp.tril(triangle, k=0)
    triangle = jnp.flip(triangle, axis=0)
    triangle = jnp.tril(triangle, k=0)
    triangle = jnp.roll(triangle, (0, size // 3))
    return colorise_tile(triangle, colour)


def render_triangle_south(size: int = TILE_SIZE, colour: Array = RED) -> Array:
    triangle = render_triangle_east(size)
    triangle = jnp.rot90(triangle, k=3)
    return colorise_tile(triangle, colour)


def render_triangle_west(size: int = TILE_SIZE, colour: Array = RED) -> Array:
    triangle = render_triangle_east(size)
    triangle = jnp.rot90(triangle, k=2)
    return colorise_tile(triangle, colour)


def render_triangle_north(size: int = TILE_SIZE, colour: Array = RED) -> Array:
    triangle = render_triangle_east(size)
    triangle = jnp.rot90(triangle, k=1)
    return colorise_tile(triangle, colour)


def render_diamond(size: int = TILE_SIZE, colour: Array = GOLD) -> Array:
    diamond = jnp.ones((size, size), dtype=jnp.int32)
    diamond = jnp.tril(diamond, k=size // 2.5)
    diamond = jnp.flip(diamond, axis=0)
    diamond = jnp.tril(diamond, k=size // 2.5)
    diamond = jnp.flip(diamond, axis=1)
    diamond = jnp.tril(diamond, k=size // 2.5)
    diamond = jnp.flip(diamond, axis=0)
    diamond = jnp.tril(diamond, k=size // 2.5)
    return colorise_tile(diamond, colour)


def render_door_closed(size: int = TILE_SIZE, colour: Array = BROWN) -> Array:
    frame_size = size - 6
    door = jnp.zeros((frame_size, frame_size), dtype=jnp.int32)
    door = jnp.pad(door, 1, "constant", constant_values=1)
    door = jnp.pad(door, 1, "constant", constant_values=0)
    door = jnp.pad(door, 1, "constant", constant_values=1)

    x_0 = size - size // 4
    y_centre = size // 2
    y_size = size // 5
    door = door.at[y_centre - y_size // 2 : y_centre + y_size // 2, x_0 : x_0 + 1].set(
        1
    )
    return colorise_tile(door, colour)


def render_door_locked(size: int = TILE_SIZE, colour: Array = BROWN) -> Array:
    frame_size = size - 4
    door = jnp.zeros((frame_size, frame_size), dtype=jnp.int32)
    door = jnp.pad(door, 2, "constant", constant_values=1)

    x_0 = size - size // 4
    y_centre = size // 2
    y_size = size // 5
    door = door.at[y_centre - y_size // 2 : y_centre + y_size // 2, x_0 : x_0 + 1].set(
        1
    )
    return colorise_tile(door, colour, background=colour / 2)


def render_door_open(size: int = TILE_SIZE, colour: Array = BROWN) -> Array:
    door = jnp.zeros((size, size), dtype=jnp.int32)
    door = door.at[0].set(1)
    door = door.at[3].set(1)
    door = door.at[:3, 0].set(1)
    door = door.at[:3, -1].set(1)
    return colorise_tile(door, colour)


def render_key(size: int = TILE_SIZE, colour: Array = BRONZE) -> Array:
    key = jnp.zeros((size, size), dtype=jnp.int32)

    # Handle (Round Part)
    handle_radius = size // 4
    handle_center = (size // 2, size // 4)
    y, x = jnp.ogrid[:size, :size]
    mask = (x - handle_center[0]) ** 2 + (
        y - handle_center[1]
    ) ** 2 <= handle_radius**2
    key = jnp.where(mask, 1, key)

    # Shaft (Straight Part)
    shaft_width = size // 8
    shaft_height = size // 2
    shaft_start = (size // 2 - shaft_width // 2, size // 2 - shaft_height // 2)
    shaft_end = (size // 2 + shaft_width // 2, size // 2 + shaft_height // 2)
    shaft_mask = jnp.logical_and(
        jnp.logical_and(x >= shaft_start[0], x <= shaft_end[0]),
        jnp.logical_and(y >= shaft_start[1], y <= shaft_end[1]),
    )
    key = jnp.where(shaft_mask, 1, key)

    # Tooth (Pointy End)
    tooth_width = size // 15
    tooth_height = size // 2
    tooth_position = (size // 2 - tooth_width // 2, size - tooth_height)
    tooth_mask = jnp.logical_and(
        jnp.logical_and(x >= tooth_position[0], x <= tooth_position[0] + tooth_width),
        jnp.logical_and(y >= tooth_position[1], y <= tooth_position[1] + tooth_height),
    )
    key = jnp.where(tooth_mask, 1, key)

    return colorise_tile(key, colour)


def render_floor(size: int = TILE_SIZE, colour: Array = WHITE) -> Array:
    floor = jnp.ones((size - 2, size - 2), dtype=jnp.int32)
    floor = jnp.pad(floor, 1, "constant", constant_values=0)
    return colorise_tile(floor, colour, background=GRAY_10)


def render_wall(size: int = TILE_SIZE, colour: Array = GRAY_80) -> Array:
    wall = jnp.ones((size, size), dtype=jnp.int32)
    return colorise_tile(wall, colour)


def tile_grid(grid: Array, tile: Array) -> Array:
    tiled = jnp.tile(tile, (*grid.shape, 1))
    return jnp.asarray(tiled, dtype=jnp.uint8)


@jax.jit
def build_sprites_registry() -> Dict[str, Any]:
    registry = {}

    wall = render_wall()
    floor = render_floor()
    player = render_triangle_east()
    goal = render_diamond()
    key = render_key()
    door_closed = render_door_closed()
    door_open = render_door_open()

    # 0: set wall sprites
    registry["wall"] = wall

    # 1: set floor sprites
    registry["floor"] = floor

    # 2: set player sprites
    registry["player"] = jnp.stack(
        [
            player,
            jnp.rot90(player, k=3),
            jnp.rot90(player, k=2),
            jnp.rot90(player, k=1),
        ]
    )

    # 3: set goal sprites
    registry["goal"] = goal

    # 4: set key sprites
    registry["key"] = key

    # 5: set door sprites
    door = jnp.zeros((4, 2, TILE_SIZE, TILE_SIZE, 3), dtype=jnp.uint8)

    door_closed_by_direction = jnp.stack(
        [
            jnp.rot90(door_closed, k=1),
            door_closed,
            jnp.rot90(door_closed, k=3),
            jnp.rot90(door_closed, k=2),
        ]
    )
    door = door.at[:, 0].set(door_closed_by_direction)

    door_open_by_direction = jnp.stack(
        [
            door_open,
            jnp.rot90(door_open, k=1),
            jnp.rot90(door_open, k=2),
            jnp.rot90(door_open, k=3),
        ]
    )
    door = door.at[:, 1].set(door_open_by_direction)

    registry["door"] = door

    return registry


SPRITES_REGISTRY: Dict[str, Any] = build_sprites_registry()


def render_background(
    grid: Array, sprites_registry: Dict[str, Any] = SPRITES_REGISTRY
) -> Array:
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
    floor_tile = tile_grid(grid, sprites_registry["floor"])
    background = jnp.where(mask[..., None], wall_tile, floor_tile)
    return background


def flatten_patches(
    image: Array, patch_size: Tuple[int, int] = (TILE_SIZE, TILE_SIZE)
) -> Array:
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
