import jax
import jax.numpy as jnp
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
            grid = grid.at[
                i * size : (i + 1) * size, j * size : (j + 1) * size
            ].set(colour)
    return grid


def colorise_tile(tile: Array, colour: Array, background: Array = GRAY_90) -> Array:
    tile = jnp.stack([tile] * colour.shape[0], axis=-1)
    tile = jnp.where(tile, colour, background)
    return tile


def rectangle_tile(size: int = TILE_SIZE, colour: Array = BLACK) -> Array:
    rectangle = jnp.ones((size - 2, size - 2), dtype=jnp.int32)
    rectangle = jnp.pad(rectangle, 1, "constant", constant_values=0)
    return colorise_tile(rectangle, colour)


def triangle_east_tile(size: int = TILE_SIZE, colour: Array = RED) -> Array:
    triangle = jnp.ones((size, size), dtype=jnp.int32)
    triangle = jnp.tril(triangle, k=0)
    triangle = jnp.flip(triangle, axis=0)
    triangle = jnp.tril(triangle, k=0)
    triangle = jnp.roll(triangle, (0, size // 3))
    return colorise_tile(triangle, colour)


def triangle_south_tile(size: int = TILE_SIZE, colour: Array = RED) -> Array:
    triangle = triangle_east_tile(size)
    triangle = jnp.rot90(triangle, k=3)
    return colorise_tile(triangle, colour)


def triangle_west_tile(size: int = TILE_SIZE, colour: Array = RED) -> Array:
    triangle = triangle_east_tile(size)
    triangle = jnp.rot90(triangle, k=2)
    return colorise_tile(triangle, colour)


def triangle_north_tile(size: int = TILE_SIZE, colour: Array = RED) -> Array:
    triangle = triangle_east_tile(size)
    triangle = jnp.rot90(triangle, k=1)
    return colorise_tile(triangle, colour)


def diamond_tile(size: int = TILE_SIZE, colour: Array = GOLD) -> Array:
    diamond = jnp.ones((size, size), dtype=jnp.int32)
    diamond = jnp.tril(diamond, k=size // 2.5)
    diamond = jnp.flip(diamond, axis=0)
    diamond = jnp.tril(diamond, k=size // 2.5)
    diamond = jnp.flip(diamond, axis=1)
    diamond = jnp.tril(diamond, k=size // 2.5)
    diamond = jnp.flip(diamond, axis=0)
    diamond = jnp.tril(diamond, k=size // 2.5)
    return colorise_tile(diamond, colour)


def door_tile(size: int = TILE_SIZE, colour: Array = BROWN) -> Array:
    frame_size = TILE_SIZE - 6
    door = jnp.zeros((frame_size, frame_size), dtype=jnp.int32)
    door = jnp.pad(door, 1, "constant", constant_values=1)
    door = jnp.pad(door, 1, "constant", constant_values=0)
    door = jnp.pad(door, 1, "constant", constant_values=1)

    x_0 = TILE_SIZE - TILE_SIZE // 4
    y_centre = TILE_SIZE // 2
    y_size = TILE_SIZE // 5
    door = door.at[y_centre - y_size // 2:y_centre + y_size // 2, x_0:x_0 + 1].set(1)
    return colorise_tile(door, colour)


def key_tile(size: int = TILE_SIZE, colour: Array = BRONZE) -> Array:
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


def floor_tile(size: int = TILE_SIZE, colour: Array = GRAY_90) -> Array:
    floor = jnp.ones((size - 2, size - 2), dtype=jnp.int32)
    floor = jnp.pad(floor, 1, "constant", constant_values=0)
    return colorise_tile(floor, colour, background=GRAY_50)


def wall_tile(size: int = TILE_SIZE, colour: Array = GRAY_50) -> Array:
    wall = jnp.ones((size, size), dtype=jnp.int32)
    return colorise_tile(wall, colour)


def mosaic(grid: Array, tile: Array) -> Array:
    tiled = jnp.tile(tile, (*grid.shape, 1))
    return jnp.asarray(tiled, dtype=jnp.uint8)
