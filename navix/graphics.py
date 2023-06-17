import jax
import jax.numpy as jnp
from jax import Array


TILE_SIZE = 32
RED = jnp.asarray([255, 0, 0, 255], dtype=jnp.uint8)
GREEN = jnp.asarray([0, 255, 0, 255], dtype=jnp.uint8)
BLUE = jnp.asarray([0, 0, 255, 255], dtype=jnp.uint8)
BLACK = jnp.asarray([0, 0, 0, 255], dtype=jnp.uint8)
WHITE = jnp.asarray([255, 255, 255, 255], dtype=jnp.uint8)
YELLOW = jnp.asarray([255, 255, 0, 255], dtype=jnp.uint8)
PURPLE = jnp.asarray([255, 0, 255, 255], dtype=jnp.uint8)
CYAN = jnp.asarray([0, 255, 255, 255], dtype=jnp.uint8)
ORANGE = jnp.asarray([255, 128, 0, 255], dtype=jnp.uint8)
PINK = jnp.asarray([255, 0, 128, 255], dtype=jnp.uint8)
BROWN = jnp.asarray([128, 64, 0, 255], dtype=jnp.uint8)
GRAY = jnp.asarray([128, 128, 128, 255], dtype=jnp.uint8)
LIGHT_GRAY = jnp.asarray([192, 192, 192, 255], dtype=jnp.uint8)
DARK_GRAY = jnp.asarray([64, 64, 64, 255], dtype=jnp.uint8)
TRANSPARENT = jnp.asarray([0, 0, 0, 0], dtype=jnp.uint8)
GOLD = jnp.asarray([255, 215, 0, 255], dtype=jnp.uint8)
SILVER = jnp.asarray([192, 192, 192, 255], dtype=jnp.uint8)
BRONZE = jnp.asarray([205, 127, 50, 255], dtype=jnp.uint8)
MAROON = jnp.asarray([128, 0, 0, 255], dtype=jnp.uint8)
NAVY = jnp.asarray([0, 0, 128, 255], dtype=jnp.uint8)
TEAL = jnp.asarray([0, 128, 128, 255], dtype=jnp.uint8)
OLIVE = jnp.asarray([128, 128, 0, 255], dtype=jnp.uint8)
LIME = jnp.asarray([0, 255, 0, 255], dtype=jnp.uint8)
AQUA = jnp.asarray([0, 255, 255, 255], dtype=jnp.uint8)
FUCHSIA = jnp.asarray([255, 0, 255, 255], dtype=jnp.uint8)
SALMON = jnp.asarray([250, 128, 114, 255], dtype=jnp.uint8)
TURQUOISE = jnp.asarray([64, 224, 208, 255], dtype=jnp.uint8)
VIOLET = jnp.asarray([238, 130, 238, 255], dtype=jnp.uint8)
INDIGO = jnp.asarray([75, 0, 130, 255], dtype=jnp.uint8)
BEIGE = jnp.asarray([245, 245, 220, 255], dtype=jnp.uint8)
MINT = jnp.asarray([189, 252, 201, 255], dtype=jnp.uint8)
LAVENDER = jnp.asarray([230, 230, 250, 255], dtype=jnp.uint8)
APRICOT = jnp.asarray([251, 206, 177, 255], dtype=jnp.uint8)
MAUVE = jnp.asarray([224, 176, 255, 255], dtype=jnp.uint8)
LILAC = jnp.asarray([200, 162, 200, 255], dtype=jnp.uint8)
TAN = jnp.asarray([210, 180, 140, 255], dtype=jnp.uint8)


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
        GRAY,
        LIGHT_GRAY,
        DARK_GRAY,
        TRANSPARENT,
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
    grid = jnp.zeros((size * len(colours), size * len(colours), 4), dtype=jnp.uint8)
    for i, colour in enumerate(colours):
        for j, colour in enumerate(colours):
            grid = grid.at[
                i * size : (i + 1) * size, j * size : (j + 1) * size
            ].set(colour)
    return grid


def colorise_tile(tile: Array, colour: Array) -> Array:
    tile = jnp.stack([tile] * 4, axis=-1)
    tile = jnp.where(tile, colour, tile)
    return tile


def rectangle_tile(size: int = TILE_SIZE, colour: Array = DARK_GRAY) -> Array:
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


def door_tile(size: int = TILE_SIZE, colour: Array = APRICOT) -> Array:
    frame_size = TILE_SIZE - TILE_SIZE // 20
    door = jnp.zeros((frame_size, frame_size), dtype=jnp.int32)
    door = jnp.pad(door, 1, "constant", constant_values=1)
    door = jnp.pad(door, 1, "constant", constant_values=0)
    door = jnp.pad(door, 1, "constant", constant_values=1)
    x_0 = TILE_SIZE - TILE_SIZE // 5
    x_1 = x_0 + TILE_SIZE // 5
    y_0 = TILE_SIZE // 2 - TILE_SIZE - TILE_SIZE // 5
    y_1 = y_0 + TILE_SIZE // 5
    door = door.at[y_0:y_1, x_0:x_1].set(1)
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


def mosaic(grid: Array, tile: Array) -> Array:
    return jnp.tile(tile, (*grid.shape, 1))
