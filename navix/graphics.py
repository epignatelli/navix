from jax import Array
import jax.numpy as jnp


TILE_SIZE = 32


def triangle_east(size: int = TILE_SIZE) -> Array:
    triangle = jnp.ones((size, size), dtype=jnp.int32)
    triangle = jnp.tril(triangle, k=0)
    triangle = jnp.flip(triangle, axis=0)
    triangle = jnp.tril(triangle, k=0)
    triangle = jnp.roll(triangle, (0, size // 3))
    return triangle


def triangle_south(size: int = TILE_SIZE) -> Array:
    triangle = triangle_east(size)
    return jnp.rot90(triangle, k=3)


def triangle_west(size: int = TILE_SIZE) -> Array:
    triangle = triangle_east(size)
    return jnp.rot90(triangle, k=2)


def triangle_north(size: int = TILE_SIZE) -> Array:
    triangle = triangle_east(size)
    return jnp.rot90(triangle, k=1)


def diamond(size: int = TILE_SIZE) -> Array:
    diamond = jnp.ones((size, size), dtype=jnp.int32)
    diamond = jnp.tril(diamond, k=size // 2.5)
    diamond = jnp.flip(diamond, axis=0)
    diamond = jnp.tril(diamond, k=size // 2.5)
    diamond = jnp.flip(diamond, axis=1)
    diamond = jnp.tril(diamond, k=size // 2.5)
    diamond = jnp.flip(diamond, axis=0)
    diamond = jnp.tril(diamond, k=size // 2.5)
    return diamond


def door(size: int = TILE_SIZE) -> Array:
    frame_size = TILE_SIZE - TILE_SIZE // 20
    door = jnp.zeros((frame_size, frame_size), dtype=jnp.int32)
    door = jnp.pad(door, 1, "constant", constant_values=1)
    door = jnp.pad(door, 1, "constant", constant_values=0)
    door = jnp.pad(door, 1, "constant", constant_values=1)
    x_0 = TILE_SIZE - TILE_SIZE // 5
    x_1 = x_0 + TILE_SIZE // 5
    y_0 = TILE_SIZE // 2 - TILE_SIZE - TILE_SIZE // 5
    y_1 = y_0 + TILE_SIZE // 5
    door = door.at[y_0: y_1, x_0: x_1].set(1)
    return door


def key(size: int = TILE_SIZE) -> Array:
    key = jnp.zeros((size, size), dtype=jnp.int32)

    # Handle (Round Part)
    handle_radius = size // 4
    handle_center = (size // 2, size // 4)
    y, x = jnp.ogrid[:size, :size]
    mask = (x - handle_center[0]) ** 2 + (y - handle_center[1]) ** 2 <= handle_radius ** 2
    key = jnp.where(mask, 1, key)

    # Shaft (Straight Part)
    shaft_width = size // 8
    shaft_height = size // 2
    shaft_start = (size // 2 - shaft_width // 2, size // 2 - shaft_height // 2)
    shaft_end = (size // 2 + shaft_width // 2, size // 2 + shaft_height // 2)
    shaft_mask = jnp.logical_and(jnp.logical_and(x >= shaft_start[0], x <= shaft_end[0]),
                                 jnp.logical_and(y >= shaft_start[1], y <= shaft_end[1]))
    key = jnp.where(shaft_mask, 1, key)

    # Tooth (Pointy End)
    tooth_width = size // 15
    tooth_height = size // 2
    tooth_position = (size // 2 - tooth_width // 2, size - tooth_height)
    tooth_mask = jnp.logical_and(jnp.logical_and(x >= tooth_position[0], x <= tooth_position[0] + tooth_width),
                                 jnp.logical_and(y >= tooth_position[1], y <= tooth_position[1] + tooth_height))
    key = jnp.where(tooth_mask, 1, key)

    return key