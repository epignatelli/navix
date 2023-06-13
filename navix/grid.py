from __future__ import annotations


from typing import Tuple
import jax
import jax.numpy as jnp
from jax.random import KeyArray
from jax.typing import ArrayLike
from chex import Array


def room(width: int, height: int):
    grid = jnp.zeros((width, height), dtype=jnp.int32)
    return jnp.pad(grid, 1, mode="constant", constant_values=-1)

def coordinates_to_idx(grid: Array, coordinates: ArrayLike):
    return coordinates[0] * grid.shape[0] + coordinates[1]

def idx_to_coordinates(grid: Array, idx: int):
    return jnp.stack(jnp.divmod(idx, grid.shape[0]))

def coordinate_to_mask(grid: Array, coordinates: ArrayLike) -> Array:
    raise NotImplementedError()

def entity_coordinates(grid: Array, entity_id: int) -> Tuple[int, int]:
    idx = mask_entity(grid, entity_id).reshape(-1,)
    idx = jnp.asarray(idx, dtype=jnp.int32)
    coordinates = tuple(idx_to_coordinates(grid, idx))
    return coordinates

def mask_entity(grid: Array, entity_id: int) -> Array:
    return jnp.asarray(grid == entity_id, dtype=jnp.float32)

def place_entity(grid: Array, entity_id: int, coordinates: ArrayLike) -> Array:
    return grid.at[coordinates].set(entity_id)

def spawn_entity(grid: Array, entity_id: int, key: KeyArray) -> Array:
    assert entity_id > 0, f"Reserved id {entity_id}, please specify an id > 0"
    mask = mask_entity(grid, 0)
    mask_coordinates = mask.reshape(-1,)
    idx = jax.random.categorical(key, jnp.log(mask.reshape(-1,)))
    coordinates = tuple(idx_to_coordinates(grid, idx))
    grid = place_entity(grid, entity_id, coordinates)
    return grid

def remove_entity(grid: Array, entity_id: int, replacement: int = 0) -> Array:
    mask = mask_entity(grid, entity_id)
    return jnp.where(mask, 0, grid)