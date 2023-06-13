from __future__ import annotations
from chex import Array


import jax.numpy as jnp
from .components import State, StepType
from .grid import mask_entity


def check_truncation(terminated: Array, truncated: Array) -> Array:
    return jnp.asarray(truncated + 2 * terminated, dtype=jnp.int32)


def on_navigation_completion(state: State) -> Array:
    player_mask = mask_entity(state.grid, state.entities["player/0"].id)
    goal_mask = mask_entity(state.grid, state.entities["goal/0"].id)
    return jnp.array_equal(player_mask, goal_mask)
