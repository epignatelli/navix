from __future__ import annotations


import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from chex import Array

from .components import State
from .grid import mask_entity


def navigation(state: State, prob: ArrayLike = 1.0) -> Array:
    player_mask = mask_entity(state.grid, state.entities["player/0"].id)
    goal_mask = mask_entity(state.grid, state.entities["goal/0"].id)
    condition = jax.random.uniform(state.key, ()) >= prob
    return jax.lax.cond(
        condition,
        lambda _: jnp.sum(player_mask * goal_mask),
        lambda _: jnp.asarray(0.0),
        (),
    )


def free(state: State) -> Array:
    return jnp.asarray(0.0)
