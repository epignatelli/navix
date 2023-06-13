from __future__ import annotations


import jax
from jax.random import KeyArray
import jax.numpy as jnp
from .environment import Environment
from ..components import State, Component, StepType, Timestep
from ..grid import room, spawn_entity


class Room(Environment):
    def reset(self, key: KeyArray) -> Timestep:
        key, k1, k2 = jax.random.split(key, 3)

        # entities
        player = Component(id=1, direction=0)
        goal = Component(id=2)
        entities = {
            "player/0": player,
            "goal/0": goal,
        }

        # system
        grid = room(self.width, self.height)
        grid = spawn_entity(grid, player.id, k1)
        grid = spawn_entity(grid, goal.id, k2)
        state = State(key=key, grid=grid, entities=entities)

        return Timestep(
            t=jnp.asarray(0, dtype=jnp.int32),
            observation=self.observation(state),
            action=jnp.asarray(0, dtype=jnp.int32),
            reward=jnp.asarray(0.0, dtype=jnp.float32),
            step_type=jnp.asarray(0, dtype=jnp.int32),
            state=state,
        )
