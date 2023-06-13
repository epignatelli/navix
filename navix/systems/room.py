from __future__ import annotations


import jax
from jax.random import KeyArray
from .environment import Environment
from ..components import State, Component
from ..grid import room, spawn_entity


class Room(Environment):
    def _reset(self, key: KeyArray) -> State:
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

        return State(key=key, grid=grid, entities=entities)