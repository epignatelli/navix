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

from enum import IntEnum
from typing import Any, Dict, List, Tuple

import jax
from jax import Array
from flax import struct
from jax.random import KeyArray
import jax.numpy as jnp
from jax.typing import ArrayLike

from .grid import remove_entity


class Component(struct.PyTreeNode):
    id: int
    requires_update: bool = False
    can_act: bool = False
    can_pickup: bool = False
    can_consume: bool = False
    is_goal: bool = False

    def update(self, state: State, **kwargs) -> State:
        return state


class Player(Component):
    """Players are entities that can act around the environment"""

    can_act: bool = True
    direction: int = 0
    pocket: List[int] = struct.field(default_factory=list)

    def update(self, state: State, action: Array, actions_set) -> State:
        return jax.lax.switch(action, actions_set.values(), state)


class Goal(Component):
    """Goals are entities that can be reached by the player"""

    is_goal: bool = True

    def update(self, state: State) -> State:
        return state


class Pickable(Component):
    """Pickable items are world objects that can be picked up by the player.
    Examples of pickable items are keys, coins, etc."""

    can_pickup: bool = True
    picked: bool = False

    def update(self, state: State) -> State:
        # add the item to the player's pocket
        pocket = state.player.pocket + [self.id]
        player = state.player.replace(pocket=pocket)

        # remove the item from the grid
        grid = remove_entity(state.grid, self.id)

        # remove the item from the state.entities collection
        # state.entities.pop(self.id)  # do not pop or the number of entities will change
        # instead, we can replace the entity with a new one
        entity = self.replace(picked=True, requires_update=False)
        state.entities[self.id] = entity
        return state.replace(grid=grid, player=player, entitites=state.entities)


class Consumable(Component):
    """Consumable items are world objects that can be consumed by the player.
    Consuming an item requires a tool (e.g. a key to open a door).
    A tool is an id (int) of another item, specified in the `requires` field (-1 if no tool is required).
    After an item is consumed, it is both removed from the `state.entities` collection, and replaced in the grid
    by the item specified in the `replacement` field (0 = floor by default).
    Examples of consumables are doors (to open) food (to eat) and water (to drink), etc.
    """

    can_consume: bool = True
    consumed: bool = False
    requires: int = -1
    replacement: int = 0

    def update(self, state: State, tool: int = -1) -> State:
        tool_not_required = jnp.greater_equal(self.requires, 0)
        can_replace = jnp.array_equal(self.requires, tool)
        can_replace = jnp.logical_or(tool_not_required, can_replace)
        replacement = jnp.asarray(can_replace * self.replacement, dtype=jnp.int32)
        grid = jnp.where(state.grid == self.id, replacement, state.grid)

        state.entities[self.id] = self.replace(consumed=True, requires_update=False)
        return state.replace(grid=grid, entities=state.entities)


class State(struct.PyTreeNode):
    """The Markovian state of the environment"""

    key: KeyArray
    """The random number generator state"""
    grid: Array
    """The 2D-grid containing the ids of the entities in each position"""
    player: Player  # we can potentially extend this to multiple players easily
    """The player entity"""
    # TODO(epignatelli): ideally we would like an entity to have
    # a list of components (e.g. a key can be a pickable AND consumable, where
    # Pickable updates the player's pocket and Consumable updates the grid)
    entities: Dict[ArrayLike, Component] = struct.field(
        pytree_node=False, default_factory=dict
    )
    """The entities in the environment"""


def update_state(state, action: Array, actions_set) -> State:
    # update the player entity first
    state = state.player.update(state, action, actions_set)

    # TODO(epignatelli): we can't really work with lax.scan here
    # because entities are different and we cannot transpose the pytree
    # def body_fun(carry, x):
    #     state = carry
    #     entity = x
    #     state = jax.lax.cond(
    #         entity.requires_update,
    #         lambda state: entity.update(state),
    #         lambda state: state,
    #         state
    #     )
    #     return state, ()

    # entities = tuple(state.entities.values())
    # state, _ = jax.lax.scan(body_fun, state, entities)

    # but we can do this instead, assuming that the number
    # of registered entities does not change during traning
    # and that it is small enough (~<15) to not unroll
    # into a long set of instructions
    for _, entity in state.entities.items():
        state = jax.lax.cond(
            entity.requires_update,
            lambda state: entity.update(state),
            lambda state: state,
            state,
        )
    return state


class StepType(IntEnum):
    TRANSITION = 0
    """discount > 0, episode continues"""
    TRUNCATION = 1
    """discount > 0, episode ends"""
    TERMINATION = 2
    """discount == 0, episode ends"""


class Timestep(struct.PyTreeNode):
    t: Array
    observation: Array
    action: Array
    reward: Array
    step_type: Array
    state: State
    info: Dict[str, Any] = struct.field(default_factory=dict)
