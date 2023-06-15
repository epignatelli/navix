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
"""Contains the update functions for the Player component, which
NAVIX treats as a special entity as its update depends on an external action input.
The update of all the other components is described in the `navix.transitions module"""


from __future__ import annotations

import jax
import jax.numpy as jnp

from .components import State, Component
from .grid import mask_entity, remove_entity


DIRECTIONS = {0: "east", 1: "south", 2: "west", 3: "north"}


def _rotate(state: State, spin: int) -> State:
    direction = (state.player.direction + spin) % 3
    player = state.player.replace(direction=direction)
    return state.replace(player=player)


def _move(state: State, entity_id: int, direction: int) -> State:
    moves = (
        lambda x: jnp.roll(x, 1, axis=1),  # east
        lambda x: jnp.roll(x, 1, axis=0),  # south
        lambda x: jnp.roll(x, -1, axis=1),  # west
        lambda x: jnp.roll(x, -1, axis=0),  # north
    )
    mask = mask_entity(state.grid, entity_id)
    new_mask = jax.lax.switch(direction, moves, mask)
    walkable = mask_entity(state.grid, 0)
    legal = jnp.sum(new_mask * walkable) > 0

    # hitting obstacles lead to no-ops
    grid = jax.lax.cond(
        legal,
        lambda grid: jnp.where(new_mask, entity_id, remove_entity(grid, entity_id)),
        lambda grid: grid,
        state.grid,
    )
    return state.replace(grid=grid)


def undefined(state: State) -> State:
    # this is problematic because jax.lax.switch evaluates
    # all *python* branches (no XLA computation is performed)
    # even though only one is selected
    # one option is the following, but this breaks type checking
    # def raise_error(state: State) -> State:
    #     raise ValueError("Undefined action")
    # jax.debug.callback(raise_error)
    raise ValueError("Undefined action")


def noop(state: State) -> State:
    return state


def rotate_cw(state: State) -> State:
    return _rotate(state, 1)


def rotate_ccw(state: State) -> State:
    return _rotate(state, -1)


def forward(state: State) -> State:
    entity_id = state.player.id
    direction = state.player.direction
    return _move(state, entity_id, direction)


def right(state: State) -> State:
    entity_id = state.player.id
    direction = state.player.direction + 1
    return _move(state, entity_id, direction)


def backward(state: State) -> State:
    entity_id = state.player.id
    direction = state.player.direction + 2
    return _move(state, entity_id, direction)


def left(state: State) -> State:
    entity_id = state.player.id
    direction = state.player.direction + 3
    return _move(state, entity_id, direction)


def pickup(state: State) -> State:
    in_front = mask_entity(forward(state).grid, state.player.id)
    id_in_front = jnp.sum(in_front * state.grid, dtype=jnp.int32)

    can_pickup = jnp.isin(id_in_front, jnp.asarray(list(state.entities.keys())))
    entity = state.entities[id_in_front]
    can_pickup = jnp.logical_and(can_pickup, entity.can_pickup)

    state.entities.update({id_in_front: entity.replace(requires_update=can_pickup)})
    return state


def consume(state: State) -> State:
    in_front = mask_entity(forward(state).grid, state.player.id)
    id_in_front = jnp.sum(in_front * state.grid, dtype=jnp.int32)

    can_consume = jnp.isin(id_in_front, jnp.asarray(list(state.entities.keys())))
    entity = state.entities[id_in_front]
    can_consume = jnp.logical_and(can_consume, entity.can_consume)

    state.entities.update({id_in_front: entity.replace(requires_update=can_consume)})
    return state


# TODO(epignatelli): a mutable dictionary here is dangerous
ACTIONS = {
    # -1: undefined,
    0: noop,
    1: rotate_cw,
    2: rotate_ccw,
    3: forward,
    4: right,
    5: backward,
    6: left,
    # 7: pickup,
    # 8: consume,
}
