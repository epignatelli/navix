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
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from jax import Array

from .components import Consumable, Pickable, State


DIRECTIONS = {0: "east", 1: "south", 2: "west", 3: "north"}


def _rotate(state: State, spin: int) -> State:
    direction = (state.player.direction + spin) % 4
    player = state.player.replace(direction=direction)
    return state.replace(player=player)


def _translate(position: Array, direction: Array) -> Array:
    moves = (
        lambda position: position + jnp.asarray((0, 1)),  # east
        lambda position: position + jnp.asarray((1, 0)),  # south
        lambda position: position + jnp.asarray((0, -1)),  # west
        lambda position: position + jnp.asarray((-1, 0)),  # north
    )
    return jax.lax.switch(direction, moves, position)


def _move(state: State, direction: Array) -> State:
    new_position = _translate(state.player.position, direction)
    can_move = jnp.equal(state.grid[tuple(state.player.position)], 0)
    new_position = jnp.where(can_move, new_position, state.player.position)
    player = state.player.replace(position=new_position)
    return state.replace(player=player)


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
    return _move(state, state.player.direction)


def right(state: State) -> State:
    return _move(state, state.player.direction + 1)


def backward(state: State) -> State:
    return _move(state, state.player.direction + 2)


def left(state: State) -> State:
    return _move(state, state.player.direction + 3)


def pickup(state: State) -> State:
    position_in_front = _translate(state.player.position, state.player.direction)

    def _update(key: Pickable) -> Tuple[Array, Pickable]:
        match = jnp.array_equal(position_in_front, key.position)
        # update player's pocket
        pocket = jnp.where(match, key.id, state.player.pocket)
        # set to (-1, -1) the position of the key that was picked up
        unset_position = jnp.asarray((-1, -1))
        position = jnp.where(match, unset_position, key.position)
        key = key.replace(position=position)
        return pocket, key

    pockets, keys = jax.vmap(_update)(state.keys)
    pocket = jnp.max(pockets, axis=0)
    player = state.player.replace(pocket=pocket)
    return state.replace(player=player, keys=keys)


def open(state: State) -> State:
    position_in_front = _translate(state.player.position, state.player.direction)

    def _update(door: Consumable) -> Tuple[Array, Consumable]:
        match = jnp.array_equal(position_in_front, door.position)
        replacement = jnp.asarray((match - 1) * door.replacement, dtype=jnp.int32)

        # update grid
        grid = jnp.zeros_like(state.grid).at[tuple(door.position)].set(replacement)

        # set to (-1, -1) the position of the door that was opened
        unset_position = jnp.asarray((-1, -1))
        position = jnp.where(match, unset_position, door.position)
        door = door.replace(position=position)
        return grid, door

    grid, doors = jax.vmap(_update)(state.doors)
    # the max makes sure that if there was a wall (-1), and it has been opened (x>0)
    # we get the new value of the grid
    grid = jnp.max(grid, axis=0)
    return state.replace(grid=grid, doors=doors)


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
    7: pickup,
    8: open,
}
