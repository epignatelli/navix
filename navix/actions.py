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

import jax
import jax.numpy as jnp
from jax import Array

from .entities import Door, Key, State
from .components import DISCARD_PILE_COORDS
from .grid import translate, rotate, positions_equal


DIRECTIONS = {0: "east", 1: "south", 2: "west", 3: "north"}


def _rotate(state: State, spin: int) -> State:
    if "player" not in state.entities:
        return state

    player = state.get_player()
    direction = rotate(player.direction, spin)
    player = player.replace(direction=direction)
    state.entities["player"] = player[None]
    return state


def _walkable(state: State, position: Array) -> Array:
    # according to the grid
    walkable_cause_grid = jnp.equal(state.grid[tuple(position)], 0)
    walkable_cause_entity = jnp.all(jnp.asarray([state.entities[k].walkable for k in state.entities]))
    return jnp.logical_and(walkable_cause_entity, walkable_cause_grid)


def _move(state: State, direction: Array) -> State:
    if "player" not in state.entities:
        return state

    player = state.get_player()
    new_position = translate(player.position, direction)
    can_move = _walkable(state, new_position)
    new_position = jnp.where(can_move, new_position, player.position)
    player = player.replace(position=new_position)
    state.entities["player"] = player[None]
    return state


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
    player = state.get_player()
    return _move(state, player.direction)


def right(state: State) -> State:
    player = state.get_player()
    return _move(state, player.direction + 1)


def backward(state: State) -> State:
    player = state.get_player()
    return _move(state, player.direction + 2)


def left(state: State) -> State:
    player = state.get_player()
    return _move(state, player.direction + 3)


def pickup(state: State) -> State:

    if "key" not in state.entities:
        return state

    player = state.get_player()
    keys = state.get_keys()

    position_in_front = translate(player.position, player.direction)

    key_found = positions_equal(position_in_front, keys.position)

    # update keys
    positions = jnp.where(key_found, DISCARD_PILE_COORDS, keys.position)
    keys = keys.replace(position=positions)

    # update player's pocket, if the pocket has something else, we overwrite it
    key = jnp.sum(keys.id * key_found, dtype=jnp.int32)
    player = jax.lax.cond(jnp.any(key_found), lambda: player.replace(pocket=key), lambda: player)

    state.entities["player"] = player[None]
    state.entities["key"] = keys
    return state


def open(state: State) -> State:
    """Unlocks and opens an openable object (like a door) if possible"""

    if "door" not in state.entities:
        return state

    # get the tile in front of the player
    player = state.get_player()
    doors = state.get_doors()

    position_in_front = translate(player.position, player.direction)

    # check if there is a door in front of the player
    door_found = positions_equal(position_in_front, doors.position)

    # and that, if so, either it does not require a key or the player has the key
    requires_key = doors.requires != -1
    key_match = player.pocket == doors.requires
    can_open = door_found & (key_match | ~requires_key )

    # update doors if closed and can_open
    do_open = (~doors.open & can_open)
    open = jnp.where(do_open, True, doors.open)
    doors = doors.replace(open=open)

    # remove key from player's pocket
    pocket = jnp.asarray(player.pocket * jnp.any(can_open), dtype=jnp.int32)
    player = jax.lax.cond(jnp.any(can_open), lambda: player.replace(pocket=pocket), lambda: player)

    state.entities["player"] = player[None]
    state.entities["door"] = doors

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
    7: pickup,
    8: open,
}
