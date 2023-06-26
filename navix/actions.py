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
    direction = rotate(state.players.direction, spin)
    player = state.players.replace(direction=direction)
    return state.replace(players=player)


def _walkable(state: State, position: Array) -> Array:
    # according to the grid
    walkable = jnp.equal(state.grid[tuple(position)], 0)

    # and not occupied by another non-walkable entity
    occupied_keys = positions_equal(position, state.keys.position)
    occupied_doors = positions_equal(position, state.doors.position)
    occupied = jnp.any(jnp.logical_or(occupied_keys, occupied_doors))
    # return: if walkable and not occupied
    return jnp.logical_and(walkable, jnp.logical_not(occupied))


def _move(state: State, direction: Array) -> State:
    new_position = translate(state.players.position, direction)
    can_move = _walkable(state, new_position)
    new_position = jnp.where(can_move, new_position, state.players.position)
    player = state.players.replace(position=new_position)
    return state.replace(players=player)


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
    return _move(state, state.players.direction)


def right(state: State) -> State:
    return _move(state, state.players.direction + 1)


def backward(state: State) -> State:
    return _move(state, state.players.direction + 2)


def left(state: State) -> State:
    return _move(state, state.players.direction + 3)


def pickup(state: State) -> State:
    position_in_front = translate(state.players.position, state.players.direction)

    key_found = positions_equal(position_in_front, state.keys.position)

    # update keys
    positions = jnp.where(key_found, DISCARD_PILE_COORDS, state.keys.position)
    keys = state.keys.replace(position=positions)

    # update player's pocket, if the pocket has something else, we overwrite it
    key = jnp.sum(state.keys.id * key_found, dtype=jnp.int32)
    player = jax.lax.cond(jnp.any(key_found), lambda: state.players.replace(pocket=key), lambda: state.players)

    return state.replace(players=player, keys=keys)


def open(state: State) -> State:
    # get the tile in front of the player
    position_in_front = translate(state.players.position, state.players.direction)

    # check if there is a door in front of the player
    door_found = positions_equal(position_in_front, state.doors.position)

    # and that, if so, either it does not require a key or the player has the key
    requires_key = state.doors.requires != -1
    key_match = state.players.pocket == state.doors.requires
    can_open = door_found & (key_match | ~requires_key )

    # update doors
    # TODO(epignatelli): in the future we want to mark the door as open, instead
    # and have a different rendering for it
    # if the door can be opened, move it to the discard pile
    new_positions = jnp.where(can_open, DISCARD_PILE_COORDS, state.doors.position)
    doors = state.doors.replace(position=new_positions)

    # remove key from player's pocket
    pocket = jnp.asarray(state.players.pocket * jnp.any(can_open), dtype=jnp.int32)
    player = jax.lax.cond(jnp.any(can_open), lambda: state.players.replace(pocket=pocket), lambda: state.players)

    return state.replace(players=player, doors=doors)


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
