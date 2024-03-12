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
from jax import Array
import jax.numpy as jnp
import jax.tree_util as jtu

from .entities import Entities, Events, State
from .components import DISCARD_PILE_COORDS
from .grid import translate, rotate, positions_equal


class Directions:
    EAST = 0
    SOUTH = 1
    WEST = 2
    NORTH = 3


def _rotate(state: State, spin: int) -> State:
    if Entities.PLAYER not in state.entities:
        return state

    player = state.get_player(idx=0)

    # update player's direction
    direction = rotate(player.direction, spin)

    # update sprite representation
    player = player.replace(direction=direction)

    state = state.set_player(player)

    return state


def _can_walk_there(state: State, position: Array) -> Tuple[Array, Events]:
    # according to the grid
    walkable = jnp.equal(state.grid[tuple(position)], 0)
    events = jax.lax.cond(
        walkable, lambda: state.events, lambda: state.events.record(Entities.WALL)
    )

    for k in state.entities:
        same_position = positions_equal(state.entities[k].position, position)
        events = jax.lax.cond(
            jnp.any(same_position), lambda x: x.record(k), lambda x: x, events
        )
        obstructs = jnp.logical_and(
            jnp.logical_not(state.entities[k].walkable), same_position
        )
        walkable = jnp.logical_and(walkable, jnp.any(jnp.logical_not(obstructs)))
    return jnp.asarray(walkable, dtype=jnp.bool_), events


def _move(state: State, direction: Array) -> State:
    if Entities.PLAYER not in state.entities:
        return state

    player = state.get_player(idx=0)
    new_position = translate(player.position, direction)
    can_move, events = _can_walk_there(state, new_position)
    new_position = jnp.where(can_move, new_position, player.position)
    # update structs
    player = player.replace(position=new_position)
    state = state.set_player(player)
    events = jtu.tree_map(lambda x: jnp.any(x), events)
    return state.replace(events=events)


def noop(state: State) -> State:
    return state


def rotate_cw(state: State) -> State:
    return _rotate(state, 1)


def rotate_ccw(state: State) -> State:
    return _rotate(state, -1)


def forward(state: State) -> State:
    player = state.get_player(idx=0)
    return _move(state, player.direction)


def right(state: State) -> State:
    player = state.get_player(idx=0)
    return _move(state, player.direction + 1)


def backward(state: State) -> State:
    player = state.get_player(idx=0)
    return _move(state, player.direction + 2)


def left(state: State) -> State:
    player = state.get_player(idx=0)
    return _move(state, player.direction + 3)


def pickup(state: State) -> State:
    if Entities.KEY not in state.entities:
        return state

    player = state.get_player(idx=0)
    keys = state.get_keys()

    position_in_front = translate(player.position, player.direction)

    key_found = positions_equal(position_in_front, keys.position)

    # update keys
    positions = jnp.where(key_found, DISCARD_PILE_COORDS, keys.position)
    keys = keys.replace(position=positions)

    # update player's pocket, if the pocket has something else, we overwrite it
    key = jnp.sum(keys.id * key_found, dtype=jnp.int32)
    player = jax.lax.cond(
        jnp.any(key_found), lambda: player.replace(pocket=key), lambda: player
    )

    # update events
    events = jax.lax.cond(
        key_found, lambda: state.events.record(Entities.KEY), lambda: state.events
    )

    state = state.set_player(player)
    state = state.set_keys(keys)
    state = state.set_events(events)
    return state


def open(state: State) -> State:
    """Unlocks and opens an openable object (like a door) if possible"""

    if Entities.DOOR not in state.entities:
        return state

    # get the tile in front of the player
    player = state.get_player(idx=0)
    doors = state.get_doors()

    position_in_front = translate(player.position, player.direction)

    # check if there is a door in front of the player
    door_found = positions_equal(position_in_front, doors.position)

    # and that, if so, either it does not require a key or the player has the key
    locked = doors.requires != -1
    key_match = player.pocket == doors.requires
    can_open = door_found & (key_match | ~locked)

    # update doors if closed and can_open
    do_open = ~doors.open & can_open
    open = jnp.where(do_open, True, doors.open)
    requires = jnp.where(do_open, -1, doors.requires)
    doors = doors.replace(open=open, requires=requires)

    # remove key from player's pocket
    pocket = jnp.asarray(player.pocket * jnp.any(can_open), dtype=jnp.int32)
    player = jax.lax.cond(
        jnp.any(can_open), lambda: player.replace(pocket=pocket), lambda: player
    )

    state = state.set_player(player)
    state = state.set_doors(doors)

    return state


DEFAULT_ACTION_SET = (
    noop,
    rotate_cw,
    rotate_ccw,
    forward,
    right,
    backward,
    left,
    pickup,
    open,
)
