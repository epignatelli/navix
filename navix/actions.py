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
"""The *action* system determines the next state of the environment \
given the current state and an action."""


from __future__ import annotations
from typing import Tuple

import jax
from jax import Array
import jax.numpy as jnp

from .entities import Entities, Player
from .states import EventsManager, State
from .components import DISCARD_PILE_COORDS, Pickable
from .grid import translate, rotate, positions_equal


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


def _can_walk_there(state: State, position: Array) -> Tuple[Array, EventsManager]:
    # according to the grid
    walkable = jnp.equal(state.grid[tuple(position)], 0)
    events = jax.lax.cond(
        walkable,
        lambda: state.events,
        lambda: state.events.record_grid_hit(position),
    )

    for k in state.entities:
        same_position = positions_equal(state.entities[k].position, position)
        events = jax.lax.cond(
            jnp.any(same_position),
            lambda x: x.record_walk_into(state.entities[k], position),
            lambda x: x,
            events,
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
    return state.replace(events=events)


def noop(state: State) -> State:
    """No operation. Does nothing.

    Args:
        state (State): The current state.
    
    Returns:
        State: The same state."""
    return state


def rotate_cw(state: State) -> State:
    """Rotates the player clockwise.
    
    Args:
        state (State): The current state.
    
    Returns:
        State: The new state with the player rotated clockwise."""
    return _rotate(state, 1)


def rotate_ccw(state: State) -> State:
    """Rotates the player counter-clockwise.
    
    Args:
        state (State): The current state.

    Returns:
        State: The new state with the player rotated counter-clockwise."""
    return _rotate(state, -1)


def forward(state: State) -> State:
    """Moves the player forward.
    
    Args:
        state: The current state.

    Returns:
        State: The new state with the player moved forward."""
    player = state.get_player(idx=0)
    return _move(state, player.direction)


def right(state: State) -> State:
    """Steps the player to the right without changing the direction.

    Args:
        state (State): The current state.

    Returns:
        State: The new state with the player moved to the right."""
    player = state.get_player(idx=0)
    return _move(state, player.direction + 1)


def backward(state: State) -> State:
    """Steps the player backward without changing the direction.
        
        Args:
            state (State): The current state.

        Returns:
            State: The new state with the player moved backward."""
    player = state.get_player(idx=0)
    return _move(state, player.direction + 2)


def left(state: State) -> State:
    """Steps the player to the left without changing the direction.

    Args:
        state (State): The current state.
    
    Returns:
        State: The new state with the player moved to the left."""
    player = state.get_player(idx=0)
    return _move(state, player.direction + 3)


def pickup(state: State) -> State:
    """Picks up an item in front of the player and puts it in the pocket.
    Args:
        state (State): The current state.
    Returns:
        State: The new state with the player entity having the item in the pocket."""
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
        jnp.any(key_found),
        lambda: state.events.record_key_pickup(keys, position_in_front),
        lambda: state.events,
    )

    state = state.set_player(player)
    state = state.set_keys(keys)
    state = state.set_events(events)
    return state


def drop(state: State) -> State:
    """Replaces the position in front of the player with the item in the pocket.

    Args:
        state (State): The current state.
    
    Returns:
        State: The new state with the item in the pocket dropped in front of the player."""
    player = state.get_player(idx=0)

    position_in_front = translate(player.position, player.direction)

    has_item = player.pocket != -1
    can_drop, events = _can_walk_there(state, position_in_front)
    can_drop = jnp.logical_and(can_drop, has_item)

    for k in state.entities:
        entity = state.entities[k]
        if isinstance(entity, Pickable):
            cond = jnp.logical_and(can_drop, entity.position == DISCARD_PILE_COORDS)
            position = jnp.where(cond, position_in_front, entity.position)
            entity = entity.replace(position=position)
            state.set_entity(k, entity)
    return state


def toggle(state: State) -> State:
    """Toggles an openable object (like a door) if possible.

    Args:
        state (State): The current state.
    
    Returns:
        State: The new state with the openable object toggled."""
    return open(state)


def open(state: State) -> State:
    """Unlocks and opens an openable object (like a door) if possible.
    
    Args:
        state (State): The current state.
    
    Returns:
        State: The new state with the openable object opened."""
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

    # update events
    events = jax.lax.cond(
        jnp.any(do_open),
        lambda: state.events.record_door_opening(doors, position_in_front),
        lambda: state.events,
    )

    state = state.set_player(player)
    state = state.set_doors(doors)
    state = state.set_events(events)

    return state


def done(state: State) -> State:
    """A placeholder action that does nothing, but is a signal to the environment that the episode is over.
    This action does not terminate the episode, unless the termination function explicitly checks for it (not default).
    
    Args:
        state (State): The current state.
    
    Returns:
        State: The same state."""
    return state


# DEFAULT_ACTION_SET = (
#     rotate_ccw,
#     rotate_cw,
#     forward,
#     pickup,
#     drop,
#     toggle,
#     done
# )
"""Default action set from Minigrid. See
https://github.com/Farama-Foundation/Minigrid/blob/master/minigrid/core/actions.py"""


COMPLETE_ACTION_SET = (
    noop,
    rotate_cw,
    rotate_ccw,
    forward,
    right,
    backward,
    left,
    pickup,
    open,
    done,
)
"""Complete action set for the environment.
This set includes all the actions that can be taken by the agent, and does not mirror the Minigrid action set."""

MINIGRID_ACTION_SET = (
    rotate_ccw,
    rotate_cw,
    forward,
    pickup,
    drop,
    toggle,
    done,
)
"""Default action set from Minigrid. See
https://github.com/Farama-Foundation/Minigrid/blob/master/minigrid/core/actions.py"""

DEFAULT_ACTION_SET = MINIGRID_ACTION_SET
