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

from .entities import Entities, Player, Box
from .states import EventsManager, State
from .components import DISCARD_PILE_COORDS, EMPTY_POCKET_ID, Pickable
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
        walkable = jnp.logical_and(walkable, jnp.all(jnp.logical_not(obstructs)))
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
    """Picks up an item (`Key`, `Box`, or `Ball`) in front of the player
    and puts it in the pocket.
    Args:
        state (State): The current state.
    Returns:
        State: The new state with the player entity having the item in the pocket."""

    def pickup_entity(state: State, entity, setter) -> State:
        """Shared logic for one pickable entity type (`Key`/`Box`/
        `Ball`): if the player is facing an instance of `entity`,
        discard it and put its `id` in the player's pocket. A closure
        (not a module-level function) specifically so it stays out of
        `actions.py`'s public surface, where every other name is a real
        `State -> State` action - unlike those, this needs `entity`/
        `setter` too, to stay parametrized over which pickable entity
        type `pickup`'s call sites are handling.

        Args:
            state (State): The current state.
            entity: Every instance of this entity type in the
                environment (`state.get_keys()`/`state.get_boxes()`/
                `state.get_balls()`).
            setter (Callable[[Any], State]): `state.set_keys`/
                `state.set_boxes`/`state.set_balls` - writes the updated
                entity batch back.

        Returns:
            State: The new state with the item picked up, if the player
            was facing one."""
        player = state.get_player(idx=0)
        position_in_front = translate(player.position, player.direction)

        found = positions_equal(position_in_front, entity.position)

        # update events - before entity is moved to the discard pile
        # below, so the recorded event keeps the item's real pickup
        # position, not DISCARD_PILE_COORDS. record_pickup already
        # dispatches on isinstance(entity, Key/Box/Ball) internally, so
        # this stays correct for any Pickable type without needing a
        # per-type branch here.
        events = jax.lax.cond(
            jnp.any(found),
            lambda: state.events.record_pickup(entity, position_in_front),
            lambda: state.events,
        )

        # discard the picked-up instance - found[:, None], not found:
        # found has shape (n_instances,), and entity.position/
        # DISCARD_PILE_COORDS both end in a (2,) row/col axis, so an
        # unreshaped `found` broadcasts against the wrong axis whenever
        # n_instances == 2 (JAX aligns trailing dims: (2,) vs (n,2)
        # matches (2,) against the row/col axis, not the instance axis) -
        # confirmed directly: with 2 balls, picking up one corrupted
        # both balls' rows to the discard row while leaving both
        # columns untouched, instead of moving only the picked one to
        # DISCARD_PILE_COORDS entirely (see PR #191 review).
        positions = jnp.where(found[:, None], DISCARD_PILE_COORDS, entity.position)
        entity = entity.replace(position=positions)

        # update player's pocket, if the pocket has something else, we overwrite it
        picked_id = jnp.sum(entity.id * found, dtype=jnp.int32)
        player = jax.lax.cond(
            jnp.any(found), lambda: player.replace(pocket=picked_id), lambda: player
        )

        state = state.set_player(player)
        state = setter(entity)
        state = state.set_events(events)
        return state

    if Entities.KEY in state.entities:
        state = pickup_entity(state, state.get_keys(), state.set_keys)
    if Entities.BOX in state.entities:
        state = pickup_entity(state, state.get_boxes(), state.set_boxes)
    if Entities.BALL in state.entities:
        state = pickup_entity(state, state.get_balls(), state.set_balls)
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
            # match by id == player.pocket, not just "is this instance
            # sitting at the discard pile" - the latter alone moves
            # *every* already-consumed Pickable instance, not just the
            # one actually in the player's pocket, once two Pickable
            # entity types can coexist in one episode (see #188).
            matches_pocket = entity.id == player.pocket
            at_discard = jnp.all(entity.position == DISCARD_PILE_COORDS, axis=-1)
            cond = can_drop & matches_pocket & at_discard
            position = jnp.where(cond[:, None], position_in_front, entity.position)
            entity = entity.replace(position=position)
            state = state.set_entity(k, entity)

    # the player's pocket must go back to empty on a successful drop -
    # previously left stale (still holding the dropped item's id) after
    # every drop, which is wrong regardless of the bug above.
    player = jax.lax.cond(
        can_drop, lambda: player.replace(pocket=EMPTY_POCKET_ID), lambda: player
    )
    state = state.set_player(player)
    # _can_walk_there's events (e.g. a grid-hit record when the drop
    # destination isn't walkable) were computed above but never applied
    # to state - also fixed here while touching this function.
    state = state.set_events(events)
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
    is_open = jnp.asarray(doors.open, dtype=jnp.bool_)
    locked = doors.requires != -1
    key_match = player.pocket == doors.requires
    can_open = door_found & (key_match | ~locked)

    # update doors if closed and can_open. Note: `is_open` is only used for
    # the boolean predicate above - the write-back uses `doors.open` itself,
    # not `is_open`, so `doors.open`'s original dtype (int or bool,
    # depending on which environment constructed the door) is preserved via
    # jnp.where's weak-type promotion. Writing back `is_open` instead would
    # silently force every door to bool, which breaks jax.lax.switch's
    # requirement that every action branch produce identical output dtypes.
    do_open = ~is_open & can_open
    open = jnp.where(do_open, True, doors.open)
    requires = jnp.where(do_open, -1, doors.requires)
    doors = doors.replace(open=open, requires=requires)

    # remove key from player's pocket, but only when this action actually
    # unlocked a previously-closed, locked door with a matching key - not
    # merely because the door in front happened to already be open (some
    # environments, e.g. KeyCorridor, construct a door that is already open
    # while still marked locked; `do_open` is False there, so the key is
    # correctly left untouched)
    unlocked = do_open & locked & key_match
    pocket = jnp.where(jnp.any(unlocked), EMPTY_POCKET_ID, player.pocket)
    player = jax.lax.cond(
        jnp.any(unlocked), lambda: player.replace(pocket=pocket), lambda: player
    )

    # update events
    events = jax.lax.cond(
        jnp.any(do_open),
        lambda: state.events.record_door_opening(doors, do_open),
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
