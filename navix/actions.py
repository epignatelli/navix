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
"""The action primitives an integer action indexes into.

Each function here is a pure `State -> State` transformation of the
world. An `Environment` holds an ordered `action_set` (a tuple of these),
and `transitions_fn` applies `action_set[a]` for an integer action `a`;
the environment's `action_space` is `Discrete(len(action_set))`. So the
*meaning* of action `2` depends on which set the environment uses - see
`MINIGRID_ACTION_SET` (the default) and `COMPLETE_ACTION_SET`.

Conventions shared by all of them:

- "the player" is `state.get_player(idx=0)` (navix is single-agent for
  now); "in front" is the cell one step along `player.direction`
  (`0` east, `1` south, `2` west, `3` north).
- Movement that would enter a wall or a non-walkable entity is a no-op:
  the player stays put (and a wall/entity-hit event is recorded).
- Every action is total - it always returns a `State` of the same
  structure, so the set can go through `jax.lax.switch`.
"""


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
    """Does nothing - the world is returned unchanged. Present so an
    action set can include an explicit "wait".

    Args:
        state (State): the current state.

    Returns:
        State: `state`, unchanged."""
    return state


def rotate_cw(state: State) -> State:
    """Turns the player 90 degrees clockwise (`direction` -> `direction + 1`
    mod 4). Position is unchanged.

    Args:
        state (State): the current state.

    Returns:
        State: the state with the player's `direction` updated."""
    return _rotate(state, 1)


def rotate_ccw(state: State) -> State:
    """Turns the player 90 degrees counter-clockwise (`direction` ->
    `direction - 1` mod 4). Position is unchanged.

    Args:
        state (State): the current state.

    Returns:
        State: the state with the player's `direction` updated."""
    return _rotate(state, -1)


def forward(state: State) -> State:
    """Moves the player one cell in the direction it faces. No-op if that
    cell is a wall or a non-walkable entity (a hit event is recorded);
    `direction` never changes.

    Args:
        state (State): the current state.

    Returns:
        State: the state with the player's `position` updated (or not, if
        blocked)."""
    player = state.get_player(idx=0)
    return _move(state, player.direction)


def right(state: State) -> State:
    """Strafes the player one cell to its right (90 degrees clockwise of
    where it faces) *without* turning. Blocked-move rules as for
    `forward`. Not in the MiniGrid action set.

    Args:
        state (State): the current state.

    Returns:
        State: the state with the player's `position` updated (or not, if
        blocked)."""
    player = state.get_player(idx=0)
    return _move(state, player.direction + 1)


def backward(state: State) -> State:
    """Steps the player one cell backwards (opposite to where it faces)
    without turning. Blocked-move rules as for `forward`. Not in the
    MiniGrid action set.

    Args:
        state (State): the current state.

    Returns:
        State: the state with the player's `position` updated (or not, if
        blocked)."""
    player = state.get_player(idx=0)
    return _move(state, player.direction + 2)


def left(state: State) -> State:
    """Strafes the player one cell to its left (90 degrees
    counter-clockwise of where it faces) without turning. Blocked-move
    rules as for `forward`. Not in the MiniGrid action set.

    Args:
        state (State): the current state.

    Returns:
        State: the state with the player's `position` updated (or not, if
        blocked)."""
    player = state.get_player(idx=0)
    return _move(state, player.direction + 3)


def pickup(state: State) -> State:
    """Picks up the pickable entity (`Key`, `Box`, or `Ball`) directly in
    front of the player: the entity is moved off the grid and its `id` is
    written to `player.pocket`, overwriting whatever was there. No-op if
    the cell in front holds nothing pickable. Records a pickup event.

    Args:
        state (State): the current state.

    Returns:
        State: the state with the item removed from the grid and its
        `id` in `player.pocket`."""

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
    """Places the item currently in `player.pocket` on the cell in front
    of the player and empties the pocket. No-op if the pocket is empty or
    the cell in front is not walkable (occupied or a wall).

    Args:
        state (State): the current state.

    Returns:
        State: the state with the pocketed item back on the grid in front
        of the player and `player.pocket` cleared."""
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
    """MiniGrid's `toggle` action. An alias for `open`: in navix a door,
    once opened, stays open (there is no close), so "toggle" and "open"
    are the same operation.

    Args:
        state (State): the current state.

    Returns:
        State: see `open`."""
    return open(state)


def open_box(state: State, position_in_front: Array) -> State:
    """Opens the `Box` in front of the player, if any (verified against
    MiniGrid's actual `Box.toggle`: `env.grid.set(pos, self.contains)`
    - the box is removed and, if it held something, that object takes
    its place at the same position). Currently only `Key` boxes are
    supported (`Box.pocket` stores the id of a `Key` instance, the same
    way `player.pocket` does) - `ObstructedMaze`'s only real user of
    this, keys hidden inside boxes.

    Args:
        state (State): The current state.
        position_in_front (Array): The position directly in front of
            the player.

    Returns:
        State: The new state, with any opened box removed and its
        contents (if any) revealed at its former position."""
    boxes = state.get_boxes()
    opened = positions_equal(position_in_front, boxes.position)

    if Entities.KEY in state.entities:
        keys = state.get_keys()
        has_key = boxes.pocket != EMPTY_POCKET_ID
        revealing = opened & has_key
        revealed_key_id = jnp.sum(jnp.where(revealing, boxes.pocket, 0))
        reveal = jnp.any(revealing) & (keys.id == revealed_key_id)
        keys = keys.replace(
            position=jnp.where(reveal[:, None], position_in_front, keys.position)
        )
        state = state.set_keys(keys)

    boxes = boxes.replace(
        position=jnp.where(opened[:, None], DISCARD_PILE_COORDS, boxes.position)
    )
    return state.set_boxes(boxes)


def open(state: State) -> State:
    """Opens whatever openable thing is directly in front of the player:

    - a `Box` -> removed, and its contents (a `Key`) revealed at that
      cell (see `open_box`);
    - a `Door` -> opened iff it is closed and either unlocked
      (`requires == -1`) or the player is carrying the required key
      (`player.pocket == door.requires`), in which case that key is
      consumed from the pocket. Opening records a door-opening event.

    A `Door` that is already open, and any other cell, are left
    untouched. `toggle` is an alias for this.

    Args:
        state (State): the current state.

    Returns:
        State: the state with the door/box in front opened if the
        conditions held."""
    player = state.get_player(idx=0)
    position_in_front = translate(player.position, player.direction)

    if Entities.BOX in state.entities:
        state = open_box(state, position_in_front)

    if Entities.DOOR not in state.entities:
        return state

    doors = state.get_doors()

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
    """The agent's "I'm finished" signal. Returns the state unchanged - it
    does *not* end the episode by itself. Only environments whose
    `termination_fn` inspects for it (e.g. `GoToObject`, via
    `events.on_target_done`) react to it; under the default termination
    it is a no-op.

    Args:
        state (State): the current state.

    Returns:
        State: `state`, unchanged."""
    return state


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
"""Every action navix defines, in index order:
`0 noop`, `1 rotate_cw`, `2 rotate_ccw`, `3 forward`, `4 right`,
`5 backward`, `6 left`, `7 pickup`, `8 open`, `9 done`. Wider than
MiniGrid's set (adds strafing and `noop`); use it for environments that
need lateral movement."""

MINIGRID_ACTION_SET = (
    rotate_ccw,
    rotate_cw,
    forward,
    pickup,
    drop,
    toggle,
    done,
)
"""MiniGrid's seven actions, in the same index order MiniGrid uses:
`0 rotate_ccw` (MiniGrid "left"), `1 rotate_cw` ("right"), `2 forward`,
`3 pickup`, `4 drop`, `5 toggle`, `6 done`. See
https://github.com/Farama-Foundation/Minigrid/blob/master/minigrid/core/actions.py"""

DEFAULT_ACTION_SET = MINIGRID_ACTION_SET
"""The `action_set` an `Environment` uses unless overridden -
`MINIGRID_ACTION_SET`."""
