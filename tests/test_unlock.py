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

"""`Unlock`/`UnlockPickup` (issues #175/#176): structural checks against
the live reset state (door/key/box placement, matching id/colour), plus
a full hand-crafted gameplay walkthrough via real `env.step()` calls,
verified across several seeds since the door row and every
key/player/box position are randomised (`navix/environments/
unlock.py`'s `two_equal_rooms_with_door`).

The navigation helpers below (`walk_to`, `navigate_adjacent_and_face`)
are more careful than a naive row-then-column walk because the room
layout has two real obstacles a naive walk can collide with:

- The `Key`/`Box` being picked up is itself non-walkable - a straight
  vertical walk at the *same column* the item sits on, crossing its
  row, gets stuck against it (confirmed directly: seed 8's key landed
  in the same column the player started in).
- The two rooms are connected only through the door's own row - any
  other row of the shared wall column is a real wall, so reaching
  anything in the second room needs an explicit "cross through the
  door" step first, not a direct beeline from wherever the player was
  in the first room (confirmed directly: seed 0's box needed a route
  through the door's row, not a straight line from the key's position).
- The box can end up placed immediately at the door's exit, in which
  case the player is already adjacent-and-facing it the moment the
  door opens, with no further movement needed (confirmed directly:
  seed 2)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

import navix as nx
from navix.components import EMPTY_POCKET_ID
from navix.entities import Entities
from navix.states import EventType


EAST, SOUTH, WEST, NORTH = 0, 1, 2, 3

N_SEEDS = 10


def face(state, direction: int):
    """Rotates (0-3 times) until the player faces `direction`."""
    for _ in range(4):
        if int(state.get_player().direction) == direction:
            return state
        state = nx.actions.rotate_cw(state)
    raise AssertionError(f"could not face direction {direction}")


def walk(state, direction: int, steps: int):
    """Faces `direction`, then calls `actions.forward` `steps` times."""
    state = face(state, direction)
    for _ in range(steps):
        state = nx.actions.forward(state)
    return state


def walk_to(state, row: int, col: int):
    """Moves the player onto `(row, col)`, aligning row then column."""
    player = state.get_player()
    dr = row - int(player.position[0])
    if dr > 0:
        state = walk(state, SOUTH, dr)
    elif dr < 0:
        state = walk(state, NORTH, -dr)
    player = state.get_player()
    dc = col - int(player.position[1])
    if dc > 0:
        state = walk(state, EAST, dc)
    elif dc < 0:
        state = walk(state, WEST, -dc)
    return state


def walk_to_avoiding(state, row: int, col: int, avoid_col: int, room_left: int, room_right: int):
    """Like `walk_to`, but sidesteps to an adjacent column first if the
    player's current column already equals `avoid_col` and the target
    row differs - `walk_to`'s row phase keeps the player's *original*
    column fixed while moving vertically, so if that column already
    holds a non-walkable obstacle (e.g. an un-picked-up `Key`/`Box`),
    the vertical walk would get stuck crossing its row."""
    player = state.get_player()
    if int(player.position[1]) == avoid_col and int(player.position[0]) != row:
        sidestep_col = avoid_col + 1 if avoid_col + 1 < room_right else avoid_col - 1
        assert room_left <= sidestep_col < room_right
        state = walk_to(state, int(player.position[0]), sidestep_col)
    return walk_to(state, row, col)


def navigate_adjacent_and_face(
    state, target_row: int, target_col: int, room_left: int, room_right: int, room_top: int, room_bottom: int
):
    """Gets the player adjacent to `(target_row, target_col)` and facing
    it, within the room bounded by `[room_left, room_right)` x
    `[room_top, room_bottom]` (inclusive) - tries every cardinal
    neighbour of the target that's in-bounds, preferring one the player
    is already standing on (e.g. right at a doorway the target happens
    to sit just past) over navigating."""
    candidates = []
    if target_row - 1 >= room_top:
        candidates.append((target_row - 1, target_col, SOUTH))
    if target_row + 1 <= room_bottom:
        candidates.append((target_row + 1, target_col, NORTH))
    if target_col - 1 >= room_left:
        candidates.append((target_row, target_col - 1, EAST))
    if target_col + 1 < room_right:
        candidates.append((target_row, target_col + 1, WEST))
    assert candidates, f"no valid approach cell for ({target_row}, {target_col})"

    player = state.get_player()
    prow, pcol = int(player.position[0]), int(player.position[1])
    for row, col, direction in candidates:
        if row == prow and col == pcol:
            return face(state, direction)

    row, col, direction = candidates[0]
    state = walk_to_avoiding(state, row, col, target_col, room_left, room_right)
    return face(state, direction)


def read_layout(state, env):
    room_top, room_bottom = 1, env.height - 2
    doors = state.get_doors()
    door_row, door_col = int(doors.position[0, 0]), int(doors.position[0, 1])
    keys = state.get_keys()
    key_row, key_col = int(keys.position[0, 0]), int(keys.position[0, 1])
    return room_top, room_bottom, door_row, door_col, key_row, key_col


def test_unlock_structure():
    env = nx.make("Navix-Unlock-v0")
    for seed in range(N_SEEDS):
        timestep = env.reset(jax.random.PRNGKey(seed))
        state = timestep.state
        player = state.get_player()
        doors = state.get_doors()
        keys = state.get_keys()

        assert Entities.BOX not in state.entities
        assert int(player.position[1]) < env.height - 1, f"seed={seed}: player not in left room"
        assert int(keys.position[0, 1]) < env.height - 1, f"seed={seed}: key not in left room"
        assert int(doors.position[0, 1]) == env.height - 1, f"seed={seed}: door not on shared wall column"
        assert bool(doors.requires[0] != -1), f"seed={seed}: door should start locked"
        assert not bool(doors.open[0]), f"seed={seed}: door should start closed"
        assert int(keys.id[0]) == int(doors.requires[0]), f"seed={seed}: key id must match door requires"
        assert int(keys.colour[0]) == int(doors.colour[0]), f"seed={seed}: key colour must match door colour"
        assert not jnp.array_equal(player.position, keys.position[0]), f"seed={seed}: key overlaps player start"


def test_unlock_pickup_structure():
    env = nx.make("Navix-UnlockPickup-v0")
    for seed in range(N_SEEDS):
        timestep = env.reset(jax.random.PRNGKey(seed))
        state = timestep.state
        assert Entities.BOX in state.entities
        boxes = state.get_boxes()
        assert int(boxes.position[0, 1]) > env.height - 1, f"seed={seed}: box not in right room"


def test_unlock_gameplay_and_reward_termination():
    # real env.step() cycle throughout, to exercise Environment._step's
    # reward_fn/termination_fn wiring (rewards.on_door_open/
    # terminations.on_door_open), not just navix.actions in isolation.
    for seed in range(N_SEEDS):
        env = nx.make("Navix-Unlock-v0")
        timestep = env.reset(jax.random.PRNGKey(seed))
        room_top, room_bottom, door_row, door_col, key_row, key_col = read_layout(
            timestep.state, env
        )

        # navix.actions.DEFAULT_ACTION_SET = (rotate_ccw, rotate_cw, forward, pickup, drop, toggle, done)
        ROTATE_CW, FORWARD, PICKUP, TOGGLE = 1, 2, 3, 5

        def act(timestep, action_fn):
            """Applies one navix.actions-style state transform via a
            real env.step() call, matching whichever action index that
            transform corresponds to."""
            index = {nx.actions.rotate_cw: ROTATE_CW, nx.actions.forward: FORWARD}[action_fn]
            return env.step(timestep, jnp.asarray(index))

        def face_via_step(timestep, direction):
            for _ in range(4):
                if int(timestep.state.get_player().direction) == direction:
                    return timestep
                timestep = act(timestep, nx.actions.rotate_cw)
            raise AssertionError

        def walk_via_step(timestep, direction, steps):
            timestep = face_via_step(timestep, direction)
            for _ in range(steps):
                timestep = act(timestep, nx.actions.forward)
            return timestep

        def walk_to_via_step(timestep, row, col):
            player = timestep.state.get_player()
            dr = row - int(player.position[0])
            if dr > 0:
                timestep = walk_via_step(timestep, SOUTH, dr)
            elif dr < 0:
                timestep = walk_via_step(timestep, NORTH, -dr)
            player = timestep.state.get_player()
            dc = col - int(player.position[1])
            if dc > 0:
                timestep = walk_via_step(timestep, EAST, dc)
            elif dc < 0:
                timestep = walk_via_step(timestep, WEST, -dc)
            return timestep

        def walk_to_via_step_avoiding(timestep, row, col, avoid_col, room_left, room_right):
            player = timestep.state.get_player()
            if int(player.position[1]) == avoid_col and int(player.position[0]) != row:
                sidestep_col = avoid_col + 1 if avoid_col + 1 < room_right else avoid_col - 1
                assert room_left <= sidestep_col < room_right
                timestep = walk_to_via_step(timestep, int(player.position[0]), sidestep_col)
            return walk_to_via_step(timestep, row, col)

        def navigate_adjacent_and_face_via_step(
            timestep, target_row, target_col, room_left, room_right, room_top, room_bottom
        ):
            candidates = []
            if target_row - 1 >= room_top:
                candidates.append((target_row - 1, target_col, SOUTH))
            if target_row + 1 <= room_bottom:
                candidates.append((target_row + 1, target_col, NORTH))
            if target_col - 1 >= room_left:
                candidates.append((target_row, target_col - 1, EAST))
            if target_col + 1 < room_right:
                candidates.append((target_row, target_col + 1, WEST))
            assert candidates

            player = timestep.state.get_player()
            prow, pcol = int(player.position[0]), int(player.position[1])
            for row, col, direction in candidates:
                if row == prow and col == pcol:
                    return face_via_step(timestep, direction)

            row, col, direction = candidates[0]
            timestep = walk_to_via_step_avoiding(timestep, row, col, target_col, room_left, room_right)
            return face_via_step(timestep, direction)

        timestep = navigate_adjacent_and_face_via_step(
            timestep, key_row, key_col, 1, door_col, room_top, room_bottom
        )
        assert not timestep.state.events.happened((Entities.KEY, EventType.PICKUP))
        timestep = env.step(timestep, jnp.asarray(PICKUP))
        assert int(timestep.state.get_player().pocket) != int(EMPTY_POCKET_ID), (
            f"seed={seed}: key not picked up"
        )
        assert timestep.state.events.happened((Entities.KEY, EventType.PICKUP))

        timestep = walk_to_via_step(timestep, door_row, key_col)
        timestep = walk_to_via_step(timestep, door_row, door_col - 1)
        timestep = face_via_step(timestep, EAST)
        assert timestep.step_type == 0, f"seed={seed}: episode ended before opening the door"
        assert not timestep.state.events.happened((Entities.DOOR, EventType.OPEN))

        timestep = env.step(timestep, jnp.asarray(TOGGLE))
        assert bool(timestep.state.get_doors().open[0]), f"seed={seed}: door did not open"
        assert timestep.state.events.happened((Entities.DOOR, EventType.OPEN))
        assert timestep.step_type == 2, (
            f"seed={seed}: episode should terminate on opening the door, "
            f"got step_type={timestep.step_type}"
        )
        assert float(timestep.reward) > 0, (
            f"seed={seed}: expected positive reward for opening the door, got {timestep.reward}"
        )
        assert bool(nx.events.on_door_open(timestep.state))


def test_unlock_pickup_gameplay_and_reward_termination():
    for seed in range(N_SEEDS):
        env = nx.make("Navix-UnlockPickup-v0")
        timestep = env.reset(jax.random.PRNGKey(seed))
        state = timestep.state
        room_top, room_bottom, door_row, door_col, key_row, key_col = read_layout(state, env)
        boxes = state.get_boxes()
        box_row, box_col = int(boxes.position[0, 0]), int(boxes.position[0, 1])

        state = navigate_adjacent_and_face(
            state, key_row, key_col, 1, door_col, room_top, room_bottom
        )
        state = nx.actions.pickup(state)
        assert int(state.get_player().pocket) != int(EMPTY_POCKET_ID), f"seed={seed}: key not picked up"

        state = walk_to(state, door_row, key_col)
        state = walk_to(state, door_row, door_col - 1)
        state = face(state, EAST)
        state = nx.actions.open(state)
        assert bool(state.get_doors().open[0]), f"seed={seed}: door not opened"
        assert not state.events.happened((Entities.BOX, EventType.PICKUP))

        # cross through the door before navigating within the second
        # room - see module docstring for why a direct beeline can't
        # reach anything past the door's own wall column.
        state = walk_to(state, door_row, door_col)
        if not (box_row == door_row and box_col == door_col + 1):
            state = walk_to(state, door_row, door_col + 1)
        state = navigate_adjacent_and_face(
            state, box_row, box_col, door_col, env.width - 1, room_top, room_bottom
        )
        state_before_pickup = state
        state = nx.actions.pickup(state)
        player = state.get_player()
        assert int(player.pocket) == int(boxes.id[0]), f"seed={seed}: box not picked up"
        assert not state_before_pickup.events.happened((Entities.BOX, EventType.PICKUP))
        assert state.events.happened((Entities.BOX, EventType.PICKUP))
        assert bool(nx.events.on_box_pickup(state))

        # rewards.on_box_pickup/terminations.on_box_pickup, across the
        # real (prev_state, action, state) transition that just picked
        # up the box.
        action = jnp.asarray(3)  # pickup, per DEFAULT_ACTION_SET
        assert bool(nx.terminations.on_box_pickup(state_before_pickup, action, state))
        assert float(nx.rewards.on_box_pickup(state_before_pickup, action, state)) > 0
        # and the reverse: on the state *before* the pickup, neither
        # should report success yet.
        assert not bool(
            nx.terminations.on_box_pickup(state_before_pickup, action, state_before_pickup)
        )
