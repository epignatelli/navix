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

from collections import deque

import jax
import jax.numpy as jnp
import numpy as np

import navix as nx
from navix.components import EMPTY_POCKET_ID
from navix.entities import Entities
from navix.states import EventType


EAST, SOUTH, WEST, NORTH = 0, 1, 2, 3

# navix.actions.DEFAULT_ACTION_SET = (rotate_ccw, rotate_cw, forward, pickup, drop, toggle, done)
ROTATE_CW, FORWARD, PICKUP, DROP, TOGGLE = 1, 2, 3, 4, 5

N_SEEDS = 10

# The two gameplay tests below drive real (uncompiled) env.step() calls -
# ~15-20 per seed - which is slow on GPU (per-op dispatch overhead adds
# up: ~4.5min for 10 seeds, confirmed on berlin). Restricted to the
# specific seeds the module docstring documents as having exposed each
# of the four navigation edge cases (0, 2, 8), rather than the full
# N_SEEDS range, to keep full-suite runs reasonable while still covering
# every edge case by name by at least one seed.
GAMEPLAY_SEEDS = (0, 2, 8)


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


def face_via_step(env, timestep, direction: int):
    """Like `face`, but through real `env.step()` calls instead of
    `navix.actions` directly - exercises `Environment._step`'s full
    wiring (autoreset check, event reset, reward_fn/termination_fn),
    not just the bare action transform."""
    for _ in range(4):
        if int(timestep.state.get_player().direction) == direction:
            return timestep
        timestep = env.step(timestep, jnp.asarray(ROTATE_CW))
    raise AssertionError(f"could not face direction {direction}")


def walk_via_step(env, timestep, direction: int, steps: int):
    timestep = face_via_step(env, timestep, direction)
    for _ in range(steps):
        timestep = env.step(timestep, jnp.asarray(FORWARD))
    return timestep


def walk_to_via_step(env, timestep, row: int, col: int):
    player = timestep.state.get_player()
    dr = row - int(player.position[0])
    if dr > 0:
        timestep = walk_via_step(env, timestep, SOUTH, dr)
    elif dr < 0:
        timestep = walk_via_step(env, timestep, NORTH, -dr)
    player = timestep.state.get_player()
    dc = col - int(player.position[1])
    if dc > 0:
        timestep = walk_via_step(env, timestep, EAST, dc)
    elif dc < 0:
        timestep = walk_via_step(env, timestep, WEST, -dc)
    return timestep


def walk_to_via_step_avoiding(env, timestep, row: int, col: int, avoid_col: int, room_left: int, room_right: int):
    player = timestep.state.get_player()
    if int(player.position[1]) == avoid_col and int(player.position[0]) != row:
        sidestep_col = avoid_col + 1 if avoid_col + 1 < room_right else avoid_col - 1
        assert room_left <= sidestep_col < room_right
        timestep = walk_to_via_step(env, timestep, int(player.position[0]), sidestep_col)
    return walk_to_via_step(env, timestep, row, col)


def navigate_adjacent_and_face_via_step(
    env, timestep, target_row: int, target_col: int, room_left: int, room_right: int, room_top: int, room_bottom: int
):
    """`navigate_adjacent_and_face`, driven through `env.step()`."""
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

    player = timestep.state.get_player()
    prow, pcol = int(player.position[0]), int(player.position[1])
    for row, col, direction in candidates:
        if row == prow and col == pcol:
            return face_via_step(env, timestep, direction)

    row, col, direction = candidates[0]
    timestep = walk_to_via_step_avoiding(env, timestep, row, col, target_col, room_left, room_right)
    return face_via_step(env, timestep, direction)


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
    for seed in GAMEPLAY_SEEDS:
        env = nx.make("Navix-Unlock-v0")
        timestep = env.reset(jax.random.PRNGKey(seed))
        room_top, room_bottom, door_row, door_col, key_row, key_col = read_layout(
            timestep.state, env
        )

        timestep = navigate_adjacent_and_face_via_step(
            env, timestep, key_row, key_col, 1, door_col, room_top, room_bottom
        )
        assert not timestep.state.events.happened((Entities.KEY, EventType.PICKUP))
        timestep = env.step(timestep, jnp.asarray(PICKUP))
        assert int(timestep.state.get_player().pocket) != int(EMPTY_POCKET_ID), (
            f"seed={seed}: key not picked up"
        )
        assert timestep.state.events.happened((Entities.KEY, EventType.PICKUP))

        timestep = walk_to_via_step(env, timestep, door_row, key_col)
        timestep = walk_to_via_step(env, timestep, door_row, door_col - 1)
        timestep = face_via_step(env, timestep, EAST)
        assert timestep.step_type == 0, f"seed={seed}: episode ended before opening the door"
        assert not timestep.state.events.happened((Entities.DOOR, EventType.OPEN))

        prev_state = timestep.state
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
        assert bool(
            nx.events.on_door_open(prev_state, jnp.asarray(TOGGLE), timestep.state)
        )


def test_unlock_pickup_gameplay_and_reward_termination():
    # real env.step() cycle throughout (see test_unlock_gameplay_and_
    # reward_termination's comment) - this used to call navix.actions
    # directly and check terminations.on_box_pickup/rewards.on_box_pickup
    # against hand-built states, which never actually exercised
    # Environment._step's reward_fn/termination_fn wiring for the
    # registered Navix-UnlockPickup-v0 env (PR #186 review).
    for seed in GAMEPLAY_SEEDS:
        env = nx.make("Navix-UnlockPickup-v0")
        timestep = env.reset(jax.random.PRNGKey(seed))
        state = timestep.state
        room_top, room_bottom, door_row, door_col, key_row, key_col = read_layout(state, env)
        boxes = state.get_boxes()
        box_row, box_col = int(boxes.position[0, 0]), int(boxes.position[0, 1])

        timestep = navigate_adjacent_and_face_via_step(
            env, timestep, key_row, key_col, 1, door_col, room_top, room_bottom
        )
        timestep = env.step(timestep, jnp.asarray(PICKUP))
        assert int(timestep.state.get_player().pocket) != int(EMPTY_POCKET_ID), (
            f"seed={seed}: key not picked up"
        )

        timestep = walk_to_via_step(env, timestep, door_row, key_col)
        timestep = walk_to_via_step(env, timestep, door_row, door_col - 1)
        timestep = face_via_step(env, timestep, EAST)
        timestep = env.step(timestep, jnp.asarray(TOGGLE))
        assert bool(timestep.state.get_doors().open[0]), f"seed={seed}: door not opened"
        assert not timestep.state.events.happened((Entities.BOX, EventType.PICKUP))
        assert timestep.step_type == 0, f"seed={seed}: episode ended before picking up the box"

        # cross through the door before navigating within the second
        # room - see module docstring for why a direct beeline can't
        # reach anything past the door's own wall column.
        timestep = walk_to_via_step(env, timestep, door_row, door_col)
        if not (box_row == door_row and box_col == door_col + 1):
            timestep = walk_to_via_step(env, timestep, door_row, door_col + 1)
        timestep = navigate_adjacent_and_face_via_step(
            env, timestep, box_row, box_col, door_col, env.width - 1, room_top, room_bottom
        )
        state_before_pickup = timestep.state
        timestep = env.step(timestep, jnp.asarray(PICKUP))
        player = timestep.state.get_player()
        assert int(player.pocket) == int(boxes.id[0]), f"seed={seed}: box not picked up"
        assert not state_before_pickup.events.happened((Entities.BOX, EventType.PICKUP))
        assert timestep.state.events.happened((Entities.BOX, EventType.PICKUP))
        assert bool(
            nx.events.on_box_pickup(state_before_pickup, jnp.asarray(PICKUP), timestep.state)
        )
        assert timestep.step_type == 2, (
            f"seed={seed}: episode should terminate on picking up the box, "
            f"got step_type={timestep.step_type}"
        )
        assert float(timestep.reward) > 0, (
            f"seed={seed}: expected positive reward for picking up the box, got {timestep.reward}"
        )

        # rewards.on_box_pickup/terminations.on_box_pickup, across the
        # real (prev_state, action, state) transition that env.step()
        # just applied.
        action = jnp.asarray(PICKUP)
        assert bool(nx.terminations.on_box_pickup(state_before_pickup, action, timestep.state))
        assert float(nx.rewards.on_box_pickup(state_before_pickup, action, timestep.state)) > 0
        # and the reverse: on the state *before* the pickup, neither
        # should report success yet.
        assert not bool(
            nx.terminations.on_box_pickup(state_before_pickup, action, state_before_pickup)
        )


def test_unlock_and_unlockpickup_jit_vmap_compatible():
    """Environment.reset/step must work under jax.jit and jax.vmap,
    matching the convention test_room/test_keydoor established elsewhere
    in tests/test_environments.py - the gameplay tests above call
    env.step() eagerly (needed to make navigation decisions from
    intermediate positions), so they never actually exercise this."""
    for env_id in ("Navix-Unlock-v0", "Navix-UnlockPickup-v0"):
        env = nx.make(env_id)
        keys = jax.random.split(jax.random.PRNGKey(0), 4)

        reset = jax.jit(env.reset)
        step = jax.jit(env.step)

        timestep = jax.vmap(reset)(keys)
        for action in range(len(nx.actions.DEFAULT_ACTION_SET)):
            timestep = jax.vmap(step, in_axes=(0, None))(timestep, jnp.asarray(action))
        jax.block_until_ready(timestep)


def test_blocked_unlock_pickup_structure():
    env = nx.make("Navix-BlockedUnlockPickup-v0")
    for seed in range(N_SEEDS):
        state = env.reset(jax.random.PRNGKey(seed)).state
        assert Entities.BALL in state.entities
        assert Entities.BOX in state.entities

        balls = state.get_balls()
        doors = state.get_doors()
        door_row, door_col = int(doors.position[0, 0]), int(doors.position[0, 1])
        assert int(balls.position[0, 0]) == door_row, "ball must be on the door's row"
        assert int(balls.position[0, 1]) == door_col - 1, (
            "ball must be directly in front of the door, on the first room's side"
        )

        player = state.get_player()
        keys = state.get_keys()
        assert not jnp.array_equal(player.position, balls.position[0]), (
            f"seed={seed}: player overlaps the blocking ball"
        )
        assert not jnp.array_equal(keys.position[0], balls.position[0]), (
            f"seed={seed}: key overlaps the blocking ball"
        )


def bfs_blocked_mask(state) -> np.ndarray:
    """Like the BFS helpers in test_go_to_object.py/test_fetch.py/
    test_put_near.py (a `(height, width)` boolean array of cells the
    player can't step onto) - separate from this file's own `walk_to`/
    `navigate_adjacent_and_face` heuristics, which were tuned for
    Unlock/UnlockPickup's exact 2-obstacle (Key, Box) layout and don't
    know about BlockedUnlockPickup's extra, position-varying `Ball`.
    Recomputed fresh whenever the door might have opened, since `Door.
    walkable` depends on its own `open` state, unlike Key/Box/Ball."""
    blocked = np.asarray(state.grid) == -1
    for entity_enum, entity in state.entities.items():
        if entity_enum == Entities.PLAYER:
            continue
        walkable = np.asarray(entity.walkable)
        positions = np.asarray(entity.position)
        if positions.ndim == 1:
            positions = positions[None]
            walkable = walkable.reshape(1)
        for (row, col), w in zip(positions, walkable):
            if not w:
                blocked[row, col] = True
    return blocked


def bfs_path(blocked: np.ndarray, start: tuple, goal: tuple):
    height, width = blocked.shape
    prev = {start: None}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        if current == goal:
            break
        row, col = current
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nxt = (row + dr, col + dc)
            if (
                0 <= nxt[0] < height
                and 0 <= nxt[1] < width
                and not blocked[nxt]
                and nxt not in prev
            ):
                prev[nxt] = current
                queue.append(nxt)
    if goal not in prev:
        return None
    path = [goal]
    while path[-1] != start:
        path.append(prev[path[-1]])
    path.reverse()
    return path


def bfs_direction_between(a: tuple, b: tuple) -> int:
    dr, dc = b[0] - a[0], b[1] - a[1]
    if dr == 1:
        return SOUTH
    if dr == -1:
        return NORTH
    if dc == 1:
        return EAST
    if dc == -1:
        return WEST
    raise AssertionError(f"{a} -> {b} is not a single orthogonal step")


def bfs_walk_path_via_step(env, timestep, path):
    for a, b in zip(path[:-1], path[1:]):
        timestep = face_via_step(env, timestep, bfs_direction_between(a, b))
        timestep = env.step(timestep, jnp.asarray(FORWARD))
    return timestep


def bfs_navigate_adjacent_and_face_via_step(env, timestep, target_row: int, target_col: int):
    state = timestep.state
    blocked = bfs_blocked_mask(state)
    start = (int(state.get_player().position[0]), int(state.get_player().position[1]))
    target = (target_row, target_col)

    for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        candidate = (target[0] + dr, target[1] + dc)
        if not (0 <= candidate[0] < blocked.shape[0] and 0 <= candidate[1] < blocked.shape[1]):
            continue
        if blocked[candidate]:
            continue
        if candidate == start:
            return face_via_step(env, timestep, bfs_direction_between(candidate, target))
        path = bfs_path(blocked, start, candidate)
        if path is not None:
            timestep = bfs_walk_path_via_step(env, timestep, path)
            return face_via_step(env, timestep, bfs_direction_between(candidate, target))
    raise AssertionError(f"no reachable approach cell for target {target}")


def test_blocked_unlock_pickup_gameplay_and_reward_termination():
    # real env.step() cycle throughout, matching test_unlock_gameplay_
    # and_reward_termination's convention. Uses the BFS helpers above,
    # not this file's walk_to/navigate_adjacent_and_face heuristics -
    # those were tuned for exactly two static obstacles (Key, Box) and
    # don't know about the extra Ball, which (once dropped somewhere
    # after clearing it) could sit anywhere and confuse a blind
    # row-then-column walk (confirmed directly: seed 0 failed this way
    # with the heuristic navigation).
    #
    # Solve order matters: pickup() overwrites whatever's already in
    # the player's pocket, so the ball must be cleared *and dropped*
    # (freeing the pocket again) before the key can be picked up -
    # picking up the key first, then the ball, would silently lose the
    # key (still at the discard pile, but no longer referenced by
    # player.pocket).
    for seed in GAMEPLAY_SEEDS:
        env = nx.make("Navix-BlockedUnlockPickup-v0")
        timestep = env.reset(jax.random.PRNGKey(seed))
        state = timestep.state
        doors = state.get_doors()
        door_row, door_col = int(doors.position[0, 0]), int(doors.position[0, 1])
        keys = state.get_keys()
        key_row, key_col = int(keys.position[0, 0]), int(keys.position[0, 1])
        balls = state.get_balls()
        ball_row, ball_col = int(balls.position[0, 0]), int(balls.position[0, 1])
        boxes = state.get_boxes()
        box_row, box_col = int(boxes.position[0, 0]), int(boxes.position[0, 1])

        # clear the blocking ball: pick it up (moves it straight to the
        # discard pile - the cell is clear immediately)...
        timestep = bfs_navigate_adjacent_and_face_via_step(env, timestep, ball_row, ball_col)
        timestep = env.step(timestep, jnp.asarray(PICKUP))
        assert int(timestep.state.get_player().pocket) != int(EMPTY_POCKET_ID), (
            f"seed={seed}: ball not picked up"
        )
        # ...then step into the now-vacated cell and drop facing further
        # into the room (not back at the door's own interaction cell,
        # which dropping in place would immediately re-block).
        timestep = env.step(timestep, jnp.asarray(FORWARD))
        timestep = face_via_step(env, timestep, WEST)
        timestep = env.step(timestep, jnp.asarray(DROP))
        assert int(timestep.state.get_player().pocket) == int(EMPTY_POCKET_ID), (
            f"seed={seed}: pocket not freed after dropping the ball"
        )

        # now proceed like UnlockPickup: key -> unlock -> cross -> box
        timestep = bfs_navigate_adjacent_and_face_via_step(env, timestep, key_row, key_col)
        timestep = env.step(timestep, jnp.asarray(PICKUP))
        assert int(timestep.state.get_player().pocket) != int(EMPTY_POCKET_ID), (
            f"seed={seed}: key not picked up"
        )

        timestep = bfs_navigate_adjacent_and_face_via_step(env, timestep, door_row, door_col)
        timestep = env.step(timestep, jnp.asarray(TOGGLE))
        assert bool(timestep.state.get_doors().open[0]), f"seed={seed}: door not opened"
        assert timestep.step_type == 0, f"seed={seed}: episode ended before picking up the box"

        timestep = bfs_navigate_adjacent_and_face_via_step(env, timestep, box_row, box_col)
        timestep = env.step(timestep, jnp.asarray(PICKUP))
        assert int(timestep.state.get_player().pocket) == int(boxes.id[0]), (
            f"seed={seed}: box not picked up"
        )
        assert timestep.step_type == 2, (
            f"seed={seed}: expected termination on picking up the box"
        )
        assert float(timestep.reward) > 0, f"seed={seed}: expected positive reward"


def test_blocked_unlock_pickup_jit_vmap_compatible():
    env = nx.make("Navix-BlockedUnlockPickup-v0")
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    timestep = jax.vmap(reset)(keys)
    for action in range(len(nx.actions.DEFAULT_ACTION_SET)):
        timestep = jax.vmap(step, in_axes=(0, None))(timestep, jnp.asarray(action))
    jax.block_until_ready(timestep)
