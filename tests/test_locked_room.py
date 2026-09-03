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

"""`LockedRoom` (issue #179): structural checks against the live reset
state (six rooms, one locked, a key hidden in some other room, distinct
door colours, everyone on real floor), plus a full real-gameplay
walkthrough: fetch the key, unlock the one locked door, reach the goal."""

from __future__ import annotations

from collections import deque

import jax
import jax.numpy as jnp
import numpy as np

import navix as nx
from navix.entities import Entities
from navix.environments import locked_room


EAST, SOUTH, WEST, NORTH = 0, 1, 2, 3
ROTATE_CW, FORWARD, PICKUP, TOGGLE = 1, 2, 3, 5

N_SEEDS = 3
GAMEPLAY_SEEDS = 2

ENV_ID = "Navix-LockedRoom-v0"


def face_via_step(step_fn, timestep, direction: int):
    for _ in range(4):
        if int(timestep.state.get_player().direction) == direction:
            return timestep
        timestep = step_fn(timestep, jnp.asarray(ROTATE_CW))
    raise AssertionError(f"could not face direction {direction}")


def blocked_mask(state) -> np.ndarray:
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


def direction_between(a: tuple, b: tuple) -> int:
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


def walk_path_via_step(step_fn, timestep, path):
    for a, b in zip(path[:-1], path[1:]):
        timestep = face_via_step(step_fn, timestep, direction_between(a, b))
        timestep = step_fn(timestep, jnp.asarray(FORWARD))
    return timestep


def navigate_adjacent_and_face_via_step(step_fn, timestep, target_row: int, target_col: int):
    """Paths (BFS, 4-connected) to an orthogonal neighbour of `target`
    and faces it - same routine `test_multi_room.py` uses, duplicated
    rather than imported (each test file's own convention here)."""
    state = timestep.state
    blocked = blocked_mask(state)
    start = (int(state.get_player().position[0]), int(state.get_player().position[1]))
    target = (target_row, target_col)
    if start == target:
        return timestep
    blocked_for_path = blocked.copy()
    blocked_for_path[target] = True
    for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        candidate = (target[0] + dr, target[1] + dc)
        if not (0 <= candidate[0] < blocked.shape[0] and 0 <= candidate[1] < blocked.shape[1]):
            continue
        if blocked[candidate]:
            continue
        if candidate == start:
            return face_via_step(step_fn, timestep, direction_between(candidate, target))
        path = bfs_path(blocked_for_path, start, candidate)
        if path is not None:
            timestep = walk_path_via_step(step_fn, timestep, path)
            return face_via_step(step_fn, timestep, direction_between(candidate, target))
    raise AssertionError(f"no reachable approach cell for target {target}")


def room_index_of(position, tops, sizes) -> int:
    row, col = int(position[0]), int(position[1])
    for i, (top, size) in enumerate(zip(np.asarray(tops), np.asarray(sizes))):
        if top[0] + 1 <= row <= top[0] + size[0] - 2 and top[1] + 1 <= col <= top[1] + size[1] - 2:
            return i
    raise AssertionError(f"position {(row, col)} is not inside any of the 6 rooms")


def assert_valid_locked_room_state(state, where: str):
    tops, sizes, doors_pos, _left_wall, _right_wall = locked_room.room_layout(19, 19)
    grid = np.asarray(state.grid)
    assert grid.shape == (19, 19), f"{where}: grid must be 19x19"

    doors = state.get_doors()
    assert doors.position.shape[0] == locked_room.NUM_ROOMS, where
    assert not bool(np.asarray(doors.open).any()), f"{where}: doors must start closed"
    requires = np.asarray(doors.requires)
    locked_mask = requires != -1
    assert int(locked_mask.sum()) == 1, f"{where}: expected exactly one locked door"
    assert len(set(np.asarray(doors.colour).tolist())) == locked_room.NUM_ROOMS, (
        f"{where}: expected {locked_room.NUM_ROOMS} distinct door colours"
    )
    for row, col in np.asarray(doors.position):
        assert int(grid[row, col]) == 0, f"{where}: door at ({row},{col}) not floor"

    locked_idx = int(np.flatnonzero(locked_mask)[0])
    goal = state.get_goals()
    goal_room = room_index_of(goal.position[0], tops, sizes)
    assert goal_room == locked_idx, (
        f"{where}: goal must sit inside the locked room ({locked_idx}), found in room {goal_room}"
    )
    assert int(grid[goal.position[0, 0], goal.position[0, 1]]) == 0, f"{where}: goal not on floor"

    keys = state.get_keys()
    key_room = room_index_of(keys.position[0], tops, sizes)
    assert key_room != locked_idx, f"{where}: key must not be hidden inside the locked room itself"
    assert int(keys.id[0]) == int(requires[locked_idx]), (
        f"{where}: key id must match the locked door's own `requires`"
    )
    assert int(keys.colour[0]) == int(doors.colour[locked_idx]), (
        f"{where}: key colour must match the locked room's door colour"
    )
    assert int(grid[keys.position[0, 0], keys.position[0, 1]]) == 0, f"{where}: key not on floor"

    player = state.get_player()
    assert int(grid[player.position[0], player.position[1]]) == 0, f"{where}: player not on floor"
    left_wall, right_wall = _left_wall, _right_wall
    assert left_wall < int(player.position[1]) < right_wall, (
        f"{where}: player must start in the hallway, got col={int(player.position[1])}"
    )


def test_locked_room_structure():
    env = nx.make(ENV_ID)
    for seed in range(N_SEEDS):
        state = env.reset(jax.random.PRNGKey(seed)).state
        assert_valid_locked_room_state(state, f"{ENV_ID} seed={seed}")


def test_locked_room_gameplay_and_reward_termination():
    env = nx.make(ENV_ID)
    step_fn = jax.jit(env.step)
    for seed in range(GAMEPLAY_SEEDS):
        timestep = env.reset(jax.random.PRNGKey(seed))
        state = timestep.state

        # every door starts closed - even the 5 unlocked ones - so the
        # key's own room must be opened first, or it's unreachable
        tops, sizes, _doors_pos, _lw, _rw = locked_room.room_layout(19, 19)
        key_pos = state.get_keys().position[0]
        key_room = room_index_of(key_pos, tops, sizes)
        key_door_row, key_door_col = (int(x) for x in state.get_doors().position[key_room])
        timestep = navigate_adjacent_and_face_via_step(step_fn, timestep, key_door_row, key_door_col)
        timestep = step_fn(timestep, jnp.asarray(TOGGLE))
        assert bool(np.asarray(timestep.state.get_doors().open)[key_room]), (
            f"seed={seed}: the key room's own (unlocked) door should open on toggle, no key needed"
        )

        key_row, key_col = int(key_pos[0]), int(key_pos[1])
        timestep = navigate_adjacent_and_face_via_step(step_fn, timestep, key_row, key_col)
        timestep = step_fn(timestep, jnp.asarray(PICKUP))
        assert int(timestep.state.get_player().pocket) != -1, (
            f"seed={seed}: player should be holding the key after pickup"
        )

        doors = timestep.state.get_doors()
        locked_idx = int(np.flatnonzero(np.asarray(doors.requires) != -1)[0])
        door_row, door_col = (int(x) for x in doors.position[locked_idx])
        timestep = navigate_adjacent_and_face_via_step(step_fn, timestep, door_row, door_col)
        timestep = step_fn(timestep, jnp.asarray(TOGGLE))
        assert bool(np.asarray(timestep.state.get_doors().open)[locked_idx]), (
            f"seed={seed}: locked door should be open after toggling it with the matching key"
        )
        assert timestep.step_type == 0, f"seed={seed}: episode ended before reaching the goal"

        goal = timestep.state.get_goals()
        goal_row, goal_col = int(goal.position[0, 0]), int(goal.position[0, 1])
        timestep = navigate_adjacent_and_face_via_step(step_fn, timestep, goal_row, goal_col)
        timestep = step_fn(timestep, jnp.asarray(FORWARD))
        assert timestep.step_type == 2, f"seed={seed}: expected termination on reaching the goal"
        assert float(timestep.reward) > 0, f"seed={seed}: expected positive reward"


def test_locked_room_jit_vmap_compatible():
    env = nx.make(ENV_ID)
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    timestep = jax.vmap(reset)(keys)
    for action in range(7):
        timestep = jax.vmap(step, in_axes=(0, None))(timestep, jnp.asarray(action))
