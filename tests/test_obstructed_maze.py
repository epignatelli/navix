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

"""`ObstructedMaze1Dlhb` (issue #183, smallest variant only - see that
issue's own staged-rollout recommendation): structural checks against
the live reset state, plus a full hand-crafted gameplay walkthrough
via real `env.step()` calls - clear the blocking ball, open the box to
reveal the key, unlock and open the door, cross into the second room,
pick up the target ball. Uses BFS pathfinding (not a row-then-column
heuristic) since the blocking ball can end up dropped anywhere,
matching test_unlock.py's own `BlockedUnlockPickup` precedent."""

from __future__ import annotations

from collections import deque

import jax
import jax.numpy as jnp
import numpy as np

import navix as nx
from navix.components import DISCARD_PILE_COORDS, EMPTY_POCKET_ID
from navix.entities import Entities


EAST, SOUTH, WEST, NORTH = 0, 1, 2, 3
ROTATE_CW, FORWARD, PICKUP, DROP, TOGGLE = 1, 2, 3, 4, 5

N_SEEDS = 3
GAMEPLAY_SEEDS = 2


def face_via_step(env, timestep, direction: int):
    for _ in range(4):
        if int(timestep.state.get_player().direction) == direction:
            return timestep
        timestep = env.step(timestep, jnp.asarray(ROTATE_CW))
    raise AssertionError(f"could not face direction {direction}")


def bfs_blocked_mask(state) -> np.ndarray:
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

    if start == target:
        return timestep

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


def test_obstructed_maze_1dlhb_structure():
    env = nx.make("Navix-ObstructedMaze-1Dlhb-v0")
    for seed in range(N_SEEDS):
        state = env.reset(jax.random.PRNGKey(seed)).state

        doors = state.get_doors()
        assert bool(doors.requires[0] == 1), f"seed={seed}: door must require key id 1"
        assert not bool(doors.open[0]), f"seed={seed}: door must start closed"

        keys = state.get_keys()
        assert (keys.position[0] == DISCARD_PILE_COORDS).all(), (
            f"seed={seed}: key must start hidden (discard pile) until the box is opened"
        )

        boxes = state.get_boxes()
        assert int(boxes.pocket[0]) == 1, f"seed={seed}: box must hold key id 1"

        balls = state.get_balls()
        assert balls.position.shape[0] == 2, f"seed={seed}: expected 2 balls (blocker + target)"

        mission_row, mission_col = (int(x) for x in state.mission[0].position)
        assert (mission_row, mission_col) == (
            int(balls.position[1, 0]),
            int(balls.position[1, 1]),
        ), f"seed={seed}: mission must target the second (non-blocking) ball"


def test_obstructed_maze_1dlhb_gameplay_and_reward_termination():
    for seed in range(GAMEPLAY_SEEDS):
        env = nx.make("Navix-ObstructedMaze-1Dlhb-v0")
        timestep = env.reset(jax.random.PRNGKey(seed))
        state = timestep.state
        doors = state.get_doors()
        door_row, door_col = int(doors.position[0, 0]), int(doors.position[0, 1])
        boxes = state.get_boxes()
        box_row, box_col = int(boxes.position[0, 0]), int(boxes.position[0, 1])
        balls = state.get_balls()
        block_row, block_col = int(balls.position[0, 0]), int(balls.position[0, 1])
        target_row, target_col = int(balls.position[1, 0]), int(balls.position[1, 1])

        # clear the blocking ball out of the door's only approach cell
        timestep = bfs_navigate_adjacent_and_face_via_step(env, timestep, block_row, block_col)
        timestep = env.step(timestep, jnp.asarray(PICKUP))
        assert int(timestep.state.get_player().pocket) != int(EMPTY_POCKET_ID), (
            f"seed={seed}: blocking ball not picked up"
        )
        timestep = env.step(timestep, jnp.asarray(FORWARD))
        timestep = face_via_step(env, timestep, WEST)
        timestep = env.step(timestep, jnp.asarray(DROP))
        assert int(timestep.state.get_player().pocket) == int(EMPTY_POCKET_ID), (
            f"seed={seed}: pocket not freed after dropping the ball"
        )

        # open the box, revealing the key at the same position
        timestep = bfs_navigate_adjacent_and_face_via_step(env, timestep, box_row, box_col)
        timestep = env.step(timestep, jnp.asarray(TOGGLE))
        assert (timestep.state.get_boxes().position[0] == DISCARD_PILE_COORDS).all(), (
            f"seed={seed}: box not removed after opening"
        )
        assert (timestep.state.get_keys().position[0] == jnp.asarray([box_row, box_col])).all(), (
            f"seed={seed}: key not revealed at the box's former position"
        )

        # still facing that same cell - pick up the now-revealed key
        timestep = env.step(timestep, jnp.asarray(PICKUP))
        assert int(timestep.state.get_player().pocket) == 1, f"seed={seed}: key not picked up"

        # unlock and open the door
        timestep = bfs_navigate_adjacent_and_face_via_step(env, timestep, door_row, door_col)
        timestep = env.step(timestep, jnp.asarray(TOGGLE))
        assert bool(timestep.state.get_doors().open[0]), f"seed={seed}: door not opened"
        assert timestep.step_type == 0, f"seed={seed}: episode ended before reaching the target"

        # cross through and pick up the target ball
        timestep = bfs_navigate_adjacent_and_face_via_step(env, timestep, target_row, target_col)
        timestep = env.step(timestep, jnp.asarray(PICKUP))
        assert timestep.step_type == 2, f"seed={seed}: expected termination on the target pickup"
        assert float(timestep.reward) > 0, f"seed={seed}: expected positive reward"


def test_obstructed_maze_1dlhb_wrong_pickup_does_not_terminate():
    # picking up the blocking ball (the "wrong" pickup) must not end
    # the episode - only terminations.on_target_fetched, keyed to the
    # specific mission-target ball, does.
    env = nx.make("Navix-ObstructedMaze-1Dlhb-v0")
    timestep = env.reset(jax.random.PRNGKey(0))
    balls = timestep.state.get_balls()
    block_row, block_col = int(balls.position[0, 0]), int(balls.position[0, 1])
    timestep = bfs_navigate_adjacent_and_face_via_step(env, timestep, block_row, block_col)
    timestep = env.step(timestep, jnp.asarray(PICKUP))
    assert int(timestep.state.get_player().pocket) != int(EMPTY_POCKET_ID)
    assert timestep.step_type == 0, "picking up the blocking ball must not terminate the episode"
    assert float(timestep.reward) == 0


def test_obstructed_maze_1dlhb_jit_vmap_compatible():
    env = nx.make("Navix-ObstructedMaze-1Dlhb-v0")
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    timestep = jax.vmap(reset)(keys)
    for action in range(7):
        timestep = jax.vmap(step, in_axes=(0, None))(timestep, jnp.asarray(action))
    jax.block_until_ready(timestep)
