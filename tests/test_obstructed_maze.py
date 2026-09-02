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

"""`ObstructedMaze`'s `1D*` variants (issue #183 - the `Full`-based
`2D*`/`1Q`/`2Q`/`Full` registrations need a 3x3 room grid and are
deliberately deferred, see that issue's staged-rollout
recommendation): structural checks against the live reset state, plus
a full gameplay walkthrough via real `env.step()` calls.

The walkthrough adapts to whichever obstacles a variant actually has
rather than being written three times - clear the blocking ball (only
`1Dlhb`), open the box to reveal the key (`1Dlh`/`1Dlhb`) or pick it
straight off the floor (`1Dl`), then unlock the door, cross, and take
the target ball. Uses BFS pathfinding (not a row-then-column
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
# one seed only: each variant's *distinct solve path* is the coverage
# that matters here, and there are three of them - extra seeds replay
# the same path at real env.step() cost. #196's CI job OOMed on
# exactly this shape (several env ids x a gameplay walkthrough in one
# process, each env id its own JIT compilation).
GAMEPLAY_SEEDS = 1

ENV_IDS = (
    "Navix-ObstructedMaze-1Dl-v0",
    "Navix-ObstructedMaze-1Dlh-v0",
    "Navix-ObstructedMaze-1Dlhb-v0",
)
# key_in_box, blocked - MiniGrid's own per-id registration kwargs
VARIANT_FLAGS = {
    "Navix-ObstructedMaze-1Dl-v0": (False, False),
    "Navix-ObstructedMaze-1Dlh-v0": (True, False),
    "Navix-ObstructedMaze-1Dlhb-v0": (True, True),
}


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


def test_obstructed_maze_structure():
    for env_id in ENV_IDS:
        key_in_box, blocked = VARIANT_FLAGS[env_id]
        env = nx.make(env_id)
        for seed in range(N_SEEDS):
            state = env.reset(jax.random.PRNGKey(seed)).state
            where = f"{env_id} seed={seed}"

            doors = state.get_doors()
            assert bool(doors.requires[0] == 1), f"{where}: door must require key id 1"
            assert not bool(doors.open[0]), f"{where}: door must start closed"

            # the key is hidden in a box, or lying on the floor
            keys = state.get_keys()
            hidden = bool((keys.position[0] == DISCARD_PILE_COORDS).all())
            assert hidden == key_in_box, f"{where}: key hidden={hidden}, key_in_box={key_in_box}"
            assert (Entities.BOX in state.entities) == key_in_box, (
                f"{where}: box presence must follow key_in_box"
            )
            if key_in_box:
                assert int(state.get_boxes().pocket[0]) == 1, f"{where}: box must hold key id 1"
            else:
                row, col = int(keys.position[0, 0]), int(keys.position[0, 1])
                assert int(state.grid[row, col]) == 0, f"{where}: key must lie on a floor cell"

            # a blocking ball only exists for the `blocked` variants,
            # and the target is always the last ball
            balls = state.get_balls()
            assert balls.position.shape[0] == (2 if blocked else 1), (
                f"{where}: unexpected ball count for blocked={blocked}"
            )
            assert bool((state.mission[0].position == balls.position[-1]).all()), (
                f"{where}: mission must target the last (non-blocking) ball"
            )


def solve_via_step(env, timestep, seed_label: str):
    """Plays a variant through to the target ball, adapting to whichever
    obstacles it actually has (see this module's docstring)."""
    state = timestep.state
    doors = state.get_doors()
    door_row, door_col = int(doors.position[0, 0]), int(doors.position[0, 1])
    balls = state.get_balls()
    blocked = balls.position.shape[0] == 2
    target_row, target_col = int(balls.position[-1, 0]), int(balls.position[-1, 1])

    # 1. clear the blocking ball out of the door's only approach cell,
    # then drop it back the way we came (a known-free cell)
    if blocked:
        block_row, block_col = int(balls.position[0, 0]), int(balls.position[0, 1])
        timestep = bfs_navigate_adjacent_and_face_via_step(env, timestep, block_row, block_col)
        timestep = env.step(timestep, jnp.asarray(PICKUP))
        assert int(timestep.state.get_player().pocket) != int(EMPTY_POCKET_ID), (
            f"{seed_label}: blocking ball not picked up"
        )
        approach_dir = int(timestep.state.get_player().direction)
        timestep = env.step(timestep, jnp.asarray(FORWARD))
        timestep = face_via_step(env, timestep, (approach_dir + 2) % 4)
        timestep = env.step(timestep, jnp.asarray(DROP))
        assert int(timestep.state.get_player().pocket) == int(EMPTY_POCKET_ID), (
            f"{seed_label}: pocket not freed after dropping the ball"
        )

    # 2. get the key - opening the box first when it is hidden in one
    if Entities.BOX in timestep.state.entities:
        boxes = timestep.state.get_boxes()
        box_row, box_col = int(boxes.position[0, 0]), int(boxes.position[0, 1])
        timestep = bfs_navigate_adjacent_and_face_via_step(env, timestep, box_row, box_col)
        timestep = env.step(timestep, jnp.asarray(TOGGLE))
        assert (timestep.state.get_boxes().position[0] == DISCARD_PILE_COORDS).all(), (
            f"{seed_label}: box not removed after opening"
        )
        assert (
            timestep.state.get_keys().position[0] == jnp.asarray([box_row, box_col])
        ).all(), f"{seed_label}: key not revealed at the box's former position"
        # still facing that same cell - take the now-revealed key
        timestep = env.step(timestep, jnp.asarray(PICKUP))
    else:
        keys = timestep.state.get_keys()
        key_row, key_col = int(keys.position[0, 0]), int(keys.position[0, 1])
        timestep = bfs_navigate_adjacent_and_face_via_step(env, timestep, key_row, key_col)
        timestep = env.step(timestep, jnp.asarray(PICKUP))
    assert int(timestep.state.get_player().pocket) == 1, f"{seed_label}: key not picked up"

    # 3. unlock and open the door
    timestep = bfs_navigate_adjacent_and_face_via_step(env, timestep, door_row, door_col)
    timestep = env.step(timestep, jnp.asarray(TOGGLE))
    assert bool(timestep.state.get_doors().open[0]), f"{seed_label}: door not opened"
    assert timestep.step_type == 0, f"{seed_label}: episode ended before reaching the target"

    # 4. cross through and pick up the target ball
    timestep = bfs_navigate_adjacent_and_face_via_step(env, timestep, target_row, target_col)
    return env.step(timestep, jnp.asarray(PICKUP))


def test_obstructed_maze_gameplay_and_reward_termination():
    for env_id in ENV_IDS:
        env = nx.make(env_id)
        for seed in range(GAMEPLAY_SEEDS):
            timestep = env.reset(jax.random.PRNGKey(seed))
            timestep = solve_via_step(env, timestep, f"{env_id} seed={seed}")
            assert timestep.step_type == 2, (
                f"{env_id} seed={seed}: expected termination on the target pickup"
            )
            assert float(timestep.reward) > 0, (
                f"{env_id} seed={seed}: expected positive reward"
            )


def test_obstructed_maze_wrong_pickup_does_not_terminate():
    # picking up the blocking ball (the "wrong" pickup) must not end
    # the episode - only terminations.on_target_fetched, keyed to the
    # specific mission-target ball, does. Only `1Dlhb` has a blocking
    # ball to get this wrong with.
    env = nx.make("Navix-ObstructedMaze-1Dlhb-v0")
    timestep = env.reset(jax.random.PRNGKey(0))
    balls = timestep.state.get_balls()
    block_row, block_col = int(balls.position[0, 0]), int(balls.position[0, 1])
    timestep = bfs_navigate_adjacent_and_face_via_step(env, timestep, block_row, block_col)
    timestep = env.step(timestep, jnp.asarray(PICKUP))
    assert int(timestep.state.get_player().pocket) != int(EMPTY_POCKET_ID)
    assert timestep.step_type == 0, "picking up the blocking ball must not terminate the episode"
    assert float(timestep.reward) == 0


def test_obstructed_maze_jit_vmap_compatible():
    for env_id in ENV_IDS:
        env = nx.make(env_id)
        keys = jax.random.split(jax.random.PRNGKey(0), 4)
        reset = jax.jit(env.reset)
        step = jax.jit(env.step)
        timestep = jax.vmap(reset)(keys)
        for action in range(7):
            timestep = jax.vmap(step, in_axes=(0, None))(timestep, jnp.asarray(action))
        jax.block_until_ready(timestep)
