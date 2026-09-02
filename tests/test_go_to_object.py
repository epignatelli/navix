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

"""`GoToObject` (issue #173): structural checks against the live reset
state, plus real `env.step()` gameplay covering both the success path
(`done` while facing the target) and the failure path (`toggle` at
all).

Navigation here uses real BFS pathfinding over the room (all state
values are already concrete Python ints by the time a test reads them,
unlike navix.actions itself, which must stay jit/vmap-compatible and
so can't use this) rather than the row-then-column heuristics
`test_unlock.py` needed - simpler and robust to any obstacle layout
without needing to special-case specific seeds."""

from __future__ import annotations

from collections import deque

import jax
import jax.numpy as jnp
import numpy as np

import navix as nx
from navix.entities import Entities


EAST, SOUTH, WEST, NORTH = 0, 1, 2, 3
ROTATE_CW, TOGGLE, DONE = 1, 5, 6

N_SEEDS = 20
# BFS-navigation gameplay tests are slow on eager (uncompiled) CPU JAX -
# same finding as test_unlock.py's GAMEPLAY_SEEDS.
GAMEPLAY_SEEDS = 2


def blocked_mask(state) -> np.ndarray:
    """A `(height, width)` boolean array of cells the player can't step
    onto - the wall grid, plus every non-walkable entity's own cell
    (Key/Ball, both non-walkable; irrelevant for the player itself)."""
    blocked = np.asarray(state.grid) == -1
    for entity_enum, entity in state.entities.items():
        if entity_enum == Entities.PLAYER:
            continue
        positions = np.asarray(entity.position)
        if positions.ndim == 1:
            positions = positions[None]
        for row, col in positions:
            blocked[row, col] = True
    return blocked


def bfs_path(blocked: np.ndarray, start: tuple, goal: tuple):
    """Shortest 4-connected path from `start` to `goal` avoiding
    `blocked` cells, as a list of `(row, col)` from `start` to `goal`
    inclusive - `None` if unreachable."""
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


def face(env, timestep, direction: int):
    for _ in range(4):
        if int(timestep.state.get_player().direction) == direction:
            return timestep
        timestep = env.step(timestep, jnp.asarray(ROTATE_CW))
    raise AssertionError(f"could not face direction {direction}")


def walk_path(env, timestep, path):
    for a, b in zip(path[:-1], path[1:]):
        timestep = face(env, timestep, direction_between(a, b))
        timestep = env.step(timestep, jnp.asarray(2))  # forward
    return timestep


def navigate_adjacent_and_face(env, timestep):
    """Paths the player to an orthogonal neighbour of `state.mission`'s
    target and faces it - picks whichever reachable neighbour BFS finds
    first."""
    state = timestep.state
    target = (int(state.mission.position[0]), int(state.mission.position[1]))
    blocked = blocked_mask(state)
    start = (int(state.get_player().position[0]), int(state.get_player().position[1]))

    for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        candidate = (target[0] + dr, target[1] + dc)
        if not (0 <= candidate[0] < blocked.shape[0] and 0 <= candidate[1] < blocked.shape[1]):
            continue
        if blocked[candidate]:
            continue
        if candidate == start:
            return face(env, timestep, direction_between(candidate, target))
        path = bfs_path(blocked, start, candidate)
        if path is not None:
            timestep = walk_path(env, timestep, path)
            return face(env, timestep, direction_between(candidate, target))
    raise AssertionError(f"no reachable approach cell for target {target}")


def test_go_to_object_structure():
    for env_id in ("Navix-GoToObject-6x6-N2-v0", "Navix-GoToObject-8x8-N2-v0"):
        env = nx.make(env_id)
        for seed in range(N_SEEDS):
            state = env.reset(jax.random.PRNGKey(seed)).state
            assert state.mission is not None, f"{env_id} seed={seed}: expected a mission"

            positions = []
            if Entities.KEY in state.entities:
                positions.append(state.get_keys().position)
            if Entities.BALL in state.entities:
                positions.append(state.get_balls().position)
            positions = jnp.concatenate(positions, axis=0)
            assert positions.shape[0] == 2, f"{env_id} seed={seed}: expected 2 objects total"

            matches = jnp.all(positions == state.mission.position, axis=-1)
            assert int(jnp.sum(matches)) == 1, (
                f"{env_id} seed={seed}: mission.position must match exactly one object"
            )


def test_go_to_object_success_on_done():
    # one representative size - structural/jit_vmap tests already cover
    # every registered size cheaply; the *logic* under test here doesn't
    # vary by room size.
    for env_id in ("Navix-GoToObject-6x6-N2-v0",):
        for seed in range(GAMEPLAY_SEEDS):
            env = nx.make(env_id)
            timestep = env.reset(jax.random.PRNGKey(seed))
            timestep = navigate_adjacent_and_face(env, timestep)
            assert timestep.step_type == 0, f"{env_id} seed={seed}: episode ended before done"

            timestep = env.step(timestep, jnp.asarray(DONE))
            assert timestep.step_type == 2, (
                f"{env_id} seed={seed}: expected termination on done while facing the target"
            )
            assert float(timestep.reward) > 0, f"{env_id} seed={seed}: expected positive reward"


def test_go_to_object_toggle_fails_immediately():
    env = nx.make("Navix-GoToObject-6x6-N2-v0")
    for seed in range(N_SEEDS):
        timestep = env.reset(jax.random.PRNGKey(seed))
        timestep = env.step(timestep, jnp.asarray(TOGGLE))
        assert timestep.step_type == 2, f"seed={seed}: expected termination on any toggle"
        assert float(timestep.reward) == 0, f"seed={seed}: expected zero reward"


def test_go_to_object_jit_vmap_compatible():
    for env_id in ("Navix-GoToObject-6x6-N2-v0", "Navix-GoToObject-8x8-N2-v0"):
        env = nx.make(env_id)
        keys = jax.random.split(jax.random.PRNGKey(0), 4)
        reset = jax.jit(env.reset)
        step = jax.jit(env.step)
        timestep = jax.vmap(reset)(keys)
        for action in range(len(nx.actions.DEFAULT_ACTION_SET)):
            timestep = jax.vmap(step, in_axes=(0, None))(timestep, jnp.asarray(action))
        jax.block_until_ready(timestep)
