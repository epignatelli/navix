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

"""`PutNear` (issue #178): structural checks against the live reset
state (`mission`/`mission2` distinct, each matching a real object),
plus real `env.step()` gameplay covering picking up the wrong object
(immediate failure), and picking up the right one and dropping it near
- vs. not near - the target."""

from __future__ import annotations

from collections import deque

import jax
import jax.numpy as jnp
import numpy as np

import navix as nx
from navix.components import DISCARD_PILE_COORDS
from navix.entities import Entities


EAST, SOUTH, WEST, NORTH = 0, 1, 2, 3
ROTATE_CW, FORWARD, PICKUP, DROP = 1, 2, 3, 4

N_SEEDS = 20
# BFS-navigation gameplay tests are slow on eager (uncompiled) CPU JAX -
# same finding as test_unlock.py's GAMEPLAY_SEEDS.
GAMEPLAY_SEEDS = 2


def blocked_mask(state) -> np.ndarray:
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
        timestep = env.step(timestep, jnp.asarray(FORWARD))
    return timestep


def navigate_adjacent_and_face(env, timestep, target_row: int, target_col: int):
    state = timestep.state
    blocked = blocked_mask(state)
    start = (int(state.get_player().position[0]), int(state.get_player().position[1]))
    target = (target_row, target_col)

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


def navigate_to_drop_near(env, timestep, target_row: int, target_col: int):
    """Faces the player towards some empty cell within Chebyshev
    distance 1 of `(target_row, target_col)` - i.e. wherever `drop`
    would land the carried object close enough to count."""
    state = timestep.state
    blocked = blocked_mask(state)
    start = (int(state.get_player().position[0]), int(state.get_player().position[1]))
    height, width = blocked.shape

    drop_candidates = [
        (target_row + dr, target_col + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    ]
    for drop in drop_candidates:
        if not (0 <= drop[0] < height and 0 <= drop[1] < width) or blocked[drop]:
            continue
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            approach = (drop[0] + dr, drop[1] + dc)
            if not (0 <= approach[0] < height and 0 <= approach[1] < width) or blocked[approach]:
                continue
            if approach == start:
                return face(env, timestep, direction_between(approach, drop))
            path = bfs_path(blocked, start, approach)
            if path is not None:
                timestep = walk_path(env, timestep, path)
                return face(env, timestep, direction_between(approach, drop))
    raise AssertionError(f"no reachable drop cell near target {(target_row, target_col)}")


def all_object_positions(state):
    # Key/Ball/Box each always have n_objects slots (see
    # navix/environments/go_to_object.py's module docstring for the
    # padding-sentinel design put_near.py shares) - filtered down to
    # just the n_objects real ones, matching this helper's pre-Box
    # contract exactly so every caller keeps working unchanged.
    positions = jnp.concatenate(
        [state.get_keys().position, state.get_balls().position, state.get_boxes().position],
        axis=0,
    )
    on_grid = jnp.any(positions != DISCARD_PILE_COORDS, axis=-1)
    return positions[on_grid]


def test_put_near_structure():
    for env_id in ("Navix-PutNear-6x6-N2-v0", "Navix-PutNear-8x8-N3-v0"):
        env = nx.make(env_id)
        for seed in range(N_SEEDS):
            state = env.reset(jax.random.PRNGKey(seed)).state
            assert state.mission is not None and state.mission2 is not None, (
                f"{env_id} seed={seed}: expected both mission targets"
            )
            assert not jnp.array_equal(state.mission.position, state.mission2.position), (
                f"{env_id} seed={seed}: move and target objects must be distinct"
            )
            positions = all_object_positions(state)
            move_matches = jnp.all(positions == state.mission.position, axis=-1)
            target_matches = jnp.all(positions == state.mission2.position, axis=-1)
            assert int(jnp.sum(move_matches)) == 1
            assert int(jnp.sum(target_matches)) == 1


def test_put_near_never_spawns_already_solved():
    # PR #191 review "New risks": random_distinct_positions alone lets
    # the move/target pair spawn already within "near" (Chebyshev <= 1)
    # of each other, trivially solvable with no navigation - quantified
    # directly at 36% of seeds for the 6x6 size before this was fixed.
    # A larger seed count than the other structural tests specifically
    # because this is a statistical claim ("never"), not a per-seed
    # structural invariant - passing on N_SEEDS wouldn't rule out a
    # residual few-percent chance the earlier bug allowed.
    n_stat_seeds = 500
    for env_id in ("Navix-PutNear-6x6-N2-v0", "Navix-PutNear-8x8-N3-v0"):
        env = nx.make(env_id)
        too_close = 0
        for seed in range(n_stat_seeds):
            state = env.reset(jax.random.PRNGKey(seed)).state
            move, target = state.mission.position, state.mission2.position
            chebyshev = jnp.maximum(jnp.abs(move[0] - target[0]), jnp.abs(move[1] - target[1]))
            if int(chebyshev) <= 1:
                too_close += 1
        assert too_close == 0, (
            f"{env_id}: {too_close}/{n_stat_seeds} seeds spawned with the move "
            f"object already within Chebyshev-1 of the target (already \"solved\")"
        )


def test_put_near_wrong_pickup_fails_immediately():
    env = nx.make("Navix-PutNear-6x6-N2-v0")
    for seed in range(GAMEPLAY_SEEDS):
        timestep = env.reset(jax.random.PRNGKey(seed))
        state = timestep.state
        positions = all_object_positions(state)
        is_move = jnp.all(positions == state.mission.position, axis=-1)
        wrong_idx = int(jnp.argmin(is_move.astype(jnp.int32)))
        assert not bool(is_move[wrong_idx]), f"seed={seed}: expected a non-move object to exist"
        wrong_row, wrong_col = int(positions[wrong_idx, 0]), int(positions[wrong_idx, 1])

        timestep = navigate_adjacent_and_face(env, timestep, wrong_row, wrong_col)
        timestep = env.step(timestep, jnp.asarray(PICKUP))
        assert timestep.step_type == 2, f"seed={seed}: expected termination on wrong pickup"
        assert float(timestep.reward) == 0, f"seed={seed}: expected zero reward"


def test_put_near_success():
    for seed in range(GAMEPLAY_SEEDS):
        env = nx.make("Navix-PutNear-6x6-N2-v0")
        timestep = env.reset(jax.random.PRNGKey(seed))
        state = timestep.state
        move_row, move_col = int(state.mission.position[0]), int(state.mission.position[1])
        target_row, target_col = int(state.mission2.position[0]), int(state.mission2.position[1])

        timestep = navigate_adjacent_and_face(env, timestep, move_row, move_col)
        timestep = env.step(timestep, jnp.asarray(PICKUP))
        assert timestep.step_type == 0, f"seed={seed}: episode ended on the correct pickup"
        assert int(timestep.state.get_player().pocket) != -1, f"seed={seed}: item not picked up"

        timestep = navigate_to_drop_near(env, timestep, target_row, target_col)
        timestep = env.step(timestep, jnp.asarray(DROP))
        assert timestep.step_type == 2, f"seed={seed}: expected termination on the drop"
        assert float(timestep.reward) > 0, f"seed={seed}: expected positive reward for dropping near the target"


def test_put_near_drop_far_from_target_gives_no_reward():
    env = nx.make("Navix-PutNear-6x6-N2-v0")
    for seed in range(GAMEPLAY_SEEDS):
        timestep = env.reset(jax.random.PRNGKey(seed))
        state = timestep.state
        move_row, move_col = int(state.mission.position[0]), int(state.mission.position[1])

        timestep = navigate_adjacent_and_face(env, timestep, move_row, move_col)
        timestep = env.step(timestep, jnp.asarray(PICKUP))
        assert timestep.step_type == 0

        # drop right where we stand (facing away from the move object's
        # old cell), almost certainly not near mission2's target. The
        # drop itself lands on translate(player.position, player.
        # direction) - one cell *in front of* the player, not the
        # player's own cell (PR #191 review: computing distance from
        # the player's own position instead was a latent flaky-test
        # bug - a layout with Chebyshev(player, target) == 2 but
        # Chebyshev(front, target) == 1 would wrongly expect reward 0).
        state = timestep.state
        target_row, target_col = int(state.mission2.position[0]), int(state.mission2.position[1])
        player = state.get_player()
        drop_pos = nx.grid.translate(player.position, player.direction)
        drop_row, drop_col = int(drop_pos[0]), int(drop_pos[1])
        far_from_target = max(abs(drop_row - target_row), abs(drop_col - target_col)) > 1
        if not far_from_target:
            continue  # rare layouts where the player already ended up near the target

        timestep = env.step(timestep, jnp.asarray(DROP))
        assert timestep.step_type == 2, f"seed={seed}: expected termination on any genuine drop"
        assert float(timestep.reward) == 0, f"seed={seed}: expected zero reward, dropped far from target"


def test_put_near_jit_vmap_compatible():
    for env_id in ("Navix-PutNear-6x6-N2-v0", "Navix-PutNear-8x8-N3-v0"):
        env = nx.make(env_id)
        keys = jax.random.split(jax.random.PRNGKey(0), 4)
        reset = jax.jit(env.reset)
        step = jax.jit(env.step)
        timestep = jax.vmap(reset)(keys)
        for action in range(len(nx.actions.DEFAULT_ACTION_SET)):
            timestep = jax.vmap(step, in_axes=(0, None))(timestep, jnp.asarray(action))
        jax.block_until_ready(timestep)
