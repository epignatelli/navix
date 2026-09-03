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

"""`MultiRoom` (issue #182): structural checks against the live reset
state (room count, in-bounds/non-overlapping rooms, valid door/player/
goal placement) across all 3 registrations, plus a full real-gameplay
walkthrough for `N2`/`N4` only.

`Navix-MultiRoom-N6-v0` is deliberately kept to structural + jit/vmap
checks, no real gameplay walkthrough - see `multi_room.py`'s own
module docstring for why it's the most expensive environment in
navix by a wide margin (a 25x25 grid, matching MiniGrid's own fixed
default for every registration regardless of room count/size). A full
BFS walkthrough through that maze, on top of `N6`'s own generation
being the heaviest compile in the whole codebase, would risk exactly
the CI cost explosion #196's OOM and #199's review both already
flagged for smaller environments - deliberately not repeated here."""

from __future__ import annotations

from collections import deque

import jax
import jax.numpy as jnp
import numpy as np

import navix as nx
from navix.entities import Entities
from navix.environments import multi_room


EAST, SOUTH, WEST, NORTH = 0, 1, 2, 3
ROTATE_CW, FORWARD, TOGGLE = 1, 2, 5

N_SEEDS = 3
GAMEPLAY_SEEDS = 2

ALL_ENV_IDS = (
    "Navix-MultiRoom-N2-S4-v0",
    "Navix-MultiRoom-N4-S5-v0",
    "Navix-MultiRoom-N6-v0",
)
# num_rooms - matches this file's own registration kwargs (N4-S5 here
# is MiniGrid's own -v1 fix, 4 rooms - not the misleadingly-named
# legacy -v0, which is actually 6 - see multi_room.py's registration
# comment)
NUM_ROOMS = {
    "Navix-MultiRoom-N2-S4-v0": 2,
    "Navix-MultiRoom-N4-S5-v0": 4,
    "Navix-MultiRoom-N6-v0": 6,
}
GAMEPLAY_ENV_IDS = ("Navix-MultiRoom-N2-S4-v0", "Navix-MultiRoom-N4-S5-v0")


def face_via_step(step_fn, timestep, direction: int):
    for _ in range(4):
        if int(timestep.state.get_player().direction) == direction:
            return timestep
        timestep = step_fn(timestep, jnp.asarray(ROTATE_CW))
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


def bfs_walk_path_via_step(step_fn, timestep, path):
    for a, b in zip(path[:-1], path[1:]):
        timestep = face_via_step(step_fn, timestep, bfs_direction_between(a, b))
        timestep = step_fn(timestep, jnp.asarray(FORWARD))
    return timestep


def bfs_navigate_adjacent_and_face_via_step(step_fn, timestep, target_row: int, target_col: int):
    state = timestep.state
    blocked = bfs_blocked_mask(state)
    start = (int(state.get_player().position[0]), int(state.get_player().position[1]))
    target = (target_row, target_col)
    if start == target:
        return timestep
    # The target itself (e.g. the goal) is ordinary walkable floor in
    # `blocked`, so a route to some *other* approach cell past it could
    # path straight through it - stepping onto the goal cell mid-route
    # already terminates the episode, and a subsequent step against
    # that terminated timestep silently autoresets (this is exactly
    # what produced the "player back at the original start position"
    # symptom this test was chasing). Block the target cell for
    # path-finding purposes only, so routes go around it, never
    # through it - it's still a valid *destination* via the
    # `candidate == start`/direct-neighbour checks below.
    blocked_for_path = blocked.copy()
    blocked_for_path[target] = True
    for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        candidate = (target[0] + dr, target[1] + dc)
        if not (0 <= candidate[0] < blocked.shape[0] and 0 <= candidate[1] < blocked.shape[1]):
            continue
        if blocked[candidate]:
            continue
        if candidate == start:
            return face_via_step(step_fn, timestep, bfs_direction_between(candidate, target))
        path = bfs_path(blocked_for_path, start, candidate)
        if path is not None:
            timestep = bfs_walk_path_via_step(step_fn, timestep, path)
            return face_via_step(step_fn, timestep, bfs_direction_between(candidate, target))
    raise AssertionError(f"no reachable approach cell for target {target}")


def assert_valid_multi_room_state(state, n: int, where: str):
    assert state.grid.shape == (25, 25), f"{where}: grid must be 25x25"

    doors = state.get_doors()
    assert doors.position.shape[0] == n - 1, where
    assert not bool(np.asarray(doors.open).any()), f"{where}: doors must start closed"
    assert bool((np.asarray(doors.requires) == -1).all()), f"{where}: doors are unlocked"

    player = state.get_player()
    goal = state.get_goals()
    grid = np.asarray(state.grid)
    assert int(grid[player.position[0], player.position[1]]) == 0, f"{where}: player not on floor"
    assert int(grid[goal.position[0, 0], goal.position[0, 1]]) == 0, (
        f"{where}: goal not on floor (this exact failure mode - a degenerate room "
        f"whose 'floor' carves to nothing - is what fallback_layout exists to "
        f"prevent; see multi_room.py)"
    )

    # every door cell must itself be real floor (punched through
    # the wall, not left as an orphaned -1)
    for row, col in np.asarray(doors.position):
        assert int(grid[row, col]) == 0, f"{where}: door at ({row},{col}) not floor"

    # every door must actually be reachable from the room before it and
    # the room after it - the exact class of bug a purely
    # bounds/overlap-based check (the assertions above) can miss: a
    # door landing on a room's own corner instead of its interior
    # passes every check above but disconnects the two rooms it's
    # supposed to join (found via direct tracing on Navix-MultiRoom-
    # N4-S5-v0, PRNGKey(0) - see multi_room.py's _room_top_left_*
    # docstrings).
    blocked = np.asarray(state.grid) == -1
    for row, col in np.asarray(doors.position):
        neighbours = [(row + 1, col), (row - 1, col), (row, col + 1), (row, col - 1)]
        floor_neighbours = sum(
            1
            for r, c in neighbours
            if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and not blocked[r, c]
        )
        assert floor_neighbours >= 2, (
            f"{where}: door at ({row},{col}) has only {floor_neighbours} floor "
            f"neighbour(s) - should border a room on each side"
        )


def test_multi_room_structure():
    for env_id in ALL_ENV_IDS:
        n = NUM_ROOMS[env_id]
        env = nx.make(env_id)
        for seed in range(N_SEEDS):
            state = env.reset(jax.random.PRNGKey(seed)).state
            assert_valid_multi_room_state(state, n, f"{env_id} seed={seed}")


def test_multi_room_fallback_layout_valid():
    """Direct unit test of `fallback_layout` itself (the exact
    function that shipped with a real off-grid bug for larger `n` -
    see multi_room.py), across every real registration's room count -
    in bounds, non-overlapping, doors on real floor once carved into a
    grid the same way `_reset` does."""
    for n in sorted(set(NUM_ROOMS.values())):
        tops, sizes, door_positions, _door_colours = multi_room.fallback_layout(
            jax.random.PRNGKey(0), n
        )
        tops_np, sizes_np = np.asarray(tops), np.asarray(sizes)
        where = f"n={n}"

        for i in range(n):
            top, size = tops_np[i], sizes_np[i]
            assert (top >= 0).all() and (top + size <= multi_room.GRID_SIZE).all(), (
                f"{where}: room {i} out of [0, {multi_room.GRID_SIZE}) bounds "
                f"(top={top}, size={size})"
            )

        grid = -np.ones((multi_room.GRID_SIZE, multi_room.GRID_SIZE), dtype=np.int32)
        for i in range(n):
            top, size = tops_np[i], sizes_np[i]
            grid[top[0] + 1 : top[0] + size[0] - 1, top[1] + 1 : top[1] + size[1] - 1] = 0
        for row, col in np.asarray(door_positions):
            grid[row, col] = 0

        n_floor = int((grid == 0).sum())
        assert n_floor > 0, f"{where}: fallback layout carves to no floor at all"

        blocked = grid == -1
        for row, col in np.asarray(door_positions):
            neighbours = [(row + 1, col), (row - 1, col), (row, col + 1), (row, col - 1)]
            floor_neighbours = sum(
                1
                for r, c in neighbours
                if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and not blocked[r, c]
            )
            assert floor_neighbours >= 2, (
                f"{where}: fallback door at ({row},{col}) has only {floor_neighbours} "
                f"floor neighbour(s)"
            )


def test_multi_room_fallback_path_forced(monkeypatch):
    """Forces every episode's random layout search to fail (`place_
    room` can never succeed with 0 retries), so `_reset` always falls
    back to `fallback_layout` - exercises the `jnp.where(valid, ...,
    fallback...)` integration path directly through a real `env.reset`
    call, not just `fallback_layout` in isolation, per the PR review's
    own suggestion. Room 0 itself isn't retried (see `generate_
    layout`), so `all_valid` is forced False by every *other* room's
    placement failing - true for every real registration (all have
    `n >= 2`)."""
    monkeypatch.setattr(multi_room, "MAX_PLACEMENT_TRIES", 0)
    for env_id in ALL_ENV_IDS:
        n = NUM_ROOMS[env_id]
        env = nx.make(env_id)
        for seed in range(N_SEEDS):
            state = env.reset(jax.random.PRNGKey(seed)).state
            assert_valid_multi_room_state(state, n, f"{env_id} seed={seed} (forced fallback)")


def solve_multi_room(step_fn, timestep):
    """Plays a MultiRoom variant through to the goal - walk through
    each door in order (they connect consecutive rooms 0->1->...-
    >n-1, so opening them in door-array order is always a valid route
    from the start room to the goal room), then to the goal itself."""
    for door_row, door_col in np.asarray(timestep.state.get_doors().position):
        timestep = bfs_navigate_adjacent_and_face_via_step(
            step_fn, timestep, int(door_row), int(door_col)
        )
        timestep = step_fn(timestep, jnp.asarray(TOGGLE))
        assert timestep.step_type == 0, "episode ended before reaching the goal"

    goal = timestep.state.get_goals()
    goal_row, goal_col = int(goal.position[0, 0]), int(goal.position[0, 1])
    timestep = bfs_navigate_adjacent_and_face_via_step(step_fn, timestep, goal_row, goal_col)
    return step_fn(timestep, jnp.asarray(FORWARD))


def test_multi_room_gameplay_and_reward_termination():
    for env_id in GAMEPLAY_ENV_IDS:
        env = nx.make(env_id)
        # `env.step` embeds a full `env.reset` as its `jax.lax.cond`
        # autoreset branch (see `Environment.step`) - for most navix
        # envs that's cheap, but MultiRoom's reset is a genuinely
        # heavy nested bounded-retry search (see multi_room.py's
        # module docstring). Called un-jitted, each top-level
        # `env.step(...)` call retraces and recompiles that whole
        # thing from scratch (eager `lax.cond` doesn't cache across
        # separate Python calls the way `jax.jit` does) - a BFS walk
        # of dozens of steps was measured to balloon to minutes of
        # compile time and multi-GB memory before crashing with an
        # LLVM "Cannot allocate memory" error, reproducible even on an
        # otherwise-idle machine. Jitting once and reusing across the
        # whole walk (the same pattern `test_multi_room_jit_vmap_
        # compatible` already uses) avoids the retrace entirely.
        step_fn = jax.jit(env.step)
        for seed in range(GAMEPLAY_SEEDS):
            timestep = env.reset(jax.random.PRNGKey(seed))
            timestep = solve_multi_room(step_fn, timestep)
            assert timestep.step_type == 2, (
                f"{env_id} seed={seed}: expected termination on reaching the goal"
            )
            assert float(timestep.reward) > 0, f"{env_id} seed={seed}: expected positive reward"


def test_multi_room_jit_vmap_compatible():
    for env_id in ALL_ENV_IDS:
        env = nx.make(env_id)
        keys = jax.random.split(jax.random.PRNGKey(0), 4)
        reset = jax.jit(env.reset)
        step = jax.jit(env.step)
        timestep = jax.vmap(reset)(keys)
        for action in range(7):
            timestep = jax.vmap(step, in_axes=(0, None))(timestep, jnp.asarray(action))
        jax.block_until_ready(timestep)
