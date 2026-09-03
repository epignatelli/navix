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

"""`ObstructedMaze`'s `Full`-based variants (issue #183: `2Dl`, `2Dlh`,
`2Dlhb`, `1Q`, `2Q`, `Full` - the 3x3-room-grid registrations `test_
obstructed_maze.py`'s own docstring deferred). Structural checks
against the live reset state, plus:

- a full gameplay walkthrough (as `test_obstructed_maze.py` does for
  the `1D*` variants) for the three single-quarter ids only (`2Dl`/
  `2Dlh`/`2Dlhb`) - `1Q`/`2Q`/`Full` are big enough (up to 12 doors, 8
  boxes, 9 balls, a 16x16 grid) that a real BFS walkthrough through
  the actual GitHub Actions runner would very plausibly repeat #196's
  CI OOM, and development-time verification on a much bigger machine
  confirmed even *that* environment's own process-lifetime jaxlib
  ceiling (already tracked from #194) crashes a full walkthrough there
  intermittently, purely from its size - independent of anything this
  file can control.
- for `1Q`/`2Q`/`Full` instead: a direct, single-step door/box
  isolation check via state surgery (teleport the player, not a real
  BFS walk) - this is real functional coverage for the thing that
  actually differs at this scale (does opening one of several live
  doors/boxes affect only that one?), at a fraction of the cost. This
  is exactly the class of bug #191 already found once (`actions.
  pickup`'s N=2 broadcasting bug) - worth checking directly rather
  than only inferring it from the N=1 case working."""

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
GAMEPLAY_SEEDS = 1  # see test_obstructed_maze.py's GAMEPLAY_SEEDS for why

FULL_ENV_IDS = (
    "Navix-ObstructedMaze-2Dl-v0",
    "Navix-ObstructedMaze-2Dlh-v0",
    "Navix-ObstructedMaze-2Dlhb-v0",
    "Navix-ObstructedMaze-1Q-v0",
    "Navix-ObstructedMaze-2Q-v0",
    "Navix-ObstructedMaze-Full-v0",
)
SINGLE_QUARTER_ENV_IDS = (
    "Navix-ObstructedMaze-2Dl-v0",
    "Navix-ObstructedMaze-2Dlh-v0",
    "Navix-ObstructedMaze-2Dlhb-v0",
)
MULTI_QUARTER_ENV_IDS = (
    "Navix-ObstructedMaze-2Q-v0",
    "Navix-ObstructedMaze-Full-v0",
)
# num_quarters, key_in_box, blocked - MiniGrid's own per-id kwargs
# (2Dlhb/1Q/2Q/Full carry MiniGrid's -v1 fix, registered as -v0 - see
# ObstructedMazeFull's own docstring)
VARIANT_SPEC = {
    "Navix-ObstructedMaze-2Dl-v0": (1, False, False),
    "Navix-ObstructedMaze-2Dlh-v0": (1, True, False),
    "Navix-ObstructedMaze-2Dlhb-v0": (1, True, True),
    "Navix-ObstructedMaze-1Q-v0": (1, True, True),
    "Navix-ObstructedMaze-2Q-v0": (2, True, True),
    "Navix-ObstructedMaze-Full-v0": (4, True, True),
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
            for a, b in zip(path[:-1], path[1:]):
                timestep = face_via_step(env, timestep, bfs_direction_between(a, b))
                timestep = env.step(timestep, jnp.asarray(FORWARD))
            return face_via_step(env, timestep, bfs_direction_between(candidate, target))
    raise AssertionError(f"no reachable approach cell for target {target}")


def test_obstructed_maze_full_structure():
    for env_id in FULL_ENV_IDS:
        num_quarters, key_in_box, blocked = VARIANT_SPEC[env_id]
        env = nx.make(env_id)
        for seed in range(N_SEEDS):
            state = env.reset(jax.random.PRNGKey(seed)).state
            where = f"{env_id} seed={seed}"

            assert state.grid.shape == (16, 16), f"{where}: 3x3 room_size=6 grid must be 16x16"

            doors = state.get_doors()
            assert doors.position.shape[0] == 3 * num_quarters, where
            requires = np.asarray(doors.requires)
            assert int((requires != -1).sum()) == 2 * num_quarters, where
            assert not bool(np.asarray(doors.open).any()), f"{where}: doors must start closed"

            keys = state.get_keys()
            assert keys.position.shape[0] == 2 * num_quarters, where
            hidden = np.all(
                np.asarray(keys.position) == np.asarray(DISCARD_PILE_COORDS), axis=-1
            )
            assert bool(hidden.all()) == key_in_box, f"{where}: key hidden must follow key_in_box"
            # every locked door's `requires` names a real key id, and
            # vice versa - no orphaned door or unreachable key
            assert set(requires[requires != -1].tolist()) == set(
                np.asarray(keys.id).tolist()
            ), f"{where}: door.requires <-> key.id mismatch"

            assert (Entities.BOX in state.entities) == key_in_box, where
            if key_in_box:
                boxes = state.get_boxes()
                assert boxes.position.shape[0] == 2 * num_quarters, where
                assert set(np.asarray(boxes.pocket).tolist()) == set(
                    np.asarray(keys.id).tolist()
                ), f"{where}: every box hides exactly one real key"

            balls = state.get_balls()
            want_balls = (2 * num_quarters if blocked else 0) + 1
            assert balls.position.shape[0] == want_balls, where
            assert bool((state.mission[0].position == balls.position[-1]).all()), (
                f"{where}: mission must target the last (non-blocking) ball"
            )

            # MiniGrid's -v1 fix (see ObstructedMazeFull's docstring):
            # nothing may share a cell - in particular, no blocking
            # ball may cover a key
            occupied = {}
            for name, entity in state.entities.items():
                positions = np.asarray(entity.position)
                if positions.ndim == 1:
                    positions = positions[None]
                for row, col in positions:
                    if (row, col) == tuple(np.asarray(DISCARD_PILE_COORDS)):
                        continue
                    cell = (int(row), int(col))
                    assert cell not in occupied, (
                        f"{where}: {name} and {occupied.get(cell)} both at {cell}"
                    )
                    occupied[cell] = name


def test_obstructed_maze_full_target_corner_varies():
    # only Full has all 4 corners reachable - the single/double-quarter
    # variants necessarily always land in the same 1 or 2 corners
    env = nx.make("Navix-ObstructedMaze-Full-v0")
    corners = {
        tuple(int(x) for x in env.reset(jax.random.PRNGKey(seed)).state.mission[0].position)
        for seed in range(20)
    }
    assert len(corners) > 1, "target corner should vary across seeds"


def test_obstructed_maze_full_door_and_box_isolation():
    # the class of bug #191 already found once in actions.pickup (an
    # N=1-only broadcast that silently corrupted every instance once a
    # 2nd one existed) - direct proof that opening one of several live
    # doors/boxes affects only that one, via state surgery (teleporting
    # the player next to it) rather than a full walkthrough.
    for env_id in MULTI_QUARTER_ENV_IDS:
        num_quarters, _, _ = VARIANT_SPEC[env_id]
        env = nx.make(env_id)
        timestep = env.reset(jax.random.PRNGKey(0))
        state = timestep.state
        doors = state.get_doors()
        requires = np.asarray(doors.requires)
        locked = np.where(requires != -1)[0]
        assert len(locked) == 2 * num_quarters, env_id

        def adjacent_floor(state, cell, label=env_id):
            for dr, dc in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                candidate = (cell[0] + dr, cell[1] + dc)
                if int(state.grid[candidate]) == 0:
                    direction = bfs_direction_between(candidate, cell)
                    return candidate, direction
            raise AssertionError(f"{label}: no adjacent floor cell to {cell}")

        # open the first locked door only (teleport + hand the player
        # its key directly - the pickup/box-open mechanism is already
        # proven by test_obstructed_maze.py's 1D walkthrough)
        i = int(locked[0])
        door_pos = tuple(int(x) for x in np.asarray(doors.position)[i])
        adjacent, direction = adjacent_floor(state, door_pos)
        player = state.get_player().replace(
            position=jnp.asarray(adjacent), direction=jnp.asarray(direction)
        )
        player = player.replace(pocket=jnp.asarray(int(requires[i])))
        timestep = timestep.replace(state=state.set_player(player))
        timestep = env.step(timestep, jnp.asarray(TOGGLE))

        opened = np.asarray(timestep.state.get_doors().open)
        assert bool(opened[i]), f"{env_id}: door {i} should now be open"
        for j in locked:
            if j != i:
                assert not bool(opened[j]), f"{env_id}: door {j} must stay closed"

        # open the first box only
        boxes = timestep.state.get_boxes()
        box_positions = np.asarray(boxes.position)
        box_pockets = np.asarray(boxes.pocket)
        box_pos = tuple(int(x) for x in box_positions[0])
        adjacent, direction = adjacent_floor(timestep.state, box_pos)
        player = timestep.state.get_player().replace(
            position=jnp.asarray(adjacent), direction=jnp.asarray(direction)
        )
        timestep = timestep.replace(state=timestep.state.set_player(player))

        before_keys = np.asarray(timestep.state.get_keys().position).copy()
        timestep = env.step(timestep, jnp.asarray(TOGGLE))
        after_keys = np.asarray(timestep.state.get_keys().position)
        key_ids = np.asarray(timestep.state.get_keys().id)
        revealed_id = box_pockets[0]
        for key_id, before, after in zip(key_ids, before_keys, after_keys):
            if key_id == revealed_id:
                assert tuple(after) == box_pos, f"{env_id}: key {key_id} not revealed"
            else:
                assert tuple(before) == tuple(after), (
                    f"{env_id}: key {key_id} disturbed by a different box opening"
                )
        after_boxes = np.asarray(timestep.state.get_boxes().position)
        assert tuple(after_boxes[0]) == tuple(np.asarray(DISCARD_PILE_COORDS)), (
            f"{env_id}: opened box must be discarded"
        )
        for other in range(1, len(box_positions)):
            assert tuple(after_boxes[other]) == tuple(box_positions[other]), (
                f"{env_id}: box {other} must be untouched"
            )


def room_of(row: int, col: int) -> tuple:
    step = 5  # room_size - 1
    return (row // step, col // step)


def neighbour_rooms(grid: np.ndarray, row: int, col: int) -> list:
    """The room(s) either side of a door at (row, col) - a door sits on
    a shared wall, so exactly the two floor cells orthogonally adjacent
    to it (if both exist) tell us which two rooms it connects."""
    rooms = []
    for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        r, c = row + dr, col + dc
        if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and int(grid[r, c]) == 0:
            rooms.append(room_of(r, c))
    return rooms


def solve_one_quarter(env, timestep):
    """Plays a single-quarter (2Dl/2Dlh/2Dlhb) variant through to the
    target ball. `num_quarters=1` still means *two* locked doors (one
    side room, two corners) - only one of them actually leads to the
    target's corner, found here via room adjacency, the same way `1Q`/
    `2Q`/`Full`'s own dev-time verification script did. The agent
    starts inside the shared side room, so no unlocked-door crossing is
    needed first (unlike `1Q`, which starts in the centre - see
    test_obstructed_maze.py's `solve_via_step` for the 1D equivalent
    of this same "adapt to whichever obstacles exist" approach)."""
    state = timestep.state
    grid = np.asarray(state.grid)
    doors = state.get_doors()
    door_positions = np.asarray(doors.position)
    requires = np.asarray(doors.requires)

    balls = state.get_balls()
    # NOT `== 2` (that was test_obstructed_maze.py's 1D convention -
    # exactly 1 blocker + 1 target there, always). A single-quarter
    # `Full`-based room has *two* locked doors (one side room, two
    # corners), so `blocked=True` here means two blockers + one target
    # = 3 balls, not 2 - this literally never matched, silently
    # skipping blocker-clearing for `2Dlhb` every single time (root-
    # caused after a deterministic, reproducible failure was first
    # mistaken for the harness's known jaxlib flakiness - see the PR).
    blocked = balls.position.shape[0] > 1
    target_row, target_col = (int(x) for x in np.asarray(balls.position)[-1])
    target_room = room_of(target_row, target_col)

    door_idx = None
    side_room = None
    for i, (row, col) in enumerate(door_positions):
        if requires[i] == -1:
            continue
        rooms = neighbour_rooms(grid, int(row), int(col))
        if target_room in rooms:
            door_idx = i
            side_room = [r for r in rooms if r != target_room][0]
            break
    assert door_idx is not None, f"no locked door found into target room {target_room}"
    door_row, door_col = (int(x) for x in door_positions[door_idx])
    key_id = int(requires[door_idx])

    if blocked:
        # the blocking ball backed off *this specific* door, not
        # necessarily balls[0] - with two locked doors there are two
        # blockers, one per door. Identified by construction (see
        # obstructed_maze.py's `_reset`: `door_pos - SIDE_DELTAS[side]`
        # always lands one step into the *side* room, never the
        # corner) rather than "whichever is closest", which can pick
        # the wrong one when a room's two doors sit close together.
        blocker_positions = np.asarray(timestep.state.get_balls().position)[:-1]
        candidates = [
            (int(r), int(c))
            for r, c in blocker_positions
            if room_of(int(r), int(c)) == side_room
            and abs(int(r) - door_row) + abs(int(c) - door_col) == 1
        ]
        assert len(candidates) == 1, (
            f"expected exactly one blocker adjacent to the target door in {side_room}, "
            f"got {candidates}"
        )
        block_row, block_col = candidates[0]
        timestep = bfs_navigate_adjacent_and_face_via_step(env, timestep, block_row, block_col)
        timestep = env.step(timestep, jnp.asarray(PICKUP))
        assert int(timestep.state.get_player().pocket) != int(EMPTY_POCKET_ID)
        approach_dir = int(timestep.state.get_player().direction)
        timestep = env.step(timestep, jnp.asarray(FORWARD))
        timestep = face_via_step(env, timestep, (approach_dir + 2) % 4)
        timestep = env.step(timestep, jnp.asarray(DROP))
        assert int(timestep.state.get_player().pocket) == int(EMPTY_POCKET_ID)

    if Entities.BOX in timestep.state.entities:
        boxes = timestep.state.get_boxes()
        idx = int(np.where(np.asarray(boxes.pocket) == key_id)[0][0])
        box_row, box_col = (int(x) for x in np.asarray(boxes.position)[idx])
        timestep = bfs_navigate_adjacent_and_face_via_step(env, timestep, box_row, box_col)
        timestep = env.step(timestep, jnp.asarray(TOGGLE))
        revealed = np.asarray(timestep.state.get_keys().position)[
            np.asarray(timestep.state.get_keys().id) == key_id
        ][0]
        assert tuple(revealed) == (box_row, box_col), "key not revealed at the box's position"
        timestep = env.step(timestep, jnp.asarray(PICKUP))
    else:
        keys = timestep.state.get_keys()
        idx = int(np.where(np.asarray(keys.id) == key_id)[0][0])
        key_row, key_col = (int(x) for x in np.asarray(keys.position)[idx])
        timestep = bfs_navigate_adjacent_and_face_via_step(env, timestep, key_row, key_col)
        timestep = env.step(timestep, jnp.asarray(PICKUP))
    assert int(timestep.state.get_player().pocket) == key_id, "key not picked up"

    timestep = bfs_navigate_adjacent_and_face_via_step(env, timestep, door_row, door_col)
    timestep = env.step(timestep, jnp.asarray(TOGGLE))
    assert bool(timestep.state.get_doors().open[door_idx]), "door not opened"
    assert timestep.step_type == 0, "episode ended before reaching the target"

    timestep = bfs_navigate_adjacent_and_face_via_step(env, timestep, target_row, target_col)
    return env.step(timestep, jnp.asarray(PICKUP))


def _gameplay_and_reward_termination(env_id: str):
    env = nx.make(env_id)
    for seed in range(GAMEPLAY_SEEDS):
        timestep = env.reset(jax.random.PRNGKey(seed))
        timestep = solve_one_quarter(env, timestep)
        assert timestep.step_type == 2, (
            f"{env_id} seed={seed}: expected termination on the target pickup"
        )
        assert float(timestep.reward) > 0, f"{env_id} seed={seed}: expected positive reward"


# One test function per variant, not one function looping over all
# three - #196's CI OOM came from exactly this shape (several real
# gameplay walkthroughs accumulating in one process/job); separate
# top-level test functions at least give pytest-xdist the chance to
# spread them across workers instead of concentrating the load.
def test_obstructed_maze_2dl_gameplay_and_reward_termination():
    _gameplay_and_reward_termination("Navix-ObstructedMaze-2Dl-v0")


def test_obstructed_maze_2dlh_gameplay_and_reward_termination():
    _gameplay_and_reward_termination("Navix-ObstructedMaze-2Dlh-v0")


def test_obstructed_maze_2dlhb_gameplay_and_reward_termination():
    _gameplay_and_reward_termination("Navix-ObstructedMaze-2Dlhb-v0")


def test_obstructed_maze_full_jit_vmap_compatible():
    for env_id in FULL_ENV_IDS:
        env = nx.make(env_id)
        keys = jax.random.split(jax.random.PRNGKey(0), 4)
        reset = jax.jit(env.reset)
        step = jax.jit(env.step)
        timestep = jax.vmap(reset)(keys)
        for action in range(7):
            timestep = jax.vmap(step, in_axes=(0, None))(timestep, jnp.asarray(action))
        jax.block_until_ready(timestep)
