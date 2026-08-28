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

"""KeyCorridor's door placement mirrors MiniGrid's RoomGrid.connect_all
- a randomised union-find that must guarantee every non-goal room stays
reachable from the agent's start, the goal room never gets a second
(openable) connector, and no two doors ever share a cell. #160 and #161
were both silent regressions of exactly these invariants, so this
checks them directly across many seeds rather than relying on luck."""

from typing import Optional, Set, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import navix as nx
from navix.environments.key_corridor import _UNOPENABLE

_ENV_IDS = (
    "Navix-KeyCorridorS3R1-v0",
    "Navix-KeyCorridorS3R2-v0",
    "Navix-KeyCorridorS3R3-v0",
    "Navix-KeyCorridorS4R3-v0",
    "Navix-KeyCorridorS5R3-v0",
    "Navix-KeyCorridorS6R3-v0",
)
_N_SEEDS = 200
_N_ROWS_CONFIG = {3: 1, 5: 2}


def _room_of(pos: Tuple[int, int], room_size: int) -> Optional[Tuple[int, int]]:
    """(room_row, room_col) iff `pos` is strictly inside a room, else
    None - naive floor division can't tell a wall cell from an interior
    one once room_size==1 (walls and interiors sit back-to-back)."""
    r_off = (pos[0] - 1) % (room_size + 1)
    c_off = (pos[1] - 1) % (room_size + 1)
    if not (0 <= r_off < room_size and 0 <= c_off < room_size):
        return None
    return (pos[0] - 1) // (room_size + 1), (pos[1] - 1) // (room_size + 1)


def _rooms_touching_wall(
    pos: Tuple[int, int], room_size: int, n_rows: int
) -> Set[Tuple[int, int]]:
    """A wall/door cell borders exactly two rooms - inspect its 4 neighbours."""
    rooms = set()
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        room = _room_of((pos[0] + dr, pos[1] + dc), room_size)
        if room is not None and 0 <= room[0] < n_rows and 0 <= room[1] < 3:
            rooms.add(room)
    return rooms


def _reachable_rooms(
    door_positions, room_size: int, n_rows: int, start: Tuple[int, int]
) -> Set[Tuple[int, int]]:
    """Rooms reachable from `start` via door entities (any `requires`
    counts as a structural edge, matching MiniGrid's own reachability
    check) plus the always-open middle corridor."""
    adjacency = {(r, c): set() for r in range(n_rows) for c in range(3)}
    for row in range(n_rows - 1):
        adjacency[(row, 1)].add((row + 1, 1))
        adjacency[(row + 1, 1)].add((row, 1))
    for pos in door_positions:
        touching = list(_rooms_touching_wall(tuple(pos), room_size, n_rows))
        if len(touching) == 2:
            a, b = touching
            adjacency[a].add(b)
            adjacency[b].add(a)

    seen = {start}
    stack = [start]
    while stack:
        room = stack.pop()
        for neighbour in adjacency[room]:
            if neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
    return seen


@pytest.mark.parametrize("env_id", _ENV_IDS)
def test_connect_all_reachability_and_door_uniqueness(env_id):
    env = nx.make(env_id)
    n_rows = _N_ROWS_CONFIG.get(env.height, 3)
    room_size = (env.width - 3) // 3

    keys = jax.vmap(jax.random.PRNGKey)(jnp.arange(_N_SEEDS))
    timestep = jax.jit(jax.vmap(env.reset))(keys)  # also checks jit/vmap-compatibility

    positions = np.asarray(timestep.state.entities["door"].position)
    requires = np.asarray(timestep.state.entities["door"].requires)
    key_ids = np.asarray(timestep.state.entities["key"].id)[:, 0]
    agent_positions = np.asarray(timestep.state.entities["player"].position)[:, 0]
    unopenable = int(_UNOPENABLE)

    for seed in range(_N_SEEDS):
        seed_positions = positions[seed]
        seed_requires = requires[seed]
        key_id = int(key_ids[seed])

        unique_positions = set(map(tuple, seed_positions.tolist()))
        assert len(unique_positions) == len(seed_positions), (
            f"{env_id} seed={seed}: expected every door on its own cell, got "
            f"{len(seed_positions)} doors on {len(unique_positions)} distinct cells"
        )

        goal_mask = seed_requires == key_id
        assert goal_mask.sum() == 1, (
            f"{env_id} seed={seed}: expected exactly one locked goal door, "
            f"got {goal_mask.sum()}"
        )
        goal_touching = _rooms_touching_wall(
            tuple(seed_positions[goal_mask][0]), room_size, n_rows
        )
        assert len(goal_touching) == 2
        locked_room = next(r for r in goal_touching if r[1] == 2)

        agent_room = _room_of(tuple(agent_positions[seed]), room_size)
        assert agent_room is not None and agent_room[1] == 1

        reach = _reachable_rooms(seed_positions, room_size, n_rows, agent_room)
        non_locked = {(r, c) for r in range(n_rows) for c in range(3)} - {locked_room}
        assert non_locked.issubset(reach), (
            f"{env_id} seed={seed}: unreachable non-goal rooms "
            f"{non_locked - reach}"
        )

        for pos, req in zip(seed_positions, seed_requires):
            if req == key_id or req == unopenable:
                continue
            touching = _rooms_touching_wall(tuple(pos), room_size, n_rows)
            assert locked_room not in touching, (
                f"{env_id} seed={seed}: openable door at {tuple(pos)} "
                f"(requires={req}) bypasses the locked room"
            )
