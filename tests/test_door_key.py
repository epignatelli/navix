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


"""DoorKey's board follows MiniGrid's `DoorKeyEnv._gen_grid`: the wall
column, door row, goal and key for every id, and the player's cell and
direction for the `-Random-` ids. Checked as exact supports over many seeds."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import navix as nx

_SIZES = (5, 6, 8, 16)
_N_SEEDS = 1000


@pytest.mark.parametrize("size", _SIZES)
@pytest.mark.parametrize("random_start", (False, True))
def test_board_follows_minigrid(size, random_start):
    env_id = f"Navix-DoorKey-{'Random-' if random_start else ''}{size}x{size}-v0"
    env = nx.make(env_id)

    def board(key):
        entities = env.reset(key).state.entities
        return [entities[name] for name in ("player", "door", "key", "goal")]

    keys = jax.vmap(jax.random.PRNGKey)(jnp.arange(_N_SEEDS))
    player, door, key, goal = jax.jit(jax.vmap(board))(keys)
    player_pos = np.asarray(player.position)[:, 0]
    direction = np.asarray(player.direction)[:, 0]
    door_pos = np.asarray(door.position)[:, 0]
    key_pos = np.asarray(key.position)[:, 0]
    goal_pos = np.asarray(goal.position)[:, 0]

    assert set(door_pos[:, 1].tolist()) == set(range(2, size - 2))
    assert set(door_pos[:, 0].tolist()) == set(range(1, size - 2))
    assert np.all(goal_pos == [size - 2, size - 2])

    for p, k, d in zip(player_pos, key_pos, door_pos):
        for name, cell in (("player", p), ("key", k)):
            assert 1 <= cell[0] <= size - 2 and 1 <= cell[1] < d[1], (
                f"{env_id}: {name} at {cell.tolist()} outside the first room "
                f"left of the wall at column {d[1]}"
            )
        assert not np.array_equal(p, k), f"{env_id}: key on the player's cell"

    if random_start:
        assert set(direction.tolist()) == {0, 1, 2, 3}
        if size <= 8:  # at 16x16 some cells are too rare to all show up
            first_room_cells = {
                (r, c) for r in range(1, size - 1) for c in range(1, size - 3)
            }
            assert set(map(tuple, player_pos.tolist())) == first_room_cells
    else:
        assert np.all(player_pos == [1, 1]) and np.all(direction == 0)
