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

"""`Playground` (issue #184): structural checks against the live reset
state (12 doors all unlocked, exactly 12 real Key/Ball/Box objects
total, everyone mutually distinct and on real floor), plus a random
walk confirming the environment is genuinely reward-free and never
terminates before the timeout - there is no goal to reach or reward to
earn here, only truncation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import navix as nx
from navix.components import DISCARD_PILE_COORDS


N_SEEDS = 3
RANDOM_WALK_STEPS = 20

ENV_ID = "Navix-Playground-v0"


def test_playground_structure():
    env = nx.make(ENV_ID)
    for seed in range(N_SEEDS):
        state = env.reset(jax.random.PRNGKey(seed)).state
        where = f"{ENV_ID} seed={seed}"
        grid = np.asarray(state.grid)
        assert grid.shape == (19, 19), f"{where}: grid must be 19x19"

        doors = state.get_doors()
        assert doors.position.shape[0] == 12, f"{where}: expected 12 doors"
        assert not bool(np.asarray(doors.open).any()), f"{where}: doors must start closed"
        assert bool((np.asarray(doors.requires) == -1).all()), f"{where}: doors must be unlocked"
        for row, col in np.asarray(doors.position):
            assert int(grid[row, col]) == 0, f"{where}: door at ({row},{col}) not floor"

        # every object slot (Key/Ball/Box, 12 each) is either a real
        # object or pushed to DISCARD_PILE_COORDS - see playground.py's
        # module docstring (the same GoToObject padding-sentinel trick)
        object_positions = jnp.concatenate(
            [
                state.get_keys().position,
                state.get_balls().position,
                state.get_boxes().position,
            ],
            axis=0,
        )
        on_grid = np.asarray(jnp.any(object_positions != DISCARD_PILE_COORDS, axis=-1))
        assert int(on_grid.sum()) == 12, f"{where}: expected 12 real objects total"

        real_positions = np.asarray(object_positions)[on_grid]
        player_pos = np.asarray(state.get_player().position)
        all_positions = np.concatenate([player_pos[None], real_positions], axis=0)
        # mutually distinct, and none on top of a door
        assert len(set(map(tuple, all_positions.tolist()))) == all_positions.shape[0], (
            f"{where}: player/object positions must be mutually distinct"
        )
        door_positions = set(map(tuple, np.asarray(doors.position).tolist()))
        for row, col in all_positions:
            assert (int(row), int(col)) not in door_positions, (
                f"{where}: entity at ({row},{col}) must not sit on a door cell"
            )
            assert int(grid[row, col]) == 0, f"{where}: entity at ({row},{col}) not on floor"


def test_playground_reward_free_and_no_early_termination():
    env = nx.make(ENV_ID)
    step_fn = jax.jit(env.step)
    for seed in range(N_SEEDS):
        timestep = env.reset(jax.random.PRNGKey(seed))
        action_keys = jax.random.split(jax.random.PRNGKey(seed + 1000), RANDOM_WALK_STEPS)
        for i, action_key in enumerate(action_keys):
            action = jax.random.randint(action_key, (), 0, 7)
            timestep = step_fn(timestep, action)
            assert float(timestep.reward) == 0.0, f"seed={seed} step={i}: expected reward-free"
            assert timestep.step_type == 0, (
                f"seed={seed} step={i}: expected no termination/truncation "
                f"within the first {RANDOM_WALK_STEPS} of {env.max_steps} steps"
            )


def test_playground_jit_vmap_compatible():
    env = nx.make(ENV_ID)
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    timestep = jax.vmap(reset)(keys)
    for action in range(7):
        timestep = jax.vmap(step, in_axes=(0, None))(timestep, jnp.asarray(action))
