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

"""`Memory` (issue #180): structural checks against the live reset
state, plus real `env.step()` gameplay covering both the success path
(walking to `state.mission`'s target) and the failure path (walking to
the mirrored wrong-target position)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

import navix as nx
from navix.components import DISCARD_PILE_COORDS


EAST, SOUTH, WEST, NORTH = 0, 1, 2, 3
ROTATE_CCW, ROTATE_CW, FORWARD = 0, 1, 2

N_SEEDS = 20
# real env.step() gameplay walks are the expensive part (each one is a
# fresh JIT compilation) - same finding as test_unlock.py's
# GAMEPLAY_SEEDS: keep these few, structural checks cover every seed
# cheaply instead.
GAMEPLAY_SEEDS = 2

FIXED_LENGTH_ENV_IDS = (
    "Navix-MemoryS13-v0",
    "Navix-MemoryS11-v0",
    "Navix-MemoryS9-v0",
    "Navix-MemoryS7-v0",
)
RANDOM_LENGTH_ENV_IDS = (
    "Navix-MemoryS17Random-v0",
    "Navix-MemoryS13Random-v0",
)
ALL_ENV_IDS = FIXED_LENGTH_ENV_IDS + RANDOM_LENGTH_ENV_IDS


def face(env, timestep, direction: int):
    for _ in range(4):
        if int(timestep.state.get_player().direction) == direction:
            return timestep
        timestep = env.step(timestep, jnp.asarray(ROTATE_CW))
    raise AssertionError(f"could not face direction {direction}")


def walk_to(env, timestep, row: int, col: int):
    # Memory's layout is a single-file corridor with no obstacles once
    # a direction is chosen, so a straight row-then-column walk (no
    # BFS needed) always reaches any on-path cell.
    player = timestep.state.get_player()
    dc = col - int(player.position[1])
    if dc > 0:
        timestep = face(env, timestep, EAST)
        for _ in range(dc):
            timestep = env.step(timestep, jnp.asarray(FORWARD))
    player = timestep.state.get_player()
    dr = row - int(player.position[0])
    if dr > 0:
        timestep = face(env, timestep, SOUTH)
        for _ in range(dr):
            timestep = env.step(timestep, jnp.asarray(FORWARD))
    elif dr < 0:
        timestep = face(env, timestep, NORTH)
        for _ in range(-dr):
            timestep = env.step(timestep, jnp.asarray(FORWARD))
    return timestep


def test_memory_structure():
    for env_id in ALL_ENV_IDS:
        env = nx.make(env_id)
        assert int(env.action_space.maximum) + 1 == 7, f"{env_id}: expected 7 actions"
        for seed in range(N_SEEDS):
            state = env.reset(jax.random.PRNGKey(seed)).state
            size = state.grid.shape[0]
            assert state.grid.shape == (size, size), f"{env_id} seed={seed}"

            player = state.get_player()
            assert int(player.position[0]) == size // 2, (
                f"{env_id} seed={seed}: player must start on the centre row"
            )
            assert int(player.direction) == EAST, (
                f"{env_id} seed={seed}: player must start facing east"
            )

            keys = state.entities[nx.entities.Entities.KEY]
            balls = state.entities[nx.entities.Entities.BALL]
            key_active = ~jnp.all(keys.position == DISCARD_PILE_COORDS, axis=-1)
            ball_active = ~jnp.all(balls.position == DISCARD_PILE_COORDS, axis=-1)
            assert bool(key_active[0]), f"{env_id} seed={seed}: hallway key must be active"
            assert bool(ball_active[0]), f"{env_id} seed={seed}: hallway ball must be active"
            # exactly one of {start-room key, start-room ball} is active
            assert (int(key_active[1]) + int(ball_active[1])) == 1, (
                f"{env_id} seed={seed}: start room must hold exactly one object"
            )

            mission_row = int(state.mission[0].position[0])
            assert mission_row in (size // 2 - 1, size // 2 + 1), (
                f"{env_id} seed={seed}: mission target must sit off the centre row"
            )


def test_memory_success_and_failure():
    for env_id in FIXED_LENGTH_ENV_IDS:
        for seed in range(GAMEPLAY_SEEDS):
            env = nx.make(env_id)
            timestep = env.reset(jax.random.PRNGKey(seed))
            state = timestep.state
            size = state.grid.shape[0]
            success_row, success_col = (int(x) for x in state.mission[0].position)
            failure_row = 2 * (size // 2) - success_row

            success_timestep = walk_to(env, timestep, success_row, success_col)
            assert success_timestep.step_type == 2, (
                f"{env_id} seed={seed}: expected termination on reaching the target"
            )
            assert float(success_timestep.reward) > 0, (
                f"{env_id} seed={seed}: expected positive reward on success"
            )

            failure_timestep = walk_to(env, timestep, failure_row, success_col)
            assert failure_timestep.step_type == 2, (
                f"{env_id} seed={seed}: expected termination on reaching the wrong target"
            )
            assert float(failure_timestep.reward) == 0, (
                f"{env_id} seed={seed}: expected zero reward on failure"
            )


def test_memory_random_length_varies_hallway():
    for env_id in RANDOM_LENGTH_ENV_IDS:
        env = nx.make(env_id)
        cols = {
            int(env.reset(jax.random.PRNGKey(seed)).state.mission[0].position[1])
            for seed in range(N_SEEDS)
        }
        assert len(cols) > 1, f"{env_id}: hallway length should vary across seeds"


def test_memory_jit_vmap_compatible():
    for env_id in ALL_ENV_IDS:
        env = nx.make(env_id)
        keys = jax.random.split(jax.random.PRNGKey(0), 4)
        reset = jax.jit(env.reset)
        step = jax.jit(env.step)
        timestep = jax.vmap(reset)(keys)
        for action in range(7):
            timestep = jax.vmap(step, in_axes=(0, None))(timestep, jnp.asarray(action))
        jax.block_until_ready(timestep)
