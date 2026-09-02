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

"""`RedBlueDoors` (issue #172): structural checks against the live reset
state, plus real `env.step()` gameplay covering both the success path
(red before blue) and the failure path (blue first)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

import navix as nx
from navix.rendering.registry import PALETTE


EAST, SOUTH, WEST, NORTH = 0, 1, 2, 3
ROTATE_CW, FORWARD, TOGGLE = 1, 2, 5

N_SEEDS = 20


def face(env, timestep, direction: int):
    for _ in range(4):
        if int(timestep.state.get_player().direction) == direction:
            return timestep
        timestep = env.step(timestep, jnp.asarray(ROTATE_CW))
    raise AssertionError(f"could not face direction {direction}")


def walk_to(env, timestep, row: int, col: int):
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
    player = timestep.state.get_player()
    dc = col - int(player.position[1])
    if dc > 0:
        timestep = face(env, timestep, EAST)
        for _ in range(dc):
            timestep = env.step(timestep, jnp.asarray(FORWARD))
    elif dc < 0:
        timestep = face(env, timestep, WEST)
        for _ in range(-dc):
            timestep = env.step(timestep, jnp.asarray(FORWARD))
    return timestep


def test_red_blue_doors_structure():
    for env_id in ("Navix-RedBlueDoors-6x6-v0", "Navix-RedBlueDoors-8x8-v0"):
        env = nx.make(env_id)
        for seed in range(N_SEEDS):
            state = env.reset(jax.random.PRNGKey(seed)).state
            doors = state.get_doors()
            player = state.get_player()

            assert doors.position.shape[0] == 2, f"{env_id} seed={seed}: expected 2 doors"
            # fixed construction order: index 0 = red, index 1 = blue -
            # events.on_ordered_doors_* rely on this.
            assert int(doors.colour[0]) == int(PALETTE.RED)
            assert int(doors.colour[1]) == int(PALETTE.BLUE)
            assert int(doors.position[0, 1]) == int(doors.position[1, 1]), (
                f"{env_id} seed={seed}: both doors must sit in the same dividing wall"
            )
            assert int(doors.position[0, 0]) != int(doors.position[1, 0]), (
                f"{env_id} seed={seed}: doors must be at distinct rows"
            )
            assert not bool(doors.open[0]) and not bool(doors.open[1]), (
                f"{env_id} seed={seed}: doors should start closed"
            )
            assert int(doors.requires[0]) == -1 and int(doors.requires[1]) == -1, (
                f"{env_id} seed={seed}: doors should start unlocked"
            )
            wall_col = int(doors.position[0, 1])
            assert int(player.position[1]) < wall_col, (
                f"{env_id} seed={seed}: player must start in the left chamber"
            )


def test_red_blue_doors_success_red_then_blue():
    for seed in range(N_SEEDS):
        env = nx.make("Navix-RedBlueDoors-6x6-v0")
        timestep = env.reset(jax.random.PRNGKey(seed))
        doors = timestep.state.get_doors()
        red_row, wall_col = int(doors.position[0, 0]), int(doors.position[0, 1])
        blue_row = int(doors.position[1, 0])

        # approach each door from the left chamber (one cell west of it)
        timestep = walk_to(env, timestep, red_row, wall_col - 1)
        timestep = face(env, timestep, EAST)
        timestep = env.step(timestep, jnp.asarray(TOGGLE))
        assert bool(timestep.state.get_doors().open[0]), f"seed={seed}: red door did not open"
        assert timestep.step_type == 0, f"seed={seed}: episode ended after opening only red"

        timestep = walk_to(env, timestep, blue_row, wall_col - 1)
        timestep = face(env, timestep, EAST)
        timestep = env.step(timestep, jnp.asarray(TOGGLE))
        assert bool(timestep.state.get_doors().open[1]), f"seed={seed}: blue door did not open"
        assert timestep.step_type == 2, (
            f"seed={seed}: expected termination once blue opens with red already open"
        )
        assert float(timestep.reward) > 0, f"seed={seed}: expected positive reward"


def test_red_blue_doors_failure_blue_first():
    for seed in range(N_SEEDS):
        env = nx.make("Navix-RedBlueDoors-6x6-v0")
        timestep = env.reset(jax.random.PRNGKey(seed))
        doors = timestep.state.get_doors()
        blue_row, wall_col = int(doors.position[1, 0]), int(doors.position[1, 1])

        timestep = walk_to(env, timestep, blue_row, wall_col - 1)
        timestep = face(env, timestep, EAST)
        timestep = env.step(timestep, jnp.asarray(TOGGLE))
        assert bool(timestep.state.get_doors().open[1]), f"seed={seed}: blue door did not open"
        assert timestep.step_type == 2, f"seed={seed}: expected termination on opening blue first"
        assert float(timestep.reward) == 0, f"seed={seed}: expected zero reward for opening blue first"


def test_red_blue_doors_jit_vmap_compatible():
    for env_id in ("Navix-RedBlueDoors-6x6-v0", "Navix-RedBlueDoors-8x8-v0"):
        env = nx.make(env_id)
        keys = jax.random.split(jax.random.PRNGKey(0), 4)
        reset = jax.jit(env.reset)
        step = jax.jit(env.step)
        timestep = jax.vmap(reset)(keys)
        for action in range(len(nx.actions.DEFAULT_ACTION_SET)):
            timestep = jax.vmap(step, in_axes=(0, None))(timestep, jnp.asarray(action))
        jax.block_until_ready(timestep)
