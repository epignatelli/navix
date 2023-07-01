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


from __future__ import annotations
from typing import Callable


import jax
import jax.numpy as jnp
from jax import Array

from .entities import State


def compose(*fns: Callable[[State, Array, State], Array]):
    def composed(prev_state: State, action: Array, state: State) -> Array:
        reward = jnp.asarray(0.0)
        for fn in fns:
            reward += fn(prev_state, action, state)
        return reward

    return composed


def free(state: State) -> Array:
    return jnp.asarray(0.0)


def navigation(prev_state: State, action: Array, state: State) -> Array:
    player = state.get_player()
    goals = state.get_goals()

    reached = jax.vmap(jnp.array_equal, in_axes=(None, 0))(
        player.position, goals.position
    )
    any_reached = jnp.sum(reached)

    draw = jax.random.uniform(state.key, ())
    reward = any_reached * jnp.greater_equal(draw, goals.probability)
    reward = jnp.asarray(reward, jnp.float32).squeeze()

    # make sure that reward is a scalar
    assert reward.shape == (), f"Reward must be a scalar but got shape {reward.shape}"
    return reward


def action_cost(
    prev_state: State, action: Array, new_state: State, cost: float = 0.01
) -> Array:
    # noops are free
    return -jnp.asarray(action > 0, dtype=jnp.float32) * cost


def time_cost(
    prev_state: State, action: Array, new_state: State, cost: float = 0.01
) -> Array:
    # time always has a cost
    return -jnp.asarray(cost)


def wall_hit_cost(
    prev_state: State, action: Array, state: State, cost: float = 0.01
) -> Array:
    prev_player = prev_state.get_player()
    player = state.get_player()

    # if state is unchanged, maybe the wall was hit
    didnt_move = jnp.array_equal(prev_player.position, player.position)
    but_wanted_to = jnp.less_equal(3, action) * jnp.less_equal(action, 6)
    hit = jnp.logical_and(didnt_move, but_wanted_to)
    return -jnp.asarray(cost) * hit
