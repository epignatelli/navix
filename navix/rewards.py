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


import jax.numpy as jnp
from jax import Array

from . import events
from .states import State


def compose(
    *reward_functions: Callable[[State, Array, State], Array],
    operator: Callable = jnp.sum,
) -> Callable:
    return lambda prev_state, action, state: operator(
        jnp.asarray(
            [f(prev_state, action, state) for f in reward_functions], dtype=jnp.float32
        )
    )


def free(state: State) -> Array:
    return jnp.asarray(0.0, dtype=jnp.float32)


def on_goal_reached(prev_state: State, action: Array, state: State) -> Array:
    return jnp.asarray(events.on_goal_reached(state), dtype=jnp.float32)


def action_cost(
    prev_state: State, action: Array, new_state: State, cost: float = 0.01
) -> Array:
    # noops are free
    return -jnp.asarray(action > 0, dtype=jnp.float32) * cost


def time_cost(
    prev_state: State, action: Array, new_state: State, cost: float = 0.01
) -> Array:
    # time always has a cost
    return -jnp.asarray(cost, dtype=jnp.float32)


def wall_hit_cost(
    prev_state: State, action: Array, state: State, cost: float = 0.01
) -> Array:
    return jnp.asarray(events.on_wall_hit(state), dtype=jnp.float32) * cost


def on_door_done(
    prev_state: State, action: Array, state: State, cost: float = 0.01
) -> Array:
    return jnp.asarray(events.on_door_done(state), dtype=jnp.float32)


DEFAULT_TASK = compose(on_goal_reached, action_cost)
