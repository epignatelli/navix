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
from jax import Array
import jax.numpy as jnp

from . import events
from .states import State
from .grid import translate
from .entities import Entities, Player


def compose(
    *term_functions: Callable[[State, Array, State], Array],
    operator: Callable = jnp.any,
) -> Callable:
    return lambda prev_state, action, state: operator(
        jnp.asarray([term_f(prev_state, action, state) for term_f in term_functions])
    )


def check_truncation(terminated: Array, truncated: Array) -> Array:
    result = jnp.asarray(truncated + 2 * terminated, dtype=jnp.int32)
    return jnp.clip(result, 0, 2)


def on_goal_reached(prev_state: State, action: Array, state: State) -> Array:
    return jnp.asarray(events.on_goal_reached(state), dtype=jnp.bool_)


def on_lava_fall(prev_state: State, action: Array, state: State) -> Array:
    return jnp.asarray(events.on_lava_fall(state), dtype=jnp.bool_)


def on_ball_hit(prev_state: State, action: Array, state: State) -> Array:
    return jnp.asarray(events.on_ball_hit(state), dtype=jnp.bool_)


def on_door_done(prev_state: State, action: Array, state: State) -> Array:
    return jnp.asarray(events.on_door_done(state), dtype=jnp.bool_)


DEFAULT_TERMINATION = compose(on_goal_reached, on_lava_fall, on_ball_hit)
