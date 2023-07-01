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

from jax import Array
import jax.numpy as jnp
from .entities import State
from .grid import positions_equal


def check_truncation(terminated: Array, truncated: Array) -> Array:
    result = jnp.asarray(truncated + 2 * terminated, dtype=jnp.int32)
    return jnp.clip(result, 0, 2)


def on_navigation_completion(prev_state: State, action: Array, state: State) -> Array:
    player = state.get_player()
    goals = state.get_goals()

    reached = positions_equal(player.position, goals.position)
    return jnp.any(reached)
