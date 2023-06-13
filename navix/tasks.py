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


import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from chex import Array

from .components import State
from .grid import mask_entity


def navigation(state: State, prob: ArrayLike = 1.0) -> Array:
    player_mask = mask_entity(state.grid, state.entities["player/0"].id)
    goal_mask = mask_entity(state.grid, state.entities["goal/0"].id)
    condition = jax.random.uniform(state.key, ()) >= prob
    return jax.lax.cond(
        condition,
        lambda _: jnp.sum(player_mask * goal_mask),
        lambda _: jnp.asarray(0.0),
        (),
    )


def free(state: State) -> Array:
    return jnp.asarray(0.0)
