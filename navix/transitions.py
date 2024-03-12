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

from typing import Callable, Dict, Tuple
from jax import Array
import jax
import jax.numpy as jnp
from .entities import State, Entities, Ball
from .grid import positions_equal, translate
from .actions import _walkable


def deterministic_transition(
    state: State, action: Array, actions_set: Tuple[Callable[[State], State], ...]
) -> State:
    return jax.lax.switch(action, actions_set, state)


def stochastic_transition(
    state: State, action: Array, actions_set: Tuple[Callable[[State], State], ...]
) -> State:
    # actions
    state = jax.lax.switch(action, actions_set, state)

    state = update_balls(state)
    return state


def update_balls(state: State) -> State:
    def update_one(position, key):
        direction = jax.random.randint(key, (), minval=0, maxval=4)
        new_position = translate(position, direction)
        can_move = _can_move_there(state, new_position)
        return jnp.where(can_move, new_position, position)

    if Entities.BALL in state.entities:
        balls: Ball = state.entities[Entities.BALL]  # type: ignore
        keys = jax.random.split(state.key, len(balls.position) + 1)
        new_position = jax.jit(jax.vmap(update_one))(balls.position, keys[1:])
        balls = balls.replace(position=new_position)
        state.entities[Entities.BALL] = balls
        state = state.replace(key=keys[0])
    return state


def _can_move_there(state: State, position: Array) -> Array:
    # according to the grid
    walkable = jnp.equal(state.grid[tuple(position)], 0)

    # according to entities
    for k in state.entities:
        obstructs = positions_equal(state.entities[k].position, position)
        walkable = jnp.logical_and(walkable, jnp.any(jnp.logical_not(obstructs)))
    return jnp.asarray(walkable, dtype=jnp.bool_)