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

from typing import Callable, Tuple
from jax import Array
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from .entities import Entities, Ball
from .states import EventsManager, State
from .grid import positions_equal, translate


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
        can_move, events = _can_spawn_there(state, new_position)
        return jnp.where(can_move, new_position, position), events

    if Entities.BALL in state.entities:
        balls: Ball = state.entities[Entities.BALL]  # type: ignore
        keys = jax.random.split(state.key, len(balls.position) + 1)
        new_position, events = jax.jit(jax.vmap(update_one))(balls.position, keys[1:])
        # update structs
        balls = balls.replace(position=new_position)
        state = state.set_balls(balls)
        events = jtu.tree_map(lambda x: jnp.any(x), events)
        state = state.replace(key=keys[0], events=events)
    return state


def _can_spawn_there(state: State, position: Array) -> Tuple[Array, EventsManager]:
    # according to the grid
    walkable = jnp.equal(state.grid[tuple(position)], 0)

    # according to entities
    events = state.events
    for k in state.entities:
        obstructs = positions_equal(state.entities[k].position, position)
        if k == Entities.PLAYER:
            events = jax.lax.cond(
                jnp.any(obstructs),
                lambda x: x.record_ball_hit(state.entities[k], position),
                lambda x: x,
                events,
            )
        walkable = jnp.logical_and(walkable, jnp.any(jnp.logical_not(obstructs)))
    return jnp.asarray(walkable, dtype=jnp.bool_), events


DEFAULT_TRANSITION = stochastic_transition
