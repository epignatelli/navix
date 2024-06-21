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
from .entities import Entities, Ball
from .states import EventsManager, State
from .grid import positions_equal, translate


def deterministic_transition(
    state: State, action: Array, actions_set: Tuple[Callable[[State], State], ...]
) -> State:
    """Deterministic transition function. It selects the action from the set of actions
    and applies it to the state.
    
    Args:
        state (State): The current state of the game.
        action (Array): The action to be taken.
        actions_set (Tuple[Callable[[State], State]): A set of actions that can be taken.
    
    Returns:
        State: The new state of the game."""
    return jax.lax.switch(action, actions_set, state)


def stochastic_transition(
    state: State, action: Array, actions_set: Tuple[Callable[[State], State], ...]
) -> State:
    """Stochastic transition function. It selects the action from the set of actions
    and applies it to the state, and updates entities that have stochastic transitions,
    such as balls.
    
    Args:
        state (State): The current state of the game.
        action (Array): The action to be taken.
        actions_set (Tuple[Callable[[State], State]): A set of actions that can be taken.
    
    Returns:
        State: The new state of the game."""
    # actions
    state = jax.lax.switch(action, actions_set, state)

    state = update_balls(state)
    return state


def update_balls(state: State) -> State:
    """Update the position of the balls in the game.
    Balls move in a random direction if they can, otherwise they stay in place.
    
    Args:
        state (State): The current state of the game.
    
    Returns:
        State: The new state of the game."""
    def update_one(ball: Ball, key: Array) -> Tuple[Array, EventsManager]:
        direction = jax.random.randint(key, (), minval=0, maxval=4)
        new_position = translate(ball.position, direction)
        new_ball = ball.replace(position=new_position)
        can_move, events = _can_spawn_there(state, new_ball)
        return jnp.where(can_move, new_ball.position, ball.position), events

    if Entities.BALL in state.entities:
        balls: Ball = state.entities[Entities.BALL]  # type: ignore
        keys = jax.random.split(state.key, len(balls.position) + 1)
        new_position, new_events = jax.jit(jax.vmap(update_one))(balls, keys[1:])
        # update balls
        balls = balls.replace(position=new_position)
        state = state.set_balls(balls)
        # update events
        # take only the first happened event (even if happened already)
        idx = jnp.where(new_events.ball_hit.happened, size=1)[0][0]  # scalar
        ball_hits = jax.tree.map(lambda x: x[idx], new_events.ball_hit)
        events = state.events.replace(ball_hit=ball_hits)
        state = state.replace(key=keys[0], events=events)
    return state


def _can_spawn_there(state: State, ball: Ball) -> Tuple[Array, EventsManager]:
    # according to the grid
    walkable = jnp.equal(state.grid[tuple(ball.position)], 0)

    # according to entities
    events = state.events
    entities = state.entities
    for k in state.entities:
        obstructs = positions_equal(entities[k].position, ball.position)[0]
        if k == Entities.PLAYER:
            events = jax.lax.cond(
                obstructs,
                lambda x: x.record_ball_hit(ball),
                lambda x: x,
                events,
            )
        walkable = jnp.logical_and(walkable, jnp.any(jnp.logical_not(obstructs)))
    return jnp.asarray(walkable, dtype=jnp.bool_), events


DEFAULT_TRANSITION = stochastic_transition
