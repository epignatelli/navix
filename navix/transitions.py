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


"""Transition functions: given `(state, action, action_set)`, produce the
next `State`.

An `Environment`'s `transitions_fn` is one of these. They all start by
applying the agent's action (`action_set[action]`, a `state -> state`
primitive - see `navix.actions`) and then optionally advance any
autonomous entities. `DEFAULT_TRANSITION` is `stochastic_transition`.
"""

from __future__ import annotations

from typing import Callable, Tuple
from jax import Array
import jax
import jax.numpy as jnp
from .entities import Entities, Ball
from .states import State
from .grid import positions_equal, translate


def deterministic_transition(
    state: State, action: Array, actions_set: Tuple[Callable[[State], State], ...]
) -> State:
    """Applies only the agent's action - nothing else in the world moves.

    Args:
        state (State): the current state, $s_t$.
        action (Array): a scalar integer action, `i32[]`, in
            `[0, len(actions_set))`.
        actions_set (tuple[Callable, ...]): the environment's
            `action_set`; `actions_set[action]` is applied via
            `jax.lax.switch`.

    Returns:
        State: $s_{t+1}$."""
    return jax.lax.switch(action, actions_set, state)


def stochastic_transition(
    state: State, action: Array, actions_set: Tuple[Callable[[State], State], ...]
) -> State:
    """Applies the agent's action, then moves every `Ball` one random
    step (`update_balls`). This is `DEFAULT_TRANSITION` - environments
    without balls behave identically to `deterministic_transition`.

    Args:
        state (State): the current state, $s_t$.
        action (Array): a scalar integer action, `i32[]`, in
            `[0, len(actions_set))`.
        actions_set (tuple[Callable, ...]): the environment's
            `action_set`.

    Returns:
        State: $s_{t+1}$, with balls advanced."""
    # actions
    state = jax.lax.switch(action, actions_set, state)

    state = update_balls(state)
    return state


def update_balls(state: State) -> State:
    """Moves every `Ball` one cell in a uniformly random direction, or
    leaves it in place if that cell is blocked. A ball that would move
    onto the player instead stays put and records a
    `(BALL, HIT)` event (used by `rewards.on_ball_hit` /
    `terminations.on_ball_hit`).

    Args:
        state (State): the current state. If it has no `Ball` entities
            this is a no-op.

    Returns:
        State: the state with ball positions and ball-hit events
        updated, and `state.key` advanced."""
    def update_one(ball: Ball, key: Array) -> Tuple[Array, Array]:
        direction = jax.random.randint(key, (), minval=0, maxval=4)
        new_position = translate(ball.position, direction)
        new_ball = ball.replace(position=new_position)
        can_move, hit = _can_spawn_there(state, new_ball)
        return jnp.where(can_move, new_ball.position, ball.position), hit

    if Entities.BALL in state.entities:
        balls: Ball = state.entities[Entities.BALL]  # type: ignore
        keys = jax.random.split(state.key, len(balls.position) + 1)
        # hit is one boolean per ball, computed independently for every
        # ball in the same vmapped call - merged into events.events in one
        # shot below, so two different balls hitting the player the same
        # step both survive (see EventsManager.record_ball_hit/merge_event,
        # issue #139), instead of collapsing to only the first one found.
        new_position, hit = jax.jit(jax.vmap(update_one))(balls, keys[1:])
        # update balls
        balls = balls.replace(position=new_position)
        state = state.set_balls(balls)
        # update events
        events = state.events.record_ball_hit(balls, hit)
        state = state.replace(key=keys[0], events=events)
    return state


def _can_spawn_there(state: State, ball: Ball) -> Tuple[Array, Array]:
    # according to the grid
    walkable = jnp.equal(state.grid[tuple(ball.position)], 0)

    # according to entities
    hit = jnp.asarray(False)
    entities = state.entities
    for k in state.entities:
        obstructs = positions_equal(entities[k].position, ball.position)[0]
        if k == Entities.PLAYER:
            hit = jnp.logical_or(hit, obstructs)
        walkable = jnp.logical_and(walkable, jnp.any(jnp.logical_not(obstructs)))
    return jnp.asarray(walkable, dtype=jnp.bool_), hit


DEFAULT_TRANSITION = stochastic_transition
"""The `transitions_fn` an `Environment` uses unless overridden:
`stochastic_transition` (agent action + random ball motion)."""
