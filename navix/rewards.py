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
    """Compose multiple reward functions into a single reward function.
    The functions are called in order and the results are reduced using the `operator` \
    function.
    
    Args:
        *reward_functions (Callable[[State, Array, State], Array]): A list of reward functions.
        operator (Callable): The operator to reduce the results of the reward functions.
        It must be a function that takes a list of arrays, or an array and returns an \
        array of size `f32[]`.
    
    Returns:
        Callable: A composed reward function that applies the `operator` to the results of the \
        reward functions."""
    return lambda prev_state, action, state: operator(
        jnp.asarray(
            [f(prev_state, action, state) for f in reward_functions], dtype=jnp.float32
        )
    )


def free(state: State) -> Array:
    """A reward function that always returns 0, to simulate reward-free learning.

    Args:
        state (State): The current state of the game.

    Returns:
        Array: A scalar array `f32[]` with value 0."""
    return jnp.asarray(0.0, dtype=jnp.float32)


def on_goal_reached(prev_state: State, action: Array, state: State) -> Array:
    """A reward function that returns 1 when the goal is reached, and 0 otherwise.

    Args:
        state (State): The current state of the game.

    Returns:
        Array: A scalar array `f32[]` with value 1 if the goal is reached, and 0 otherwise.
    """
    return jnp.asarray(events.on_goal_reached(state), dtype=jnp.float32)


def action_cost(
    prev_state: State, action: Array, new_state: State, cost: float = 0.01
) -> Array:
    """A reward function that returns a negative value when an action is taken. 
    All actions have a cost of `cost`, except for noops.
    
    Args:
        prev_state (State): The previous state of the game.
        action (Array): The action taken.
        new_state (State): The new state of the game.
        cost (float): The cost of taking an action.
    
    Returns:
        Array: A scalar array `f32[]` with value -`cost` if the action is not a noop, \
        and 0 otherwise."""
    # noops are free
    return -jnp.asarray(action != 6, dtype=jnp.float32) * cost


def time_cost(
    prev_state: State, action: Array, new_state: State, cost: float = 0.01
) -> Array:
    """A reward function that returns a negative value as time passes, paying a cost \
    of `cost` at each time step.

    Args:
        prev_state (State): The previous state of the game.
        action (Array): The action taken.
        new_state (State): The new state of the game.
        cost (float): The cost of time passing.
    
    Returns:
        Array: A scalar array `f32[]` with value -`cost`.
    """
    # time always has a cost
    return -jnp.asarray(cost, dtype=jnp.float32)


def wall_hit_cost(
    prev_state: State, action: Array, state: State, cost: float = 0.01
) -> Array:
    """A reward function that returns a negative value when the agent hits a wall, \
    paying a cost of `cost` for each wall hit.
    
    Args:
        state (State): The current state of the game.
        cost (float): The cost of hitting a wall.
    
    Returns:
        Array: A scalar array `f32[]` with value -`cost` if the agent hits a wall, \
        and 0 otherwise."""
    return jnp.asarray(events.on_wall_hit(state), dtype=jnp.float32) * cost


def on_door_done(prev_state: State, action: Array, state: State) -> Array:
    """A reward function that returns a positive value when the agent uses the action \
    `done` in front of a door.
    
    Args:
        state (State): The current state of the game.
        
    Returns:
        Array: A scalar array `f32[]` with value 1 if the agent uses the action `done` in \
        front of a door, and 0 otherwise."""

    return jnp.asarray(events.on_door_done(state), dtype=jnp.float32)


DEFAULT_TASK = compose(on_goal_reached, action_cost)
"""The default task for the game, composed of the `on_goal_reached` and `action_cost` reward functions."""
