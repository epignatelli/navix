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


def compose(
    *term_functions: Callable[[State, Array, State], Array],
    operator: Callable = jnp.any,
) -> Callable:
    """Compose termination functions into a single termination function.

    Args:
        *term_functions (Callable): List of termination functions.
        operator (Callable): Operator to combine the termination functions.

    Returns:
        Callable: A single termination function."""
    return lambda prev_state, action, state: operator(
        jnp.asarray([term_f(prev_state, action, state) for term_f in term_functions])
    )


def check_truncation(terminated: Array, truncated: Array) -> Array:
    """Check if the episode is truncated or terminated, and returns a value
    that conforms to the `StepType` enum.

    Args:
        terminated (Array): A boolean array indicating whether the episode is terminated.
        truncated (Array): A boolean array indicating whether the episode is truncated.

    Returns:
        Array: An integer array that represents the step type."""
    result = jnp.asarray(truncated + 2 * terminated, dtype=jnp.int32)
    return jnp.clip(result, 0, 2)


def on_goal_reached(prev_state: State, action: Array, state: State) -> Array:
    """Check if the goal has been reached using the `goal_reached` event.

    Args:
        prev_state (State): The previous state of the game.
        action (Array): The action taken by the player.
        state (State): The current state of the game.

    Returns:
        Array: A boolean array indicating whether the goal has been reached."""
    return jnp.asarray(events.on_goal_reached(state), dtype=jnp.bool_)


def on_lava_fall(prev_state: State, action: Array, state: State) -> Array:
    """Check if the lava has fallen using the `lava_fall` event.

    Args:
        prev_state (State): The previous state of the game.
        action (Array): The action taken by the player.
        state (State): The current state of the game.

    Returns:
        Array: A boolean array indicating whether the lava has fallen."""
    return jnp.asarray(events.on_lava_fall(state), dtype=jnp.bool_)


def on_ball_hit(prev_state: State, action: Array, state: State) -> Array:
    """Check if the ball has hit something using the `ball_hit` event.

    Args:
        prev_state (State): The previous state of the game.
        action (Array): The action taken by the player.
        state (State): The current state of the game.

    Returns:
        Array: A boolean array indicating whether the ball has hit something."""
    return jnp.asarray(events.on_ball_hit(state), dtype=jnp.bool_)


def on_door_done(prev_state: State, action: Array, state: State) -> Array:
    """Check if the action `done` has been called in front of a `Door` object with the \
        correct colour.

    Args:
        prev_state (State): The previous state of the game.
        action (Array): The action taken by the player.
        state (State): The current state of the game.

    Returns:
        Array: A boolean array indicating whether the action `done` has been called in \
        front of a `Door` object with the correct colour.
    """
    return jnp.asarray(events.on_door_done(state), dtype=jnp.bool_)


DEFAULT_TERMINATION = compose(on_goal_reached, on_lava_fall, on_ball_hit)
