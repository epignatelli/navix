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


def on_ball_pickup(prev_state: State, action: Array, state: State) -> Array:
    """Check if any ball was picked up this step, using the `ball_pickup`
    event - for environments (e.g. `DynamicObstacles`) where `Ball`
    represents a hazard to avoid touching, not an object to carry:
    composing this alongside `on_ball_hit` makes picking one up end the
    episode the same way walking into it already does, rather than
    silently removing it from play now that `Ball` is `Pickable`.

    Args:
        prev_state (State): The previous state of the game.
        action (Array): The action taken by the player.
        state (State): The current state of the game.

    Returns:
        Array: A boolean array indicating whether any ball was picked up."""
    return jnp.asarray(events.on_ball_pickup(state), dtype=jnp.bool_)


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


def on_door_open(prev_state: State, action: Array, state: State) -> Array:
    """Check if any door was opened this step, using the `door_opening`
    event.

    Args:
        prev_state (State): The previous state of the game.
        action (Array): The action taken by the player.
        state (State): The current state of the game.

    Returns:
        Array: A boolean array indicating whether any door was opened."""
    return jnp.asarray(events.on_door_open(state), dtype=jnp.bool_)


def on_box_pickup(prev_state: State, action: Array, state: State) -> Array:
    """Check if any box was picked up this step, using the `box_pickup`
    event.

    Args:
        prev_state (State): The previous state of the game.
        action (Array): The action taken by the player.
        state (State): The current state of the game.

    Returns:
        Array: A boolean array indicating whether any box was picked up."""
    return jnp.asarray(events.on_box_pickup(state), dtype=jnp.bool_)


def on_ordered_doors_resolved(prev_state: State, action: Array, state: State) -> Array:
    """`RedBlueDoors`' termination: ends the episode as soon as blue
    opens, whether that was a success (red already open) or a failure
    (blue opened first).

    Args:
        prev_state (State): The previous state of the game.
        action (Array): The action taken by the player.
        state (State): The current state of the game.

    Returns:
        Array: A boolean array."""
    success = events.on_ordered_doors_success(prev_state, state)
    failure = events.on_ordered_doors_failure(prev_state, state)
    return jnp.asarray(jnp.logical_or(success, failure), dtype=jnp.bool_)


def on_target_done(prev_state: State, action: Array, state: State) -> Array:
    """`GoToObject`'s success termination: `done` was called while
    facing the mission target.

    Args:
        prev_state (State): The previous state of the game.
        action (Array): The action taken by the player.
        state (State): The current state of the game.

    Returns:
        Array: A boolean array."""
    return jnp.asarray(events.on_target_done(action, state), dtype=jnp.bool_)


def on_wrong_toggle(prev_state: State, action: Array, state: State) -> Array:
    """`GoToObject`'s failure termination: the `toggle` action was used
    at all.

    Args:
        prev_state (State): The previous state of the game.
        action (Array): The action taken by the player.
        state (State): The current state of the game.

    Returns:
        Array: A boolean array."""
    return jnp.asarray(events.on_wrong_toggle(action), dtype=jnp.bool_)


def on_any_target_pickup(prev_state: State, action: Array, state: State) -> Array:
    """`Fetch`'s termination: any `Key`/`Ball` pickup this step ends the
    episode, right or wrong (see `rewards.on_target_fetched` for which
    one determines the reward).

    Args:
        prev_state (State): The previous state of the game.
        action (Array): The action taken by the player.
        state (State): The current state of the game.

    Returns:
        Array: A boolean array."""
    return jnp.asarray(events.on_any_target_pickup(state), dtype=jnp.bool_)


def on_target_fetched(prev_state: State, action: Array, state: State) -> Array:
    """`ObstructedMaze`'s termination: only picking up the *specific*
    mission-target `Key`/`Ball` ends the episode - unlike `Fetch`'s
    `on_any_target_pickup`, a wrong pickup (the blocking `Ball`, say)
    does not end it here, matching MiniGrid's actual `ObstructedMazeEnv`
    (reward + termination only on picking up the target).

    Args:
        prev_state (State): The previous state of the game.
        action (Array): The action taken by the player.
        state (State): The current state of the game.

    Returns:
        Array: A boolean array."""
    return jnp.asarray(events.on_target_fetched(state), dtype=jnp.bool_)


def on_put_near_wrong_pickup(prev_state: State, action: Array, state: State) -> Array:
    """`PutNear`'s failure-on-pickup termination: the wrong object was
    picked up this step.

    Args:
        prev_state (State): The previous state of the game.
        action (Array): The action taken by the player.
        state (State): The current state of the game.

    Returns:
        Array: A boolean array."""
    return jnp.asarray(events.on_put_near_wrong_pickup(state), dtype=jnp.bool_)


def on_put_near_drop_attempted(prev_state: State, action: Array, state: State) -> Array:
    """`PutNear`'s termination on a genuine drop attempt (success or
    failure - see `rewards.on_put_near_success` for which one this was).

    Args:
        prev_state (State): The previous state of the game.
        action (Array): The action taken by the player.
        state (State): The current state of the game.

    Returns:
        Array: A boolean array."""
    return jnp.asarray(
        events.on_put_near_drop_attempted(prev_state, action), dtype=jnp.bool_
    )


def on_memory_success(prev_state: State, action: Array, state: State) -> Array:
    """Check if the player reached `Memory`'s target position, using
    the `on_memory_success` event.

    Args:
        prev_state (State): The previous state of the game.
        action (Array): The action taken by the player.
        state (State): The current state of the game.

    Returns:
        Array: A boolean array indicating whether the target was reached."""
    return jnp.asarray(events.on_memory_success(state), dtype=jnp.bool_)


def on_memory_failure(prev_state: State, action: Array, state: State) -> Array:
    """Check if the player reached `Memory`'s wrong (failure) position,
    using the `on_memory_failure` event.

    Args:
        prev_state (State): The previous state of the game.
        action (Array): The action taken by the player.
        state (State): The current state of the game.

    Returns:
        Array: A boolean array indicating whether the wrong position was reached."""
    return jnp.asarray(events.on_memory_failure(state), dtype=jnp.bool_)


DEFAULT_TERMINATION = compose(on_goal_reached, on_lava_fall, on_ball_hit)
