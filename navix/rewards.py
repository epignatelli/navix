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
"""Reward functions: the scalar reward for one transition.

An `Environment`'s `reward_fn` has the signature shared with
`navix.terminations` and `navix.events`:

    fn(prev_state: State, action: Array, state: State) -> Array

- `prev_state` is $s_t$, `action` is $a_t$, `state` is $s_{t+1}$. Some
  functions need `prev_state` to reward a *change* this step; most only
  read `state`.
- The return is a scalar `f32[]`. `Environment.reward_space` bounds it
  (`[-1, 1]` by default). Positive functions here return `1.0` on the
  rewarded event and `0.0` otherwise; the `*_cost` functions return a
  small negative shaping term every step.

`compose` reduces several into one (summed by default), and
`DEFAULT_TASK` is `on_goal_reached` minus a per-step `action_cost`.
"""

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
    """Combines several reward functions into one.

    Args:
        *reward_functions (Callable): reward functions to combine, each
            `(prev_state, action, state) -> f32[]`.
        operator (Callable): reduces the stacked `f32[len(reward_functions)]`
            results to a scalar `f32[]`. Default `jnp.sum` - the rewards
            add up (use `on_goal_reached` for `+1` and an `action_cost`
            for the `-` shaping term, as `DEFAULT_TASK` does).

    Returns:
        Callable: a single `(prev_state, action, state) -> f32[]`
        function."""
    return lambda prev_state, action, state: operator(
        jnp.asarray(
            [f(prev_state, action, state) for f in reward_functions], dtype=jnp.float32
        )
    )


def free(prev_state: State, action: Array, state: State) -> Array:
    """Always `0.0` - the reward-free setting, for unsupervised or
    exploration-driven training where only the transition dynamics
    matter.

    Returns:
        Array: `f32[]`, always `0.0`."""
    return jnp.asarray(0.0, dtype=jnp.float32)


def on_goal_reached(prev_state: State, action: Array, state: State) -> Array:
    """A reward function that returns 1 when the goal is reached, and 0 otherwise.

    Args:
        state (State): The current state of the game.

    Returns:
        Array: A scalar array `f32[]` with value 1 if the goal is reached, and 0 otherwise.
    """
    return jnp.asarray(events.on_goal_reached(prev_state, action, state), dtype=jnp.float32)


def action_cost(
    prev_state: State, action: Array, new_state: State, cost: float = 0.01
) -> Array:
    """A per-step penalty of `-cost` on every action except the one at
    index `6` (the `done` action in the default `MINIGRID_ACTION_SET`),
    which is free. Part of `DEFAULT_TASK`, where it makes shorter
    successful episodes score higher.

    Args:
        prev_state (State): $s_t$ (unused).
        action (Array): the integer action taken.
        new_state (State): $s_{t+1}$ (unused).
        cost (float): the per-action penalty magnitude. Default `0.01`.

    Returns:
        Array: `f32[]` - `-cost` if `action != 6`, else `0.0`."""
    # noops are free
    return -jnp.asarray(action != 6, dtype=jnp.float32) * cost


def time_cost(
    prev_state: State, action: Array, new_state: State, cost: float = 0.01
) -> Array:
    """A flat `-cost` on every step, regardless of the action (unlike
    `action_cost`, which exempts `done`).

    Args:
        prev_state (State): $s_t$ (unused).
        action (Array): the integer action taken (unused).
        new_state (State): $s_{t+1}$ (unused).
        cost (float): the per-step penalty magnitude. Default `0.01`.

    Returns:
        Array: `f32[]`, always `-cost`.
    """
    # time always has a cost
    return -jnp.asarray(cost, dtype=jnp.float32)


def wall_hit_cost(
    prev_state: State, action: Array, state: State, cost: float = 0.01
) -> Array:
    """`-cost` on any step where the player moved into a wall this step
    (detected via `events.on_wall_hit`), `0.0` otherwise. Opt-in shaping
    for tasks that want to discourage bumping walls; same sign convention
    as `action_cost` / `time_cost`.

    Args:
        prev_state (State): $s_t$ (unused).
        action (Array): the integer action taken (unused).
        state (State): $s_{t+1}$ - read for the wall-hit event.
        cost (float): the penalty magnitude on a wall hit. Default
            `0.01`.

    Returns:
        Array: `f32[]` - `-cost` on a wall hit, else `0.0`."""
    return (
        -jnp.asarray(events.on_wall_hit(prev_state, action, state), dtype=jnp.float32)
        * cost
    )


def on_door_done(prev_state: State, action: Array, state: State) -> Array:
    """A reward function that returns a positive value when the agent uses the action \
    `done` in front of a door.
    
    Args:
        state (State): The current state of the game.
        
    Returns:
        Array: A scalar array `f32[]` with value 1 if the agent uses the action `done` in \
        front of a door, and 0 otherwise."""

    return jnp.asarray(events.on_door_done(prev_state, action, state), dtype=jnp.float32)


def on_door_open(prev_state: State, action: Array, state: State) -> Array:
    """A reward function that returns 1 when any door is opened this
    step, and 0 otherwise - unlike `on_door_done`, no `state.mission`
    target is needed; any door opening counts.

    Args:
        state (State): The current state of the game.

    Returns:
        Array: A scalar array `f32[]` with value 1 if a door was opened,
        and 0 otherwise."""
    return jnp.asarray(events.on_door_open(prev_state, action, state), dtype=jnp.float32)


def on_box_pickup(prev_state: State, action: Array, state: State) -> Array:
    """A reward function that returns 1 when any box is picked up this
    step, and 0 otherwise.

    Args:
        state (State): The current state of the game.

    Returns:
        Array: A scalar array `f32[]` with value 1 if a box was picked
        up, and 0 otherwise."""
    return jnp.asarray(events.on_box_pickup(prev_state, action, state), dtype=jnp.float32)


def on_ordered_doors_success(prev_state: State, action: Array, state: State) -> Array:
    """`RedBlueDoors`' reward: 1 if the blue door was opened this step
    while red was already open, 0 otherwise (including the failure case
    of opening blue first).

    Args:
        prev_state (State): The previous state of the game.
        action (Array): The action taken by the player.
        state (State): The current state of the game.

    Returns:
        Array: A scalar array `f32[]`."""
    return jnp.asarray(
        events.on_ordered_doors_success(prev_state, action, state), dtype=jnp.float32
    )


def on_target_done(prev_state: State, action: Array, state: State) -> Array:
    """`GoToObject`'s reward: `1.0` if the `done` action was taken while
    the player is orthogonally adjacent to the mission's target object
    (facing it is not required; see `events.on_target_done`), else
    `0.0`.

    Returns:
        Array: `f32[]`."""
    return jnp.asarray(events.on_target_done(prev_state, action, state), dtype=jnp.float32)


def on_target_fetched(prev_state: State, action: Array, state: State) -> Array:
    """`Fetch`'s reward: 1 if the mission's target object was the one
    picked up this step, 0 otherwise (including picking up the wrong
    one, which still ends the episode via `terminations.
    on_any_target_pickup`).

    Args:
        prev_state (State): The previous state of the game.
        action (Array): The action taken by the player.
        state (State): The current state of the game.

    Returns:
        Array: A scalar array `f32[]`."""
    return jnp.asarray(events.on_target_fetched(prev_state, action, state), dtype=jnp.float32)


def on_put_near_success(prev_state: State, action: Array, state: State) -> Array:
    """`PutNear`'s reward: 1 if the carried object was dropped within
    Chebyshev distance 1 of the second mission target, 0 otherwise.

    Args:
        prev_state (State): The previous state of the game.
        action (Array): The action taken by the player.
        state (State): The current state of the game.

    Returns:
        Array: A scalar array `f32[]`."""
    return jnp.asarray(
        events.on_put_near_success(prev_state, action, state), dtype=jnp.float32
    )


def on_memory_success(prev_state: State, action: Array, state: State) -> Array:
    """`Memory`'s reward: 1 if the player reached the target position,
    0 otherwise (including on failure). Deliberately flat, not
    MiniGrid's step-count-shaped `1 - 0.9 * (step_count / max_steps)` -
    matches navix's existing `on_goal_reached` convention, itself
    already the same simplification versus real MiniGrid's `Goal`
    reward, kept here for consistency rather than a one-off shaped
    reward unique to this environment.

    Args:
        prev_state (State): The previous state of the game.
        action (Array): The action taken by the player.
        state (State): The current state of the game.

    Returns:
        Array: A scalar array `f32[]`."""
    return jnp.asarray(events.on_memory_success(prev_state, action, state), dtype=jnp.float32)


DEFAULT_TASK = compose(on_goal_reached, action_cost)
"""The `reward_fn` an `Environment` uses unless overridden: `+1.0` on
reaching the goal, plus a small negative `action_cost` on every step, so
shorter successful episodes score higher."""
