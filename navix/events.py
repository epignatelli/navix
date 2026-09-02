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

from . import actions
from .states import State, EventType, GRID
from .grid import positions_equal, translate
from .entities import Entities, Player


# Indices into navix.actions.MINIGRID_ACTION_SET/DEFAULT_ACTION_SET -
# derived, not hardcoded, so they stay correct if that tuple's ordering
# ever changes. Only meaningful for an environment actually using that
# default action_set (matches the existing precedent of
# `rewards.action_cost` hardcoding an action index the same way).
DONE_ACTION = jnp.asarray(actions.MINIGRID_ACTION_SET.index(actions.done))
TOGGLE_ACTION = jnp.asarray(actions.MINIGRID_ACTION_SET.index(actions.toggle))
DROP_ACTION = jnp.asarray(actions.MINIGRID_ACTION_SET.index(actions.drop))


def on_goal_reached(state: State) -> Array:
    """Checks whether the goal has been reached using the `goal_reached` event.

    Args:
        state (State): The current state of the game.

    Returns:
        Array: A boolean scalar indicating whether the goal has been reached
        by any goal instance this step."""
    return state.events.happened((Entities.GOAL, EventType.REACH))


def on_lava_fall(state: State) -> Array:
    """Checks whether the lava has fallen using the `lava_fall` event.

    Args:
        state (State): The current state of the game.

    Returns:
        Array: A boolean scalar indicating whether the lava has fallen for
        any lava instance this step."""
    return state.events.happened((Entities.LAVA, EventType.FALL))


def on_ball_hit(state: State) -> Array:
    """Checks whether the ball has hit something using the `ball_hit` event.

    Args:
        state (State): The current state of the game.

    Returns:
        Array: A boolean scalar indicating whether any ball hit the player
        this step."""
    return state.events.happened((Entities.BALL, EventType.HIT))


def on_door_done(state: State) -> Array:
    """Checks whether the action `done` has been called in front of a `Door` object with the correct colour.

    Args:
        state (State): The current state of the game.

    Returns:
        Array: A boolean array indicating whether the action `done` has been called in front of a `Door` object with the correct colour.
    """
    assert (
        state.mission is not None
    ), "Termination on door done requires the state to specify a mission."
    player = state.entities[Entities.PLAYER][0]
    assert isinstance(player, Player)

    fwd_pos = translate(player.position, player.direction)
    if Entities.DOOR not in state.entities:
        return jnp.asarray(False)
    doors = state.get_doors()
    idx = jnp.where(positions_equal(doors.position, fwd_pos), size=1)[0][0]
    doors = doors[idx]
    pos_match = jnp.array_equal(fwd_pos, state.mission.position)
    colour_match = jnp.array_equal(doors.colour, state.mission.colour)
    return jnp.logical_and(pos_match, colour_match)


def on_door_open(state: State) -> Array:
    """Checks whether any door was opened using the `door_opening` event -
    unlike `on_door_done`, this doesn't need a `state.mission` target;
    any door opening (unlocking or not) counts.

    Args:
        state (State): The current state of the game.

    Returns:
        Array: A boolean scalar indicating whether any door was opened
        this step."""
    return state.events.happened((Entities.DOOR, EventType.OPEN))


def on_box_pickup(state: State) -> Array:
    """Checks whether any box was picked up using the `box_pickup` event.

    Args:
        state (State): The current state of the game.

    Returns:
        Array: A boolean scalar indicating whether any box was picked up
        this step."""
    return state.events.happened((Entities.BOX, EventType.PICKUP))


def on_wall_hit(state: State) -> Array:
    """Checks whether the wall has been hit using the `wall_hit` event -
    either an actual `Wall` entity, or the grid boundary/a non-walkable
    empty cell with no entity there (see `navix.states.GRID`).

    Args:
        state (State): The current state of the game.

    Returns:
        Array: A boolean scalar indicating whether a wall was hit this
        step."""
    return jnp.logical_or(
        state.events.happened((Entities.WALL, EventType.HIT)),
        state.events.happened((GRID, EventType.HIT)),
    )


def on_key_pickup(state: State) -> Array:
    """Checks whether any key was picked up using the `key_pickup` event.

    Args:
        state (State): The current state of the game.

    Returns:
        Array: A boolean scalar indicating whether any key was picked up
        this step."""
    return state.events.happened((Entities.KEY, EventType.PICKUP))


def on_ball_pickup(state: State) -> Array:
    """Checks whether any ball was picked up using the `ball_pickup` event.

    Args:
        state (State): The current state of the game.

    Returns:
        Array: A boolean scalar indicating whether any ball was picked up
        this step."""
    return state.events.happened((Entities.BALL, EventType.PICKUP))


def on_ordered_doors_success(prev_state: State, state: State) -> Array:
    """`RedBlueDoors`' win condition: `doors[1]` (blue - see
    `RedBlueDoors._reset`'s fixed construction order, no colour search
    needed) transitions closed -> open exactly this step, while
    `doors[0]` (red) was *already* open in `prev_state`.

    Unlike every other function in this module, this needs both
    `prev_state` and `state` - the win condition is about the *order*
    two doors were opened in, which no single state can encode on its
    own (events are reset every step, and a `Door.open` flag alone can't
    tell you whether it just changed or has been open for a while).

    Args:
        prev_state (State): The state before this step's action.
        state (State): The current state of the game.

    Returns:
        Array: A boolean scalar."""
    red_already_open = jnp.asarray(prev_state.get_doors().open[0], dtype=jnp.bool_)
    blue_just_opened = jnp.logical_and(
        jnp.logical_not(jnp.asarray(prev_state.get_doors().open[1], dtype=jnp.bool_)),
        jnp.asarray(state.get_doors().open[1], dtype=jnp.bool_),
    )
    return jnp.logical_and(red_already_open, blue_just_opened)


def on_ordered_doors_failure(prev_state: State, state: State) -> Array:
    """`RedBlueDoors`' fail condition: `doors[1]` (blue) transitions
    closed -> open exactly this step, while `doors[0]` (red) was *not*
    already open - i.e. blue was opened first (or simultaneously).

    Args:
        prev_state (State): The state before this step's action.
        state (State): The current state of the game.

    Returns:
        Array: A boolean scalar."""
    red_already_open = jnp.asarray(prev_state.get_doors().open[0], dtype=jnp.bool_)
    blue_just_opened = jnp.logical_and(
        jnp.logical_not(jnp.asarray(prev_state.get_doors().open[1], dtype=jnp.bool_)),
        jnp.asarray(state.get_doors().open[1], dtype=jnp.bool_),
    )
    return jnp.logical_and(jnp.logical_not(red_already_open), blue_just_opened)


def on_target_done(action: Array, state: State) -> Array:
    """`GoToObject`'s real win condition (verified against MiniGrid's
    `GoToObjectEnv.step`): the `done` action was called while facing
    `state.mission`'s target position - unlike `on_door_done` (kept
    as-is for `GoToDoor`'s existing, already-shipped behavior), this
    actually checks `action`, matching MiniGrid precisely.

    Args:
        action (Array): The action taken by the player.
        state (State): The current state of the game.

    Returns:
        Array: A boolean scalar."""
    assert (
        state.mission is not None
    ), "on_target_done requires the state to specify a mission."
    player = state.entities[Entities.PLAYER][0]
    assert isinstance(player, Player)
    fwd_pos = translate(player.position, player.direction)
    facing_target = jnp.array_equal(fwd_pos, state.mission.position)
    called_done = jnp.asarray(action == DONE_ACTION)
    return jnp.logical_and(facing_target, called_done)


def on_wrong_toggle(action: Array) -> Array:
    """`GoToObject`'s real fail condition (verified against MiniGrid):
    calling `toggle` at all, regardless of what's in front, immediately
    fails the episode.

    Args:
        action (Array): The action taken by the player.

    Returns:
        Array: A boolean scalar."""
    return jnp.asarray(action == TOGGLE_ACTION)


def on_target_fetched(state: State) -> Array:
    """`Fetch`'s success signal: a `Key`/`Ball` pickup event fired this
    step at `state.mission`'s tracked position - `Event.position` keeps
    the picked-up instance's real (pre-discard-pile) position, and
    positions are unique per instance, so this alone identifies whether
    the *specific* target object (not just any object) was picked up.

    Args:
        state (State): The current state of the game.

    Returns:
        Array: A boolean scalar."""
    assert (
        state.mission is not None
    ), "on_target_fetched requires the state to specify a mission."
    fetched_key = state.events.happened_at(
        (Entities.KEY, EventType.PICKUP), state.mission.position
    )
    fetched_ball = state.events.happened_at(
        (Entities.BALL, EventType.PICKUP), state.mission.position
    )
    return jnp.logical_or(fetched_key, fetched_ball)


def on_any_target_pickup(state: State) -> Array:
    """`Fetch`'s termination trigger: MiniGrid's real `FetchEnv` ends the
    episode on *any* `Key`/`Ball` pickup, right or wrong - only the
    reward differs (see `on_target_fetched`).

    Args:
        state (State): The current state of the game.

    Returns:
        Array: A boolean scalar."""
    return jnp.logical_or(on_key_pickup(state), on_ball_pickup(state))


def on_put_near_wrong_pickup(state: State) -> Array:
    """`PutNear`'s fail-on-pickup condition: a `Key`/`Ball` pickup
    happened this step, but not at `state.mission`'s tracked position
    (the "object to carry") - matches MiniGrid's real `PutNearEnv`,
    which ends the episode immediately if the wrong object is picked up.

    Args:
        state (State): The current state of the game.

    Returns:
        Array: A boolean scalar."""
    assert (
        state.mission is not None
    ), "on_put_near_wrong_pickup requires the state to specify a mission."
    any_pickup = on_any_target_pickup(state)
    right_pickup = state.events.happened_at(
        (Entities.KEY, EventType.PICKUP), state.mission.position
    ) | state.events.happened_at((Entities.BALL, EventType.PICKUP), state.mission.position)
    return jnp.logical_and(any_pickup, jnp.logical_not(right_pickup))


def on_put_near_drop_attempted(prev_state: State, action: Array) -> Array:
    """`PutNear`'s termination trigger for a drop: MiniGrid's real
    `PutNearEnv` ends the episode on any genuine drop attempt (a `drop`
    action while actually holding something) - success or failure is
    determined separately by `on_put_near_success`.

    Args:
        prev_state (State): The state before this step's action.
        action (Array): The action taken by the player.

    Returns:
        Array: A boolean scalar."""
    was_holding = prev_state.get_player().pocket != -1
    return jnp.logical_and(was_holding, action == DROP_ACTION)


def on_put_near_success(prev_state: State, action: Array, state: State) -> Array:
    """`PutNear`'s win condition: a genuine drop attempt (see
    `on_put_near_drop_attempted`) that actually placed the item (the
    player's pocket went from holding something to empty - `actions.
    drop` only clears the pocket on a *successful* drop, see #189) at a
    cell within Chebyshev distance 1 of `state.mission2`'s tracked
    position (the "object to drop near"). Note: by the time a `drop`
    action is reached without the episode already having ended via
    `on_put_near_wrong_pickup`, the held item is guaranteed to be the
    right one (`state.mission`'s object) - wrong pickups end the episode
    immediately, so no separate "is this the right item" check is
    needed here.

    Args:
        prev_state (State): The state before this step's action.
        action (Array): The action taken by the player.
        state (State): The current state of the game.

    Returns:
        Array: A boolean scalar."""
    assert (
        state.mission2 is not None
    ), "on_put_near_success requires the state to specify a second mission target."
    drop_attempted = on_put_near_drop_attempted(prev_state, action)
    drop_succeeded = jnp.logical_and(drop_attempted, state.get_player().pocket == -1)
    prev_player = prev_state.get_player()
    drop_position = translate(prev_player.position, prev_player.direction)
    row_diff = jnp.abs(drop_position[0] - state.mission2.position[0])
    col_diff = jnp.abs(drop_position[1] - state.mission2.position[1])
    near_target = jnp.maximum(row_diff, col_diff) <= 1
    return jnp.logical_and(drop_succeeded, near_target)
