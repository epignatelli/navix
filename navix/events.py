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

from .states import State, EventType, GRID
from .grid import positions_equal, translate
from .entities import Entities, Player


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
