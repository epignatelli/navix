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

from .states import State
from .grid import positions_equal, translate
from .entities import Entities, Player


def on_goal_reached(state: State) -> Array:
    """Checks whether the goal has been reached using the `goal_reached` event.

    Args:
        state (State): The current state of the game.

    Returns:
        Array: A boolean array indicating whether the goal has been reached."""
    return state.events.goal_reached.happened


def on_lava_fall(state: State) -> Array:
    """Checks whether the lava has fallen using the `lava_fall` event.

    Args:
        state (State): The current state of the game.

    Returns:
        Array: A boolean array indicating whether the lava has fallen."""
    return state.events.lava_fall.happened


def on_ball_hit(state: State) -> Array:
    """Checks whether the ball has hit something using the `ball_hit` event.

    Args:
        state (State): The current state of the game.

    Returns:
        Array: A boolean array indicating whether the ball has hit something."""
    return state.events.ball_hit.happened


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


def on_wall_hit(state: State) -> Array:
    """Checks whether the wall has been hit using the `wall_hit` event.

    Args:
        state (State): The current state of the game.

    Returns:
        Array: A boolean array indicating whether the wall has been hit."""
    return state.events.wall_hit.happened
