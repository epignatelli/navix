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

import jax
import jax.numpy as jnp

import navix as nx
from navix import observations
from navix.components import EMPTY_POCKET_ID
from navix.entities import Ball, Entities, Player
from navix.rendering.cache import RenderingCache
from navix.rendering.registry import PALETTE
from navix.states import State


def test_82():
    env = nx.make(
        "Navix-DoorKey-5x5-v0",
        max_steps=100,
        observation_fn=observations.rgb,
    )
    key = jax.random.PRNGKey(5)
    timestep = env.reset(key)
    # Seed 5 is:
    # # # # #
    # P # . #
    # . # . #
    # K D G #
    # # # # #

    # start agent direction = EAST
    prev_pos = timestep.state.entities["player"].position
    # action 2 is forward
    timestep = env.step(timestep, 2)  # should not walk into wall
    pos = timestep.state.entities["player"].position
    assert jnp.array_equal(prev_pos, pos)


def test_91():
    # https://github.com/epignatelli/navix/issues/91
    # walking into a ball must record a ball_hit event (and so terminate,
    # via the default on_ball_hit termination), matching MiniGrid's
    # DynamicObstacles semantics. record_walk_into previously only handled
    # Goal/Wall/Lava, so the player's own move into a ball was silently a
    # no-op: blocked (Ball.walkable=False) but no event recorded.
    height, width = 5, 5
    grid = jnp.zeros((height - 2, width - 2), dtype=jnp.int32)
    grid = jnp.pad(grid, pad_width=1, mode="constant", constant_values=1)
    player = Player(
        position=jnp.asarray((1, 1)), direction=jnp.asarray(0), pocket=EMPTY_POCKET_ID
    )
    ball = Ball.create(
        position=jnp.asarray((1, 2)),
        colour=PALETTE.BLUE,
        probability=jnp.asarray(0.0),
    )
    cache = RenderingCache.init(grid)
    state = State(
        key=jax.random.PRNGKey(0),
        grid=grid,
        cache=cache,
        entities={
            Entities.PLAYER: player[None],
            Entities.BALL: ball[None],
        },
    )

    state = nx.actions.forward(state)  # player attempts to walk into the ball

    player = state.get_player()
    assert jnp.array_equal(player.position, jnp.asarray((1, 1))), (
        "Expected the player to remain in place, since balls are not walkable"
    )
    assert state.events.ball_hit.happened, (
        "Expected walking into a ball to record a ball_hit event"
    )


if __name__ == "__main__":
    test_82()
    test_91()
