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
from navix.components import DISCARD_PILE_COORDS, EMPTY_POCKET_ID
from navix.entities import Ball, Entities, EntityIds, Goal, Key, Player
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


def test_98():
    # https://github.com/epignatelli/navix/issues/98
    env = nx.make("Navix-KeyCorridorS3R1-v0")
    timestep = env.reset(jax.random.PRNGKey(0))
    state = timestep.state

    # 1. every door position must be floor (0) in the base grid, not a
    # hardcoded wall - otherwise a door's own open/closed state can never
    # actually control whether the agent can pass, since _can_walk_there
    # requires both the grid cell and the entity to be walkable
    doors = state.get_doors()
    door_cells = state.grid[tuple(doors.position.T)]
    assert jnp.all(door_cells == 0), (
        "Expected every door position to be floor (0) in the base grid, "
        "got {}".format(door_cells)
    )

    # 2. the target must be a walkable Goal, not a Ball (Ball.walkable=False,
    # which made the room unsolvable - the agent could never reach it)
    goal = state.get_goals()
    assert isinstance(goal, Goal), "Expected the target entity to be a Goal"
    assert jnp.all(goal.walkable), "Expected the Goal to be walkable"


def test_135():
    # https://github.com/epignatelli/navix/issues/135
    # a picked-up entity's position is moved to DISCARD_PILE_COORDS =
    # (0, -1). categorical()/symbolic() scatter-write the entity's
    # tag/symbol at its position without checking whether that position
    # is actually on the grid - JAX's negative-index wraparound then
    # writes it into a real cell instead of dropping it, producing a
    # "ghost" duplicate of the picked-up item: categorical()'s flat
    # index -1 wraps to the bottom-right corner, symbolic()'s (0, -1)
    # wraps to the top-right corner. Both should remain untouched
    # (still showing the real wall there), since a picked-up entity is
    # carried, not on the grid.
    height, width = 5, 5
    grid = jnp.zeros((height - 2, width - 2), dtype=jnp.int32)
    grid = jnp.pad(grid, pad_width=1, mode="constant", constant_values=-1)
    player = Player(
        position=jnp.asarray((1, 1)), direction=jnp.asarray(0), pocket=EMPTY_POCKET_ID
    )
    key = Key.create(
        position=DISCARD_PILE_COORDS,  # picked up
        colour=PALETTE.BLUE,
        id=jnp.asarray(0),
    )
    cache = RenderingCache.init(grid)
    state = State(
        key=jax.random.PRNGKey(0),
        grid=grid,
        cache=cache,
        entities={
            Entities.PLAYER: player[None],
            Entities.KEY: key[None],
        },
    )

    categorical_obs = observations.categorical(state)
    assert categorical_obs[height - 1, width - 1] == -1, (
        "Expected the picked-up key not to wrap around into the "
        "bottom-right corner of categorical(), got tag "
        f"{categorical_obs[height - 1, width - 1]} instead of the wall (-1)"
    )

    symbolic_obs = observations.symbolic(state)
    wall_symbol = jnp.array([EntityIds.WALL, 5, 0], dtype=jnp.uint8)
    assert jnp.array_equal(symbolic_obs[0, width - 1], wall_symbol), (
        "Expected the picked-up key not to wrap around into the "
        "top-right corner of symbolic(), got "
        f"{symbolic_obs[0, width - 1]} instead of the wall symbol {wall_symbol}"
    )


def test_146():
    # https://github.com/epignatelli/navix/issues/146
    # same bug class as #135, but in categorical_first_person():
    # transparency_map/state.grid are scattered via raw entity
    # positions without validating they're on-grid, so a picked-up
    # entity at DISCARD_PILE_COORDS = (0, -1) wraps into row 0's last
    # column instead of being dropped. A correctly-handled picked-up
    # entity should have zero effect on the observation (matching
    # MiniGrid: picked-up items are removed from the grid entirely),
    # so compare against an otherwise-identical state with no key
    # entity at all - avoids having to hand-compute exactly which
    # cell of crop()'s pad/roll/rotate/slice output the wraparound
    # would land in.
    height, width = 5, 5
    grid = jnp.zeros((height - 2, width - 2), dtype=jnp.int32)
    grid = jnp.pad(grid, pad_width=1, mode="constant", constant_values=-1)
    player = Player(
        position=jnp.asarray((2, 2)), direction=jnp.asarray(0), pocket=EMPTY_POCKET_ID
    )
    key = Key.create(
        position=DISCARD_PILE_COORDS,  # picked up
        colour=PALETTE.BLUE,
        id=jnp.asarray(0),
    )
    cache = RenderingCache.init(grid)

    state_with_key = State(
        key=jax.random.PRNGKey(0),
        grid=grid,
        cache=cache,
        entities={
            Entities.PLAYER: player[None],
            Entities.KEY: key[None],
        },
    )
    state_without_key = State(
        key=jax.random.PRNGKey(0),
        grid=grid,
        cache=cache,
        entities={Entities.PLAYER: player[None]},
    )

    obs_with_key = observations.categorical_first_person(state_with_key)
    obs_without_key = observations.categorical_first_person(state_without_key)
    assert jnp.array_equal(obs_with_key, obs_without_key), (
        "Expected a picked-up key to have no effect on "
        "categorical_first_person(), since it should be treated as "
        f"off-grid - got\n{obs_with_key}\ninstead of\n{obs_without_key}"
    )


if __name__ == "__main__":
    test_82()
    test_91()
    test_98()
    test_135()
    test_146()
