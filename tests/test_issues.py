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
import numpy as np

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


def test_147():
    # https://github.com/epignatelli/navix/issues/147
    # crop() places the agent at the *bottom* row of the 2*RADIUS+1
    # first-person view, so the far row is 2*RADIUS cells forward of
    # the agent - but categorical_first_person()/rgb_first_person()
    # only diffused view_cone's visibility RADIUS cells, so the
    # forward half of every first-person view was permanently marked
    # unseen regardless of whether real walls were there. Verified
    # against a real MiniGrid render of the identical scenario (open
    # room, agent centred, no walls within 2*RADIUS in any direction):
    # pre-fix, Navix's forward half is solid black where MiniGrid
    # shows open floor; post-fix they closely match.
    import gymnasium as gym
    import minigrid  # noqa: F401 - registers MiniGrid-* env ids

    radius = observations.RADIUS
    tile_size = 8
    centre = 8  # Navix-Empty-16x16-v0 / MiniGrid-Empty-16x16-v0 interior

    env = nx.make(
        "Navix-Empty-16x16-v0", observation_fn=observations.rgb_first_person
    )
    timestep = env.reset(jax.random.PRNGKey(0))
    player = timestep.state.entities[Entities.PLAYER]
    player = player.replace(
        position=jnp.asarray([[centre, centre]]), direction=jnp.asarray([0])
    )
    state = timestep.state.replace(
        entities={**timestep.state.entities, Entities.PLAYER: player}
    )
    navix_img = np.asarray(observations.rgb_first_person(state))

    mg_env = gym.make(
        "MiniGrid-Empty-16x16-v0",
        agent_view_size=2 * radius + 1,
        tile_size=tile_size,
    )
    mg_env.reset(seed=0)
    mg_env.unwrapped.agent_pos = (centre, centre)
    mg_env.unwrapped.agent_dir = 0
    mg_img = mg_env.unwrapped.get_frame(agent_pov=True, tile_size=tile_size)

    diff = np.abs(navix_img.astype(int) - mg_img.astype(int))
    assert diff.mean() < 20, (
        "Expected Navix's rgb_first_person to closely match a real MiniGrid "
        "render in an open room with no walls within reach, got mean "
        f"abs pixel diff {diff.mean():.1f} (max {diff.max()}) - the forward "
        "half of the view is likely still being marked unseen"
    )


def test_141():
    # https://github.com/epignatelli/navix/issues/141
    # KeyCorridor's goal-room door was hardcoded to open=jnp.asarray(2)
    # at reset. Door.walkable casts `open` to bool, and bool(2) is True
    # - so the locked door on the critical path to the goal was walkable
    # from the very start, regardless of requires=key_id. Before PR #126
    # this was masked by the grid hardcoding every door position as a
    # wall; once #126 punched door cells to floor, this became live:
    # the agent could walk straight to the goal without ever picking up
    # the key, defeating the environment's own puzzle. Flagged by an
    # automated review on PR #126 and left unfixed until now - open=2
    # also corrupted Door.symbolic_state (closed = 1 - 2 = -1, out of
    # range) and Door.sprite's index (open + 2*locked = 4, out of range
    # on a size-3 axis, silently clamped by JAX).
    for env_id in [
        "Navix-KeyCorridorS3R1-v0",
        "Navix-KeyCorridorS3R2-v0",
        "Navix-KeyCorridorS3R3-v0",
    ]:
        env = nx.make(env_id)
        keys = jax.random.split(jax.random.PRNGKey(0), 32)
        for k in keys:
            timestep = env.reset(k)
            doors = timestep.state.get_doors()
            locked = doors.requires != EMPTY_POCKET_ID
            assert jnp.all(doors.open[locked] == 0), (
                f"{env_id}: expected every locked door (requires a key) to "
                f"start closed (open=0), got open={doors.open} for "
                f"requires={doors.requires} - a locked door starting open "
                "lets the agent bypass the key entirely"
            )


def test_160():
    # https://github.com/epignatelli/navix/issues/160
    # KeyCorridor built *two* Door entities per row boundary at the same
    # cell: the `for row in range(n_rows - 1)` loop split `k9, k10, k11,
    # k12` but reused the single `door_pos` from `k9` for both doors, so
    # `k11` - clearly meant to draw a second, different position - was
    # never used. The pair only differed in colour (`k10` vs `k12`).
    # They behave as one door (`open` and `_walkable` both reduce over the
    # whole door array), so this is not a dynamics bug, but it does corrupt
    # rendering: `observations.rgb` writes every door sprite into the frame
    # with a single `.at[idx].set(...)`, and a duplicate-index scatter is
    # undefined in JAX - on GPU it resolves per element, so the tile comes
    # out a per-pixel mix of two differently coloured door sprites.
    for env_id in [
        "Navix-KeyCorridorS3R1-v0",
        "Navix-KeyCorridorS3R2-v0",
        "Navix-KeyCorridorS3R3-v0",
        "Navix-KeyCorridorS6R3-v0",
    ]:
        env = nx.make(env_id)
        keys = jax.random.split(jax.random.PRNGKey(0), 32)
        for k in keys:
            timestep = env.reset(k)
            positions = timestep.state.get_doors().position
            n_unique = jnp.unique(positions, axis=0).shape[0]
            assert n_unique == positions.shape[0], (
                f"{env_id}: expected every door to occupy its own cell, got "
                f"{positions.shape[0]} doors on {n_unique} distinct cells "
                f"(positions={positions.tolist()}) - stacked doors make the "
                "duplicated cell render as a mix of both sprites"
            )


if __name__ == "__main__":
    test_82()
    test_91()
    test_98()
    test_135()
    test_146()
    test_147()
    test_141()
    test_160()
