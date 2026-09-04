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

"""`RedBlueDoors` (issue #172): a single room split by one dividing wall
into two chambers, with a red and a blue `Door` both embedded in that
same wall (at two distinct rows) - both reachable from the left chamber
alone, no need to ever cross into the right one. Reward + termination on
opening blue while red is already open; opening blue first ends the
episode with no reward.

Deliberately simplified from MiniGrid's actual `RedBlueDoorEnv._gen_grid`
(checked directly, after PR #191's review caught an earlier, incorrect
claim of an exact match here): real MiniGrid places the agent in a
*third*, middle chamber, with red and blue on two *different* walls
leading to two separate outer chambers, not one shared wall - but since
`toggle` only needs the player adjacent-and-facing a door (never
actually walking through one), and termination fires the instant blue
opens (before either outer chamber could ever be reached), the two
layouts are behaviourally indistinguishable through gameplay: order,
not position, is all that determines success either way."""

from __future__ import annotations
from typing import Union

import jax
import jax.numpy as jnp
from jax import Array

from navix import observations, rewards, terminations

from ..components import EMPTY_POCKET_ID
from ..entities import Door, Entities, Player
from ..grid import mask_by_coordinates, random_directions, random_positions, two_rooms
from ..rendering.cache import RenderingCache
from ..rendering.registry import PALETTE
from ..states import State
from . import Environment, Timestep
from .registry import register_env


class RedBlueDoors(Environment):
    """`Navix-RedBlueDoors-*`. One room, a dividing wall holding a red and
    a blue `Door` (both reachable from the start side). Success is
    opening the blue door *after* the red one is already open; opening
    blue first ends the episode with no reward. The task is about
    ordering, not navigation."""

    def _reset(self, key: Array, cache: Union[RenderingCache, None] = None) -> Timestep:
        k_wall, k_rows, k_pos, k_dir = jax.random.split(key, 4)

        grid, wall_col = two_rooms(self.height, self.width, k_wall)

        # two distinct rows within the dividing wall's interior span for
        # the red/blue doors - fixed construction order (red = index 0,
        # blue = index 1) is relied on by events.on_ordered_doors_*, no
        # colour search needed there.
        rows = jax.random.choice(
            k_rows, jnp.arange(1, self.height - 1), shape=(2,), replace=False
        )
        positions = jnp.stack([rows, jnp.full((2,), wall_col)], axis=1)
        grid = grid.at[rows, wall_col].set(0)

        doors = Door.create(
            position=positions,
            requires=jnp.asarray([-1, -1]),  # unlocked - just closed
            colour=jnp.asarray([PALETTE.RED, PALETTE.BLUE]),
            open=jnp.asarray([False, False]),
        )

        # agent starts in the left chamber only, matching MiniGrid
        left_chamber_mask = mask_by_coordinates(
            grid, (jnp.asarray(self.height), jnp.asarray(wall_col)), jnp.less
        )
        left_chamber = jnp.where(left_chamber_mask, grid, -1)
        player_pos = random_positions(k_pos, left_chamber)
        player = Player.create(
            position=player_pos,
            direction=random_directions(k_dir),
            pocket=EMPTY_POCKET_ID,
        )

        entities = {
            Entities.PLAYER: player[None],
            Entities.DOOR: doors,
        }

        state = State(
            key=key,
            grid=grid,
            cache=cache or RenderingCache.init(grid),
            entities=entities,
        )
        return Timestep(
            t=jnp.asarray(0, dtype=jnp.int32),
            observation=self.observation_fn(state),
            action=jnp.asarray(-1, dtype=jnp.int32),
            reward=jnp.asarray(0.0, dtype=jnp.float32),
            step_type=jnp.asarray(0, dtype=jnp.int32),
            state=state,
        )


register_env(
    "Navix-RedBlueDoors-6x6-v0",
    lambda *args, **kwargs: RedBlueDoors.create(
        height=6,
        width=6,
        observation_fn=kwargs.pop("observation_fn", observations.symbolic),
        reward_fn=kwargs.pop("reward_fn", rewards.on_ordered_doors_success),
        termination_fn=kwargs.pop(
            "termination_fn", terminations.on_ordered_doors_resolved
        ),
        *args,
        **kwargs,
    ),
)
register_env(
    "Navix-RedBlueDoors-8x8-v0",
    lambda *args, **kwargs: RedBlueDoors.create(
        height=8,
        width=8,
        observation_fn=kwargs.pop("observation_fn", observations.symbolic),
        reward_fn=kwargs.pop("reward_fn", rewards.on_ordered_doors_success),
        termination_fn=kwargs.pop(
            "termination_fn", terminations.on_ordered_doors_resolved
        ),
        *args,
        **kwargs,
    ),
)
