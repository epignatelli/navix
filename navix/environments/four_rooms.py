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
from typing import Union

import jax
import jax.numpy as jnp
from jax import Array

from navix import observations, rewards, terminations

from ..components import EMPTY_POCKET_ID
from ..entities import Entities, Goal, Player, Wall
from ..states import State
from ..grid import (
    random_positions,
    random_directions,
    room,
    horizontal_wall,
    vertical_wall,
)
from ..rendering.cache import RenderingCache
from .environment import Environment, Timestep
from .registry import register_env


class FourRooms(Environment):
    def _reset(self, key: Array, cache: Union[RenderingCache, None] = None) -> Timestep:
        assert self.height > 4, f"Insufficient height for room {self.height} < 4"
        assert self.width > 4, f"Insufficient width for room {self.width} < 4"
        key, k1, k2 = jax.random.split(key, 3)

        # map
        grid = room(height=self.height, width=self.width)

        # vertical partition
        opening_1 = jax.random.randint(k1, shape=(), minval=1, maxval=self.height // 2)
        opening_2 = jax.random.randint(
            k1, shape=(), minval=self.height // 2 + 2, maxval=self.height
        )
        openings = jnp.stack([opening_1, opening_2])
        wall_pos_vert = vertical_wall(grid, 9, openings)

        # horizontal partition
        opening_1 = jax.random.randint(k2, shape=(), minval=1, maxval=self.width // 2)
        opening_2 = jax.random.randint(
            k1, shape=(), minval=self.width // 2 + 2, maxval=self.width
        )
        openings = jnp.stack([opening_1, opening_2])
        wall_pos_hor = horizontal_wall(grid, 9, openings)

        walls_pos = jnp.concatenate([wall_pos_vert, wall_pos_hor])
        walls = Wall.create(position=walls_pos)

        # player
        player_pos, goal_pos = random_positions(k1, grid, n=2, exclude=walls_pos)
        direction = random_directions(k2, n=1)
        player = Player.create(
            position=player_pos,
            direction=direction,
            pocket=EMPTY_POCKET_ID,
        )
        # goal
        goal = Goal.create(position=goal_pos, probability=jnp.asarray(1.0))

        entities = {
            Entities.PLAYER: player[None],
            Entities.GOAL: goal[None],
            Entities.WALL: walls,
        }

        # systems
        state = State(
            key=key,
            grid=grid,
            cache=cache or RenderingCache.init(grid),
            entities=entities,
        )

        return Timestep(
            t=jnp.asarray(0, dtype=jnp.int32),
            observation=self.observation_fn(state),
            action=jnp.asarray(0, dtype=jnp.int32),
            reward=jnp.asarray(0.0, dtype=jnp.float32),
            step_type=jnp.asarray(0, dtype=jnp.int32),
            state=state,
        )


register_env(
    "Navix-FourRooms-v0",
    lambda *args, **kwargs: FourRooms.create(
        height=19,
        width=19,
        observation_fn=kwargs.pop("observation_fn", observations.symbolic),
        reward_fn=kwargs.pop("reward_fn", rewards.on_goal_reached),
        termination_fn=kwargs.pop("termination_fn", terminations.on_goal_reached),
        *args,
        **kwargs,
    ),
)
