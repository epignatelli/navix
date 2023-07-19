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
from typing import Callable, Union
import sys

import jax
import jax.numpy as jnp
from jax.random import KeyArray

from navix.spaces import Space


from .. import observations
from ..spaces import Space, Continuous, Discrete
from ..components import EMPTY_POCKET_ID
from ..entities import Entities, Goal, Player, State
from ..grid import random_positions, random_directions, room
from ..graphics import RenderingCache, TILE_SIZE
from .environment import Environment, Timestep


class Room(Environment):
    @property
    def observation_space(self) -> Space:
        if self.observation_fn == observations.none:
            return Continuous(shape=())
        elif self.observation_fn == observations.categorical:
            return Discrete(sys.maxsize, shape=(self.height, self.width))
        elif self.observation_fn == observations.categorical_first_person:
            radius = 3  # TOD0: make this a variable
            return Discrete(sys.maxsize, shape=(radius + 1, radius * 2 + 1))
        elif self.observation_fn == observations.rgb:
            return Discrete(
                256,
                shape=(self.height * TILE_SIZE, self.width * TILE_SIZE, 3),
                dtype=jnp.uint8,
            )
        elif self.observation_fn == observations.rgb_first_person:
            radius = 3
            return Discrete(
                256,
                shape=(radius * TILE_SIZE * 2 + 1, radius * TILE_SIZE * 2 + 1, 3),
                dtype=jnp.uint8,
            )
        else:
            raise NotImplementedError(
                "Unknown observation space for observation function {}".format(
                    self.observation_fn
                )
            )

    def reset(
        self, key: KeyArray, cache: Union[RenderingCache, None] = None
    ) -> Timestep:
        key, k1, k2 = jax.random.split(key, 3)

        # map
        grid = room(height=self.height, width=self.width)

        # player
        player_pos, goal_pos = random_positions(k1, grid, n=2)
        direction = random_directions(k2, n=1)
        player = Player(
            position=player_pos,
            direction=direction,
            pocket=EMPTY_POCKET_ID,
        )
        # goal
        goal = Goal(position=goal_pos, probability=jnp.asarray(1.0))

        entities = {
            Entities.PLAYER.value: player[None],
            Entities.GOAL.value: goal[None],
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
            observation=self.observation(state),
            action=jnp.asarray(0, dtype=jnp.int32),
            reward=jnp.asarray(0.0, dtype=jnp.float32),
            step_type=jnp.asarray(0, dtype=jnp.int32),
            state=state,
        )
