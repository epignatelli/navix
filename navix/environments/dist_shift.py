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

import jax.numpy as jnp
from jax import Array
from flax import struct

from navix import observations, rewards, terminations

from ..components import EMPTY_POCKET_ID
from ..entities import Entities, Goal, Lava, Player
from ..states import State
from ..grid import room
from ..rendering.cache import RenderingCache
from .environment import Environment, Timestep
from .registry import register_env


class DistShift(Environment):
    split_lava: bool = struct.field(pytree_node=False, default=False)

    def _reset(self, key: Array, cache: Union[RenderingCache, None] = None) -> Timestep:
        # map
        grid = room(height=self.height, width=self.width)

        # goal and player
        player_pos = jnp.asarray([1, 1])
        direction = jnp.asarray(0)
        player = Player.create(
            position=player_pos,
            direction=direction,
            pocket=EMPTY_POCKET_ID,
        )
        # goal
        goal_pos = jnp.asarray([1, self.width - 2])
        goal = Goal.create(position=goal_pos, probability=jnp.asarray(1.0))

        # lava
        last_row = 5 if self.split_lava else 2
        lava_pos = jnp.asarray(
            [[1, 3], [1, 4], [1, 5], [last_row, 3], [last_row, 4], [last_row, 5]]
        )
        lava = Lava.create(lava_pos)

        entities = {
            Entities.PLAYER: player[None],
            Entities.GOAL: goal[None],
            Entities.LAVA: lava,
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
    "Navix-DistShift1-v0",
    lambda *args, **kwargs: DistShift.create(
        height=7,
        width=9,
        split_lava=False,
        observation_fn=kwargs.pop("observation_fn", observations.symbolic),
        reward_fn=kwargs.pop("reward_fn", rewards.on_goal_reached),
        termination_fn=kwargs.pop("termination_fn", terminations.on_goal_reached),
        *args,
        **kwargs,
    ),
)
register_env(
    "Navix-DistShift2-v0",
    lambda *args, **kwargs: DistShift.create(
        height=7,
        width=9,
        split_lava=True,
        observation_fn=kwargs.pop("observation_fn", observations.symbolic),
        reward_fn=kwargs.pop("reward_fn", rewards.on_goal_reached),
        termination_fn=kwargs.pop("termination_fn", terminations.on_goal_reached),        
        *args,
        **kwargs,
    ),
)
