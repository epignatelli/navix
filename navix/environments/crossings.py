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


from typing import Union
import jax
import jax.numpy as jnp
from jax import Array
from flax import struct

from navix import observations, rewards, terminations

from ..components import EMPTY_POCKET_ID
from ..rendering.cache import RenderingCache
from . import Environment
from ..entities import Player, Goal, Lava
from ..states import State
from . import Timestep
from .registry import register_env


class Crossings(Environment):
    n_crossings: int = struct.field(pytree_node=False, default=1)
    is_lava: bool = struct.field(pytree_node=False, default=False)

    def _reset(self, key: Array, cache: Union[RenderingCache, None] = None) -> Timestep:
        assert (
            self.height == self.width
        ), f"Crossings are only defined for square grids, got height {self.height} and \
            width {self.width}"
        # check minimum height and width
        key, k1, k2 = jax.random.split(key, num=3)

        grid = jnp.zeros((self.height - 2, self.width - 2), dtype=jnp.int32)

        # player
        player_pos = jnp.asarray([1, 1])
        player_dir = jnp.asarray(0)
        player = Player.create(
            position=player_pos, direction=player_dir, pocket=EMPTY_POCKET_ID
        )
        # goal
        goal_pos = jnp.asarray([self.height - 2, self.width - 2])
        goals = Goal.create(position=goal_pos, probability=jnp.asarray(1.0))

        entities = {
            "player": player[None],
            "goal": goals[None],
        }

        # crossings
        obstacles_hor = jnp.mgrid[
            1 : self.height - 2 : 2, 1 : self.width - 1
        ].transpose(1, 2, 0)
        obstacles_ver = jnp.mgrid[
            1 : self.height - 1, 1 : self.width - 2 : 2
        ].transpose(2, 1, 0)
        all_obstacles_pos = jnp.concatenate([obstacles_hor, obstacles_ver])
        num_obstacles = min(self.n_crossings, len(all_obstacles_pos))
        obstacles_pos = jax.random.choice(
            k1, all_obstacles_pos, (num_obstacles,), replace=False
        )

        if self.is_lava:
            entities["lava"] = Lava.create(position=obstacles_pos)
        else:
            grid = grid.at[tuple(obstacles_pos.T)].set(-1)

        # path to goal
        def update(direction, start, grid, step_size):
            return jax.lax.cond(
                direction == jnp.asarray(0, dtype=jnp.int32),
                lambda: (
                    start + jnp.asarray([0, step_size]),
                    jax.lax.dynamic_update_slice(
                        grid, jnp.zeros((1, step_size), dtype=jnp.int32), tuple(start.T)
                    ),
                ),
                lambda: (
                    start + jnp.asarray([step_size, 0]),
                    jax.lax.dynamic_update_slice(
                        grid, jnp.zeros((step_size, 1), dtype=jnp.int32), tuple(start.T)
                    ),
                ),
            )

        start = jnp.asarray([0, 0], dtype=jnp.int32)
        step_size = 3
        for i in range(10):
            k2, k3 = jax.random.split(k2)
            direction = jax.random.randint(k2, (), minval=0, maxval=2)
            start, grid = update(direction, start, grid, step_size)

        grid = jnp.pad(grid, 1, mode="constant", constant_values=-1)

        state = State(
            key=key,
            grid=grid,
            cache=RenderingCache.init(grid),
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
    "Navix-Crossings-S9N1-v0",
    lambda *args, **kwargs: Crossings.create(
        height=9,
        width=9,
        n_crossings=1,
        observation_fn=kwargs.pop("observation_fn", observations.symbolic),
        reward_fn=kwargs.pop("reward_fn", rewards.on_goal_reached),
        termination_fn=kwargs.pop("termination_fn", terminations.on_goal_reached),
        *args,
        **kwargs,
    ),
)
register_env(
    "Navix-Crossings-S9N2-v0",
    lambda *args, **kwargs: Crossings.create(
        height=9,
        width=9,
        n_crossings=2,
        observation_fn=kwargs.pop("observation_fn", observations.symbolic),
        reward_fn=kwargs.pop("reward_fn", rewards.on_goal_reached),
        termination_fn=kwargs.pop("termination_fn", terminations.on_goal_reached),
        *args,
        **kwargs,
    ),
)
register_env(
    "Navix-Crossings-S9N3-v0",
    lambda *args, **kwargs: Crossings.create(
        height=9,
        width=9,
        n_crossings=3,
        observation_fn=kwargs.pop("observation_fn", observations.symbolic),
        reward_fn=kwargs.pop("reward_fn", rewards.on_goal_reached),
        termination_fn=kwargs.pop("termination_fn", terminations.on_goal_reached),
        *args,
        **kwargs,
    ),
)
register_env(
    "Navix-Crossings-S11N5-v0",
    lambda *args, **kwargs: Crossings.create(
        height=11,
        width=11,
        n_crossings=5,
        observation_fn=kwargs.pop("observation_fn", observations.symbolic),
        reward_fn=kwargs.pop("reward_fn", rewards.on_goal_reached),
        termination_fn=kwargs.pop("termination_fn", terminations.on_goal_reached),
        *args,
        **kwargs,
    ),
)
