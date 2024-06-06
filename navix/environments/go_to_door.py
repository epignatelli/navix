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
from flax import struct

from navix import observations

from .. import rewards, terminations
from ..components import EMPTY_POCKET_ID
from ..entities import Entities, Door, Player
from ..states import EventType, State, Event
from ..grid import random_colour, random_directions
from ..rendering.cache import RenderingCache
from .environment import Environment, Timestep
from .registry import register_env


class GoToDoor(Environment):
    split_lava: bool = struct.field(pytree_node=False, default=False)

    def _reset(self, key: Array, cache: Union[RenderingCache, None] = None) -> Timestep:
        # map
        grid = jnp.zeros((self.height, self.width), dtype=jnp.int32)

        k1, k2, k3, k4, k5, k6 = jax.random.split(key, num=6)
        room_height = jax.random.randint(k1, (), minval=5, maxval=self.height)
        room_width = jax.random.randint(k1, (), minval=5, maxval=self.width)

        # set wall on grid
        grid = grid.at[jnp.asarray([0, room_height - 1])].set(-1)
        grid = grid.at[:, jnp.asarray([0, room_width - 1])].set(-1)

        # player
        player_row = jax.random.randint(k2, (), minval=1, maxval=room_height - 1)
        player_col = jax.random.randint(k3, (), minval=1, maxval=room_width - 1)
        player_pos = jnp.asarray([player_row, player_col])
        direction = random_directions(k4)
        player = Player(
            position=player_pos,
            direction=direction,
            pocket=EMPTY_POCKET_ID,
        )

        # doors
        k6, k7 = jax.random.split(k5, num=2)
        rows = jax.random.randint(k6, (2,), minval=2, maxval=room_height - 2)
        cols = jax.random.randint(k7, (2,), minval=2, maxval=room_width - 2)
        positions = jnp.asarray(
            [
                [rows[0], room_width - 1],
                [room_height - 1, cols[0]],
                [rows[1], 0],
                [0, cols[1]],
            ]
        )
        colours = random_colour(key, n=4)
        open = jnp.asarray([0] * 4)
        requires = jnp.asarray([-1] * 4)
        doors = Door.create(
            position=positions, requires=requires, colour=colours, open=open
        )

        entities = {
            Entities.PLAYER: player[None],
            Entities.DOOR: doors,
        }

        idx = jax.random.randint(k6, (), minval=0, maxval=4)
        target_door = doors[idx]
        mission = Event(
            position=target_door.position,
            colour=target_door.colour,
            happened=jnp.asarray(False),
            event_type=EventType.REACH,
        )

        # systems
        state = State(
            key=key,
            grid=grid,
            cache=RenderingCache.init(grid),
            entities=entities,
            mission=mission,
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
    "Navix-GoToDoor-5x5-v0",
    lambda *args, **kwargs: GoToDoor.create(
        height=5,
        width=5,
        observation_fn=kwargs.pop("observation_fn", observations.symbolic),
        reward_fn=kwargs.pop("reward_fn", rewards.on_door_done),
        termination_fn=kwargs.pop("termination_fn", terminations.on_door_done),
        *args,
        **kwargs,
    ),
)
register_env(
    "Navix-GoToDoor-6x6-v0",
    lambda *args, **kwargs: GoToDoor.create(
        height=6,
        width=6,
        observation_fn=kwargs.pop("observation_fn", observations.symbolic),
        reward_fn=kwargs.pop("reward_fn", rewards.on_door_done),
        termination_fn=kwargs.pop("termination_fn", terminations.on_door_done),
        *args,
        **kwargs,
    ),
)
register_env(
    "Navix-GoToDoor-8x8-v0",
    lambda *args, **kwargs: GoToDoor.create(
        height=8,
        width=8,
        observation_fn=kwargs.pop("observation_fn", observations.symbolic),
        reward_fn=kwargs.pop("reward_fn", rewards.on_door_done),
        termination_fn=kwargs.pop("termination_fn", terminations.on_door_done),
        *args,
        **kwargs,
    ),
)
