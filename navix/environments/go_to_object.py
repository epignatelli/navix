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

"""`GoToObject` (issue #173): a single room scattered with `n_objects`
`Key`/`Ball` instances of distinct colours; one is chosen as the
per-episode target via `State.mission` (same pattern `GoToDoor` already
uses). Success requires calling `done` while facing the target (verified
against MiniGrid's actual `GoToObjectEnv.step` - not just proximity, and
calling `toggle` at all immediately fails), unlike the already-shipped
`GoToDoor`/`events.on_door_done`, which doesn't check the action at all
(a pre-existing gap, left as-is here - see PR description)."""

from __future__ import annotations
from typing import Union

import jax
import jax.numpy as jnp
from jax import Array
from flax import struct

from navix import observations, rewards, terminations

from ..components import EMPTY_POCKET_ID
from ..entities import Ball, Entities, Key, Player
from ..grid import random_colour, random_directions, random_positions, room
from ..rendering.cache import RenderingCache
from ..states import Event, State
from . import Environment, Timestep
from .registry import register_env


class GoToObject(Environment):
    n_objects: int = struct.field(pytree_node=False, default=2)

    def _reset(self, key: Array, cache: Union[RenderingCache, None] = None) -> Timestep:
        k_pos, k_dir, k_obj_pos, k_colour, k_target = jax.random.split(key, 5)

        grid = room(self.height, self.width)

        player_pos = random_positions(k_pos, grid)
        player = Player.create(
            position=player_pos,
            direction=random_directions(k_dir),
            pocket=EMPTY_POCKET_ID,
        )

        # n_objects distinct colours, split as evenly as possible between
        # Key and Ball (n_objects=2 -> one of each, matching the
        # registered sizes below); positions exclude the player.
        colours = random_colour(k_colour, n=self.n_objects)
        colours = jnp.reshape(colours, (self.n_objects,))
        positions = random_positions(
            k_obj_pos, grid, n=self.n_objects, exclude=player_pos
        )
        positions = jnp.reshape(positions, (self.n_objects, 2))
        n_keys = self.n_objects // 2
        n_balls = self.n_objects - n_keys

        entities = {Entities.PLAYER: player[None]}
        if n_keys > 0:
            keys = Key.create(
                position=positions[:n_keys],
                colour=colours[:n_keys],
                id=jnp.arange(1, n_keys + 1, dtype=jnp.int32),
            )
            entities[Entities.KEY] = keys
        if n_balls > 0:
            balls = Ball.create(
                position=positions[n_keys:],
                colour=colours[n_keys:],
                probability=jnp.ones(n_balls),
                id=jnp.arange(n_keys + 1, self.n_objects + 1, dtype=jnp.int32),
            )
            entities[Entities.BALL] = balls

        # target: one of the n_objects positions/colours, chosen uniformly
        target_idx = jax.random.randint(k_target, (), 0, self.n_objects)
        mission = Event(
            position=positions[target_idx],
            colour=colours[target_idx],
            happened=jnp.asarray(False),
        )

        state = State(
            key=key,
            grid=grid,
            cache=cache or RenderingCache.init(grid),
            entities=entities,
            mission=mission,
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
    "Navix-GoToObject-6x6-N2-v0",
    lambda *args, **kwargs: GoToObject.create(
        height=6,
        width=6,
        n_objects=2,
        observation_fn=kwargs.pop("observation_fn", observations.symbolic),
        reward_fn=kwargs.pop("reward_fn", rewards.on_target_done),
        termination_fn=kwargs.pop(
            "termination_fn",
            terminations.compose(
                terminations.on_target_done, terminations.on_wrong_toggle
            ),
        ),
        *args,
        **kwargs,
    ),
)
register_env(
    "Navix-GoToObject-8x8-N2-v0",
    lambda *args, **kwargs: GoToObject.create(
        height=8,
        width=8,
        n_objects=2,
        observation_fn=kwargs.pop("observation_fn", observations.symbolic),
        reward_fn=kwargs.pop("reward_fn", rewards.on_target_done),
        termination_fn=kwargs.pop(
            "termination_fn",
            terminations.compose(
                terminations.on_target_done, terminations.on_wrong_toggle
            ),
        ),
        *args,
        **kwargs,
    ),
)
