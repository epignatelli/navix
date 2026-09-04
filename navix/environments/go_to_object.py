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
`Key`/`Ball`/`Box` instances of distinct colours; one is chosen as the
per-episode target via `State.mission` (same pattern `GoToDoor` already
uses). Success requires calling `done` while facing the target (verified
against MiniGrid's actual `GoToObjectEnv.step` - not just proximity, and
calling `toggle` at all immediately fails), unlike the already-shipped
`GoToDoor`/`events.on_door_done`, which doesn't check the action at all
(a pre-existing gap, left as-is here - see PR description).

Each object's type is drawn independently (matching MiniGrid's own
`types = ["key", "ball", "box"]` sampled per object, checked directly
against source - not a fixed split), so the actual per-type counts vary
by episode; implemented by always allocating `n_objects` Key/Ball/Box
slots and pushing every slot not assigned that episode's type to
`DISCARD_PILE_COORDS` - the same padding-sentinel pattern
`LavaCrossing` already uses for its own variable-count entities.

Registers with `transitions_fn=transitions.deterministic_transition`,
overriding the default `stochastic_transition` - the latter
unconditionally moves *every* `Ball` entity a random step each turn
(`transitions.update_balls`, meant for `DynamicObstacles`), which would
silently walk this environment's target/decoy objects around the room
turn by turn. Real MiniGrid's `Ball` is static outside `DynamicObstacles`
(confirmed empirically: without this override, mission.position stops
matching any object's actual position after just a couple of steps)."""

from __future__ import annotations
from typing import Union

import jax
import jax.numpy as jnp
from jax import Array
from flax import struct

from navix import observations, rewards, terminations, transitions

from ..components import DISCARD_PILE_COORDS, EMPTY_POCKET_ID
from ..entities import Ball, Box, Entities, Key, Player
from ..grid import random_colour, random_directions, random_distinct_positions, random_positions, room
from ..rendering.cache import RenderingCache
from ..states import Event, State
from . import Environment, Timestep
from .registry import register_env


class GoToObject(Environment):
    """`Navix-GoToObject-*`. A room scattered with `n_objects` coloured
    `Key`/`Ball`/`Box` objects of distinct colours; one is named in
    `State.mission`. The agent succeeds by signalling `done` while
    orthogonally adjacent to the target (using `toggle` fails outright).
    Each object's type is drawn independently per episode. Uses
    `deterministic_transition` so balls stay put.

    Attributes:
        n_objects: how many objects to scatter.
    """

    n_objects: int = struct.field(pytree_node=False, default=2)

    def _reset(self, key: Array, cache: Union[RenderingCache, None] = None) -> Timestep:
        k_pos, k_dir, k_obj_pos, k_colour, k_target, k_types = jax.random.split(key, 6)

        grid = room(self.height, self.width)

        player_pos = random_positions(k_pos, grid)
        player = Player.create(
            position=player_pos,
            direction=random_directions(k_dir),
            pocket=EMPTY_POCKET_ID,
        )

        # n_objects distinct colours and positions - random_positions(...,
        # n=n) alone would allow two objects to land on the same cell
        # (see #172's PR).
        colours = jnp.reshape(random_colour(k_colour, n=self.n_objects), (self.n_objects,))
        positions = random_distinct_positions(
            k_obj_pos, grid, n=self.n_objects, exclude=player_pos
        )

        # each object's type drawn independently, 0=Key/1=Ball/2=Box -
        # see module docstring for why every slot is allocated for every
        # type, with non-matching slots pushed off-grid.
        type_idx = jax.random.randint(k_types, (self.n_objects,), 0, 3)
        is_key, is_ball, is_box = type_idx == 0, type_idx == 1, type_idx == 2
        key_pos = jnp.where(is_key[:, None], positions, DISCARD_PILE_COORDS)
        ball_pos = jnp.where(is_ball[:, None], positions, DISCARD_PILE_COORDS)
        box_pos = jnp.where(is_box[:, None], positions, DISCARD_PILE_COORDS)

        keys = Key.create(
            position=key_pos, colour=colours, id=jnp.arange(1, self.n_objects + 1, dtype=jnp.int32)
        )
        balls = Ball.create(
            position=ball_pos,
            colour=colours,
            probability=jnp.ones(self.n_objects),
            id=jnp.arange(self.n_objects + 1, 2 * self.n_objects + 1, dtype=jnp.int32),
        )
        boxes = Box.create(
            position=box_pos,
            colour=colours,
            id=jnp.arange(2 * self.n_objects + 1, 3 * self.n_objects + 1, dtype=jnp.int32),
            pocket=jnp.full((self.n_objects,), -1, dtype=jnp.int32),
        )
        entities = {
            Entities.PLAYER: player[None],
            Entities.KEY: keys,
            Entities.BALL: balls,
            Entities.BOX: boxes,
        }

        # target: one of the n_objects positions/colours, chosen uniformly
        # - position alone identifies the target regardless of which
        # type ended up there, so no further change needed here.
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
            mission=(mission,),
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
        transitions_fn=kwargs.pop("transitions_fn", transitions.deterministic_transition),
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
        transitions_fn=kwargs.pop("transitions_fn", transitions.deterministic_transition),
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
