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

"""`PutNear` (issue #178): a single room scattered with `n_objects`
`Key`/`Ball`/`Box` instances; two distinct ones are chosen as the
per-episode "move" (`State.mission[0]` - the object to carry) and
"target" (`State.mission[1]` - the object to drop near) objects. The
agent must pick up the move object and drop it within Chebyshev
distance 1 of the target. Verified against MiniGrid's actual
`PutNearEnv.step`: picking up the wrong object ends the episode
immediately (0 reward); any genuine drop attempt (was holding
something) also ends the episode, success determined by whether it
landed near the target.

Each object's type is drawn independently (matching MiniGrid's own
`types = ["key", "ball", "box"]` sampled per object, checked directly
against source) - see go_to_object.py's module docstring for the
padding-sentinel implementation this shares.

Registers with `transitions_fn=transitions.deterministic_transition` -
see go_to_object.py's module docstring for why (the default
`stochastic_transition` would otherwise walk every `Ball` entity a
random step each turn, which real MiniGrid's `Ball` doesn't do outside
`DynamicObstacles`)."""

from __future__ import annotations
from typing import Union

import jax
import jax.numpy as jnp
from jax import Array
from flax import struct

from navix import observations, rewards, terminations, transitions

from ..components import DISCARD_PILE_COORDS, EMPTY_POCKET_ID
from ..entities import Ball, Box, Entities, Key, Player
from ..grid import (
    random_colour,
    random_directions,
    random_distinct_positions,
    random_position_far_from,
    random_positions,
    room,
)
from ..rendering.cache import RenderingCache
from ..states import Event, State
from . import Environment, Timestep
from .registry import register_env


class PutNear(Environment):
    n_objects: int = struct.field(pytree_node=False, default=2)

    def _reset(self, key: Array, cache: Union[RenderingCache, None] = None) -> Timestep:
        k_pos, k_dir, k_obj_pos, k_colour, k_targets, k_target_pos, k_types = jax.random.split(
            key, 7
        )

        grid = room(self.height, self.width)

        player_pos = random_positions(k_pos, grid)
        player = Player.create(
            position=player_pos,
            direction=random_directions(k_dir),
            pocket=EMPTY_POCKET_ID,
        )

        colours = jnp.reshape(random_colour(k_colour, n=self.n_objects), (self.n_objects,))
        # mutually distinct positions - random_positions(..., n=n) alone
        # would allow two objects to land on the same cell (see #178's PR).
        positions = random_distinct_positions(
            k_obj_pos, grid, n=self.n_objects, exclude=player_pos
        )

        # two distinct objects: move (to carry) and target (to drop near).
        # Determined before building entities, since the target's
        # position may need correcting below.
        move_idx, target_idx = jax.random.choice(
            k_targets, self.n_objects, shape=(2,), replace=False
        )

        # random_distinct_positions alone doesn't stop the move/target
        # pair from spawning already within "near" (Chebyshev <= 1) of
        # each other - trivially "solved" with no real navigation
        # needed (quantified: 36% of Navix-PutNear-6x6-N2-v0 episodes,
        # see grid.random_position_far_from's docstring). If so,
        # re-place just the target object, far enough from the move
        # object and clear of every other already-placed position.
        move_pos = positions[move_idx]
        target_pos = positions[target_idx]
        chebyshev = jnp.maximum(
            jnp.abs(move_pos[0] - target_pos[0]), jnp.abs(move_pos[1] - target_pos[1])
        )
        too_close = chebyshev <= 1
        resampled_target_pos = random_position_far_from(
            k_target_pos,
            grid,
            reference=move_pos,
            min_distance=2,
            exclude=jnp.concatenate([positions, player_pos[None]], axis=0),
        )
        positions = positions.at[target_idx].set(
            jnp.where(too_close, resampled_target_pos, target_pos)
        )

        # each object's type drawn independently, 0=Key/1=Ball/2=Box -
        # every slot is allocated for every type, with non-matching
        # slots pushed off-grid (see module docstring).
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

        # state.mission[0] = move (to carry), state.mission[1] = target
        # (to drop near) - see states.py's State.mission docstring.
        move = Event(
            position=positions[move_idx],
            colour=colours[move_idx],
            happened=jnp.asarray(False),
        )
        target = Event(
            position=positions[target_idx],
            colour=colours[target_idx],
            happened=jnp.asarray(False),
        )

        state = State(
            key=key,
            grid=grid,
            cache=cache or RenderingCache.init(grid),
            entities=entities,
            mission=(move, target),
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
    "Navix-PutNear-6x6-N2-v0",
    lambda *args, **kwargs: PutNear.create(
        height=6,
        width=6,
        n_objects=2,
        observation_fn=kwargs.pop("observation_fn", observations.symbolic),
        transitions_fn=kwargs.pop("transitions_fn", transitions.deterministic_transition),
        reward_fn=kwargs.pop("reward_fn", rewards.on_put_near_success),
        termination_fn=kwargs.pop(
            "termination_fn",
            terminations.compose(
                terminations.on_put_near_wrong_pickup,
                terminations.on_put_near_drop_attempted,
            ),
        ),
        *args,
        **kwargs,
    ),
)
register_env(
    "Navix-PutNear-8x8-N3-v0",
    lambda *args, **kwargs: PutNear.create(
        height=8,
        width=8,
        n_objects=3,
        observation_fn=kwargs.pop("observation_fn", observations.symbolic),
        transitions_fn=kwargs.pop("transitions_fn", transitions.deterministic_transition),
        reward_fn=kwargs.pop("reward_fn", rewards.on_put_near_success),
        termination_fn=kwargs.pop(
            "termination_fn",
            terminations.compose(
                terminations.on_put_near_wrong_pickup,
                terminations.on_put_near_drop_attempted,
            ),
        ),
        *args,
        **kwargs,
    ),
)
