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

"""MiniGrid's Memory environment - recall a cue seen at the start to pick the right exit.

See the environment class in this module for the task, layout and
reward/termination details.
"""


from __future__ import annotations
from typing import Union

import jax
import jax.numpy as jnp
from jax import Array
from flax import struct

from navix import observations

from .. import actions, rewards, terminations, transitions
from ..components import DISCARD_PILE_COORDS, EMPTY_POCKET_ID
from ..entities import Ball, Directions, Entities, Key, Player, Wall
from ..states import State, Event
from ..grid import coordinates, room
from ..rendering.cache import RenderingCache
from .environment import Environment, Timestep
from .registry import register_env


# `pickup` is remapped to `toggle` (both a no-op here, since nothing in
# this layout is `Toggleable` and no `Pickable` needs picking up) -
# verified against MiniGrid's actual `MemoryEnv.step`: `if action ==
# Actions.pickup: action = Actions.toggle`, done before dispatch.
MEMORY_ACTION_SET = (
    actions.rotate_ccw,
    actions.rotate_cw,
    actions.forward,
    actions.toggle,
    actions.drop,
    actions.toggle,
    actions.done,
)

MEMORY_TERMINATION_FN = terminations.compose(
    terminations.on_memory_success, terminations.on_memory_failure
)


class Memory(Environment):
    """A genuine memory test (verified against MiniGrid's actual
    `memory.py`): the agent starts in a small room seeing one green
    object (`Key` or `Ball`, chosen at random each episode), walks down
    a hallway that ends in a T-split, and must walk onto the cell
    adjacent to whichever of the two end objects matches the one it
    saw at the start - the other one ends the episode with 0 reward.
    Unlike every other navix environment, this one is not solvable
    from the current observation alone once the start room scrolls out
    of view - it requires actually carrying information across steps.

    `state.mission` stores only the success position - `failure_pos`
    is never a separate field, since it's always the mirror image of
    `success_pos` across the hallway's centre row (`height // 2`), and
    is derived algebraically in `events.on_memory_failure` instead
    (the same choice PutNear's `move`/`target` two-mission-target case
    made, for the same reason: avoid a second mission slot when the
    second position is always recoverable from the first)."""

    random_length: bool = struct.field(pytree_node=False, default=False)

    def _reset(self, key: Array, cache: Union[RenderingCache, None] = None) -> Timestep:
        height, width = self.height, self.width
        assert height % 2 == 1, f"Memory requires an odd height, got {height}"
        assert width > 6, f"Memory requires width > 6, got {width}"

        k1, k2, k3, k4 = jax.random.split(key, num=4)

        upper_room_wall = height // 2 - 2
        lower_room_wall = height // 2 + 2

        if self.random_length:
            hallway_end = jax.random.randint(k1, (), minval=4, maxval=width - 2)
        else:
            hallway_end = jnp.asarray(width - 3)

        # map: bordered room, plus the interior start-room/hallway/
        # T-split walls. hallway_end may be a per-episode random value
        # (random_length=True), so these are vectorised boolean
        # conditions over the whole grid rather than python loops with
        # dynamic bounds.
        grid = room(height=height, width=width)
        rows, cols = coordinates(grid)

        start_room_wall = jnp.logical_and(
            jnp.logical_and(cols >= 1, cols < 5),
            jnp.logical_or(rows == upper_room_wall, rows == lower_room_wall),
        )
        start_room_corner = jnp.logical_or(
            jnp.logical_and(rows == upper_room_wall + 1, cols == 4),
            jnp.logical_and(rows == lower_room_wall - 1, cols == 4),
        )
        horizontal_hallway_wall = jnp.logical_and(
            jnp.logical_and(cols >= 5, cols < hallway_end),
            jnp.logical_or(
                rows == upper_room_wall + 1, rows == lower_room_wall - 1
            ),
        )
        vertical_hallway_wall = jnp.logical_or(
            jnp.logical_and(cols == hallway_end, rows != height // 2),
            cols == hallway_end + 2,
        )
        is_wall = (
            start_room_wall
            | start_room_corner
            | horizontal_hallway_wall
            | vertical_hallway_wall
        )
        all_positions = jnp.stack([rows.ravel(), cols.ravel()], axis=1)
        wall_positions = jnp.where(
            is_wall.ravel()[:, None], all_positions, DISCARD_PILE_COORDS
        )
        walls = Wall.create(position=wall_positions)

        # player: random column in the start room/hallway, centre row,
        # always facing east (toward the hallway exit)
        player_col = jax.random.randint(k2, (), minval=1, maxval=hallway_end + 1)
        player_pos = jnp.asarray([height // 2, player_col])
        player = Player.create(
            position=player_pos,
            direction=Directions.EAST,
            pocket=EMPTY_POCKET_ID,
        )

        # objects: one in the start room, one at each hallway end. The
        # two hallway-end objects are always one Key + one Ball (order
        # random); the start-room object is a second copy of whichever
        # type the agent must remember, matching one of the two ends.
        start_is_ball = jax.random.bernoulli(k3)
        pos0_is_ball = jax.random.bernoulli(k4)

        start_pos = jnp.asarray([height // 2 - 1, 1])
        pos0 = jnp.asarray([height // 2 - 2, hallway_end + 1])
        pos1 = jnp.asarray([height // 2 + 2, hallway_end + 1])

        key_end_pos = jnp.where(pos0_is_ball, pos1, pos0)
        ball_end_pos = jnp.where(pos0_is_ball, pos0, pos1)

        key_positions = jnp.stack(
            [
                key_end_pos,
                jnp.where(start_is_ball, DISCARD_PILE_COORDS, start_pos),
            ]
        ).astype(jnp.int32)
        ball_positions = jnp.stack(
            [
                ball_end_pos,
                jnp.where(start_is_ball, start_pos, DISCARD_PILE_COORDS),
            ]
        ).astype(jnp.int32)
        green = jnp.asarray([1, 1], dtype=jnp.uint8)
        keys = Key.create(
            position=key_positions, colour=green, id=jnp.asarray([1, 2])
        )
        balls = Ball.create(
            position=ball_positions,
            colour=green,
            probability=jnp.ones(2),
            id=jnp.asarray([1, 2]),
        )

        # mission: only the success position is stored - failure_pos
        # is derived in events.on_memory_failure by mirroring it across
        # the hallway's centre row (height // 2)
        success_row = jnp.where(
            start_is_ball == pos0_is_ball, height // 2 - 1, height // 2 + 1
        )
        success_pos = jnp.asarray(
            [success_row, hallway_end + 1], dtype=jnp.int32
        )
        mission = Event(
            position=success_pos,
            colour=jnp.asarray(1, dtype=jnp.uint8),
            happened=jnp.asarray(False),
        )

        entities = {
            Entities.PLAYER: player[None],
            Entities.WALL: walls,
            Entities.KEY: keys,
            Entities.BALL: balls,
        }

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
            action=jnp.asarray(0, dtype=jnp.int32),
            reward=jnp.asarray(0.0, dtype=jnp.float32),
            step_type=jnp.asarray(0, dtype=jnp.int32),
            state=state,
        )


def _register_memory(env_id: str, size: int, random_length: bool) -> None:
    register_env(
        env_id,
        lambda *args, **kwargs: Memory.create(
            height=size,
            width=size,
            random_length=random_length,
            max_steps=kwargs.pop("max_steps", 5 * size**2),
            action_set=MEMORY_ACTION_SET,
            transitions_fn=kwargs.pop(
                "transitions_fn", transitions.deterministic_transition
            ),
            observation_fn=kwargs.pop("observation_fn", observations.symbolic),
            reward_fn=kwargs.pop("reward_fn", rewards.on_memory_success),
            termination_fn=kwargs.pop("termination_fn", MEMORY_TERMINATION_FN),
            *args,
            **kwargs,
        ),
    )


_register_memory("Navix-MemoryS17Random-v0", size=17, random_length=True)
_register_memory("Navix-MemoryS13Random-v0", size=13, random_length=True)
_register_memory("Navix-MemoryS13-v0", size=13, random_length=False)
_register_memory("Navix-MemoryS11-v0", size=11, random_length=False)
_register_memory("Navix-MemoryS9-v0", size=9, random_length=False)
_register_memory("Navix-MemoryS7-v0", size=7, random_length=False)
