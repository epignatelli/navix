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

"""`LockedRoom` (issue #179): six rooms - three stacked on the left of a
central north-south hallway, three stacked on the right - each opening
onto that hallway through its own door. One room is locked: its `Goal`
can only be reached by first finding a `Key` of the matching colour
hidden in one of the other five rooms, then unlocking that one door.

Faithful to MiniGrid's actual `LockedRoomEnv._gen_grid`: the six rooms'
positions, sizes and door cells are a fixed function of `height`/`width`
alone (not randomised, not retried - `left_wall`/`right_wall`/`room_w`/
`room_h`/the `j + 3` per-room door-row offset all reproduced exactly,
checked directly against source). Only which room is locked, the six
doors' colours, which room hides the key, and every entity's exact
position within its own room are randomised per episode. That makes
this a much simpler generation problem than `MultiRoom`'s (issue #182):
no `jax.lax.while_loop` retry search needed anywhere - every quantity
above is either a static Python value known at registration time or a
single vectorised random draw."""

from __future__ import annotations
from typing import List, Tuple, Union

import jax
import jax.numpy as jnp
from jax import Array

from navix import observations, rewards, terminations

from ..components import EMPTY_POCKET_ID
from ..entities import Door, Entities, Goal, Key, Player
from ..states import State
from ..grid import open_wall, random_directions, random_positions, room
from ..rendering.cache import RenderingCache
from .environment import Environment, Timestep
from .registry import register_env


NUM_ROOMS = 6  # 3 rows x {left, right} of the hallway - matches MiniGrid's
# own 6-colour palette exactly (one distinct door colour per room, no
# reuse), not a configurable parameter.


def room_layout(height: int, width: int) -> Tuple[Array, Array, Array, int, int]:
    """The six rooms' `(top_row, top_col)`, `(size_h, size_w)` and door
    `(row, col)` - a fixed function of `height`/`width` alone, verified
    against MiniGrid's actual `LockedRoomEnv._gen_grid`. Rooms are
    ordered left-row0, right-row0, left-row1, right-row1, left-row2,
    right-row2, matching MiniGrid's own `self.rooms` append order - the
    `locked_room`/`key_room` indices `_reset` draws index into this same
    order."""
    left_wall = width // 2 - 2
    right_wall = width // 2 + 2
    room_h = height // 3
    room_w = left_wall + 1  # MiniGrid reuses one `roomW` for both sides

    tops: List[Tuple[int, int]] = []
    sizes: List[Tuple[int, int]] = []
    doors: List[Tuple[int, int]] = []
    for n in range(3):
        j = n * room_h
        tops.append((j, 0))
        sizes.append((room_h + 1, room_w))
        doors.append((j + 3, left_wall))
        tops.append((j, right_wall))
        sizes.append((room_h + 1, width - right_wall))
        doors.append((j + 3, right_wall))

    return (
        jnp.asarray(tops, dtype=jnp.int32),
        jnp.asarray(sizes, dtype=jnp.int32),
        jnp.asarray(doors, dtype=jnp.int32),
        left_wall,
        right_wall,
    )


class LockedRoom(Environment):
    def _reset(self, key: Array, cache: Union[RenderingCache, None] = None) -> Timestep:
        tops, sizes, doors_pos, left_wall, right_wall = room_layout(self.height, self.width)

        # the hallway boundary walls (full height) and the 3 per-side
        # room-row dividers - see room_layout's docstring, same formulas
        grid = room(height=self.height, width=self.width)
        grid = grid.at[:, left_wall].set(-1)
        grid = grid.at[:, right_wall].set(-1)
        for n in range(3):
            j = n * (self.height // 3)
            grid = grid.at[j, 0:left_wall].set(-1)
            grid = grid.at[j, right_wall : self.width].set(-1)
        for door_pos in doors_pos:
            grid = open_wall(grid, door_pos)

        (
            key,
            k_locked,
            k_colours,
            k_key_room,
            k_row,
            k_col,
            k_player_pos,
            k_player_dir,
        ) = jax.random.split(key, 8)

        locked_room = jax.random.randint(k_locked, (), 0, NUM_ROOMS)
        # one of the other 5 rooms, uniformly - same nonzero-offset-mod-N
        # trick MultiRoom's door-colour selection uses for "exclude one
        # value"
        key_room = (
            locked_room + 1 + jax.random.randint(k_key_room, (), 0, NUM_ROOMS - 1)
        ) % NUM_ROOMS
        # a bijection: every room's door gets its own colour, none repeated
        colours = jax.random.permutation(k_colours, jnp.arange(NUM_ROOMS, dtype=jnp.uint8))

        # one random position inside *every* room's own interior at once
        # (vectorised, not just the two rooms that end up mattering) -
        # goal_pos/key_pos then gather whichever rows locked_room/key_room
        # pick; since those are always two different rooms, the two
        # gathered positions can never collide
        row_min, row_max = tops[:, 0] + 1, tops[:, 0] + sizes[:, 0] - 2
        col_min, col_max = tops[:, 1] + 1, tops[:, 1] + sizes[:, 1] - 2
        room_positions = jnp.stack(
            [
                jax.random.randint(k_row, (NUM_ROOMS,), row_min, row_max + 1),
                jax.random.randint(k_col, (NUM_ROOMS,), col_min, col_max + 1),
            ],
            axis=1,
        )
        goal_pos = room_positions[locked_room]
        key_pos = room_positions[key_room]
        key_id = jnp.asarray(1, dtype=jnp.int32)

        requires = jnp.where(jnp.arange(NUM_ROOMS) == locked_room, key_id, EMPTY_POCKET_ID)
        doors = Door.create(
            position=doors_pos,
            requires=requires,
            colour=colours,
            open=jnp.zeros(NUM_ROOMS, dtype=jnp.bool_),
        )

        goal = Goal.create(position=goal_pos, probability=jnp.asarray(1.0))
        key_obj = Key.create(position=key_pos, colour=colours[locked_room], id=key_id)

        # the player always starts somewhere in the corridor itself, not
        # in any room - matches MiniGrid's own `place_agent(top=
        # (left_wall, 0), size=(right_wall - left_wall, height))`
        hallway_col = (jnp.arange(self.width) > left_wall) & (
            jnp.arange(self.width) < right_wall
        )
        player_grid = jnp.where(hallway_col[None, :], grid, -1)
        player_pos = random_positions(k_player_pos, player_grid)
        player = Player.create(
            position=player_pos,
            direction=random_directions(k_player_dir),
            pocket=EMPTY_POCKET_ID,
        )

        entities = {
            Entities.PLAYER: player[None],
            Entities.DOOR: doors,
            Entities.KEY: key_obj[None],
            Entities.GOAL: goal[None],
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
    "Navix-LockedRoom-v0",
    lambda *args, **kwargs: LockedRoom.create(
        height=19,
        width=19,
        max_steps=kwargs.pop("max_steps", 10 * 19),
        observation_fn=kwargs.pop("observation_fn", observations.symbolic),
        reward_fn=kwargs.pop("reward_fn", rewards.on_goal_reached),
        termination_fn=kwargs.pop("termination_fn", terminations.on_goal_reached),
        *args,
        **kwargs,
    ),
)
