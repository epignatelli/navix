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

"""`ObstructedMaze` (issue #183): the outlier of the remaining
families, per that issue's own scoping - it needs a real `RoomGrid`
(#174, now built generically in `grid.py`'s `room_grid`/`room_mask`/
`room_grid_door_position`/etc, reusable by `MultiRoom`/#182 too) plus
a wholly new mechanic (`actions.open_box`, opening a `Box` to reveal a
hidden `Key`). Per the issue's staged-rollout recommendation, this
file starts with only the smallest variant, `ObstructedMaze1Dlhb` (a
1x2 room grid) - the other 12 registrations are deliberately deferred
to follow-up work, once this one is confirmed to play correctly end
to end."""

from __future__ import annotations
from typing import Union

import jax
import jax.numpy as jnp
from jax import Array
from flax import struct

from navix import observations

from .. import rewards, terminations, transitions
from ..components import DISCARD_PILE_COORDS, EMPTY_POCKET_ID
from ..entities import Ball, Box, Door, Entities, Key, Player
from ..states import State, Event
from ..grid import (
    open_wall,
    random_colour,
    random_directions,
    random_positions,
    room_grid,
    room_grid_dims,
    room_grid_door_position,
    room_mask,
)
from ..rendering.cache import RenderingCache
from .environment import Environment, Timestep
from .registry import register_env


ROOM_SIZE = 6  # fixed across every real ObstructedMaze variant


class ObstructedMaze1Dlhb(Environment):
    """A 1x2 `room_grid`: the agent starts in the left room, a target
    `Ball` sits in the right room, behind a locked `Door`. Optionally
    the door's key is hidden inside a `Box` in the left room (must be
    opened with `toggle` to reveal it), and/or a second `Ball` blocks
    the door on the player's own side.

    The `key_in_box`/`blocked` flags are exactly MiniGrid's own - the
    three `1D*` env ids are all the same class there too, registered
    with different values (verified against MiniGrid's actual
    registrations, `room_size=6` throughout):

    | env id      | `key_in_box` | `blocked` |
    |-------------|--------------|-----------|
    | `1Dl-v0`    | `False`      | `False`   |
    | `1Dlh-v0`   | `True`       | `False`   |
    | `1Dlhb-v0`  | `True`       | `True`    |

    Both are static (`pytree_node=False`) fields, so they branch at
    trace-construction time - each registered id gets its own entity
    layout (no `Box` at all when `key_in_box=False`, one `Ball`
    instead of two when `blocked=False`), rather than padding unused
    slots. Same convention as `unlock.py`'s `block_door`.

    Reward + termination only on picking up the *target* ball - the
    blocking ball can be picked up too (nothing in `actions.pickup`
    prevents it, matching real MiniGrid) without ending the episode,
    since `terminations`/`rewards.on_target_fetched` key off
    `state.mission`'s specific tracked position."""

    key_in_box: bool = struct.field(pytree_node=False, default=True)
    blocked: bool = struct.field(pytree_node=False, default=True)

    def _reset(self, key: Array, cache: Union[RenderingCache, None] = None) -> Timestep:
        (
            key,
            k_door_pos,
            k_door_colour,
            k_box_colour,
            k_block_colour,
            k_target_colour,
            k_player_pos,
            k_player_dir,
            k_box_pos,
            k_target_pos,
        ) = jax.random.split(key, num=10)

        grid = room_grid(ROOM_SIZE, num_rows=1, num_cols=2)
        door_pos = room_grid_door_position(k_door_pos, ROOM_SIZE, 0, 0, side=0)  # east
        grid = open_wall(grid, door_pos)

        door_colour = random_colour(k_door_colour)
        doors = Door.create(
            position=door_pos,
            requires=jnp.asarray(1),
            colour=door_colour,
            open=jnp.asarray(False),
        )

        start_room = jnp.where(room_mask(grid, ROOM_SIZE, 0, 0), grid, -1)
        target_room = jnp.where(room_mask(grid, ROOM_SIZE, 0, 1), grid, -1)

        # the blocking ball sits one cell into the start room, adjacent
        # to the door (verified: MiniGrid places it at `door_pos -
        # DIR_TO_VEC[door_idx]` - one step back the way the door faces).
        # It is the door's *only* interior-adjacent cell, so when
        # `blocked` it really does gate the whole route.
        block_pos = door_pos - jnp.asarray([0, 1])
        # nothing to keep clear when there is no blocking ball -
        # DISCARD_PILE_COORDS is off-grid, so it excludes nothing
        # (same no-op-exclude convention as unlock.py's block_door)
        keep_clear = block_pos if self.blocked else DISCARD_PILE_COORDS

        player_pos = random_positions(k_player_pos, start_room, exclude=keep_clear)
        player_dir = random_directions(k_player_dir)
        player = Player.create(
            position=player_pos, direction=player_dir, pocket=EMPTY_POCKET_ID
        )

        # the key is either hidden in a box or lying in the start room
        key_or_box_pos = random_positions(
            k_box_pos, start_room, exclude=jnp.stack([player_pos, keep_clear])
        )
        if self.key_in_box:
            boxes = Box.create(
                position=key_or_box_pos,
                colour=random_colour(k_box_colour),
                id=jnp.asarray(2),
                pocket=jnp.asarray(1),  # references the key's id below
            )
            # the key exists only conceptually "inside" the box until
            # opened (verified against MiniGrid's actual Box.toggle:
            # the key is not a separate grid object until then) -
            # starts at the discard pile, `actions.open_box` moves it
            # to the box's former position once revealed.
            key_pos = DISCARD_PILE_COORDS
        else:
            key_pos = key_or_box_pos
        keys = Key.create(position=key_pos, id=jnp.asarray(1), colour=door_colour)

        target_pos = random_positions(k_target_pos, target_room)
        target_colour = random_colour(k_target_colour)
        if self.blocked:
            balls = Ball.create(
                position=jnp.stack([block_pos, target_pos]),
                colour=jnp.stack([random_colour(k_block_colour), target_colour]),
                probability=jnp.ones(2),
                id=jnp.asarray([3, 4]),
            )
        else:
            balls = Ball.create(
                position=target_pos[None],
                colour=target_colour[None],
                probability=jnp.ones(1),
                id=jnp.asarray([4]),
            )

        mission = Event(
            position=target_pos,
            colour=target_colour,
            happened=jnp.asarray(False),
        )

        entities = {
            Entities.PLAYER: player[None],
            Entities.DOOR: doors[None],
            Entities.KEY: keys[None],
            Entities.BALL: balls,
        }
        if self.key_in_box:
            entities[Entities.BOX] = boxes[None]

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


_1D_HEIGHT, _1D_WIDTH = room_grid_dims(ROOM_SIZE, num_rows=1, num_cols=2)


def _register_1d(env_id: str, key_in_box: bool, blocked: bool) -> None:
    register_env(
        env_id,
        lambda *args, **kwargs: ObstructedMaze1Dlhb.create(
            height=_1D_HEIGHT,
            width=_1D_WIDTH,
            key_in_box=key_in_box,
            blocked=blocked,
            # 4 * num_rooms_visited * room_size**2, verified against
            # MiniGrid's own formula (2 rooms here -> 288)
            max_steps=kwargs.pop("max_steps", 4 * 2 * ROOM_SIZE**2),
            transitions_fn=kwargs.pop(
                "transitions_fn", transitions.deterministic_transition
            ),
            observation_fn=kwargs.pop("observation_fn", observations.symbolic),
            reward_fn=kwargs.pop("reward_fn", rewards.on_target_fetched),
            termination_fn=kwargs.pop("termination_fn", terminations.on_target_fetched),
            *args,
            **kwargs,
        ),
    )


_register_1d("Navix-ObstructedMaze-1Dl-v0", key_in_box=False, blocked=False)
_register_1d("Navix-ObstructedMaze-1Dlh-v0", key_in_box=True, blocked=False)
_register_1d("Navix-ObstructedMaze-1Dlhb-v0", key_in_box=True, blocked=True)
