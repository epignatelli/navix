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

"""`Playground` (issue #184): a 3x3 grid of rooms, each pair of
orthogonally-adjacent rooms joined by exactly one randomly-coloured,
randomly-positioned door - never locked. 12 objects, each independently
`Key`/`Ball`/`Box` (matching MiniGrid's own per-object random type draw -
see `GoToObject`'s module docstring for the same `DISCARD_PILE_COORDS`
fixed-slot-allocation trick this reuses), are scattered across the whole
grid.

No `Goal`, no reward beyond a flat 0, no termination beyond the timeout -
registers with `rewards.free` (see this PR's fix to that function's
signature) and `termination_fn=terminations.compose()` (no functions ->
`jnp.any` of an empty array -> always `False`, i.e. never terminates;
`check_truncation` still truncates at `max_steps` regardless).

Leaving `termination_fn` at `Environment.create`'s own default
(`DEFAULT_TERMINATION`, composing `on_goal_reached`/`on_lava_fall`/
`on_ball_hit`) was tried first and is wrong: `on_goal_reached`/
`on_lava_fall` do stay permanently `False` (`EventsManager.happened`,
see `states.py`, returns `False` rather than raising for an event slot
whose entity type was never in `entities` at all - no `Goal`/`Lava` ever
placed here) - but `on_ball_hit` does not. `actions._can_walk_there`
(called from every `forward` attempt, regardless of `transitions_fn`)
records a `(Entities.BALL, EventType.HIT)` event through the generic
`record_walk_into` dispatch whenever the player merely *bumps* a `Ball`
- not just when a moving ball collides with the player via `transitions.
update_balls` (which this environment correctly never runs - see next
paragraph). Confirmed directly: a `DEFAULT_TERMINATION`-registered
version of this environment ended episodes within the first 20 random
steps, well before `max_steps=100`, every time a random walk happened to
bump one of the 12 objects' `Ball` share.

Registers with `transitions_fn=transitions.deterministic_transition`,
matching `GoToObject`/`PutNear`/`Fetch`: without it, the default
`stochastic_transition` would walk every `Ball` around the room turn by
turn, unlike MiniGrid's actual static `Ball`/`Key`/`Box` placement here.

MiniGrid's own `PlaygroundEnv` hardcodes a fixed 19x19 grid regardless of
any `size` argument (verified against source: `self.size = 19` is set
unconditionally before `super().__init__`, no `size` kwarg even accepted)
and a flat `max_steps=100` default (not the usual
`4 * height * width`-style formula) - both reproduced as-is here."""

from __future__ import annotations
from typing import List, Tuple, Union

import jax
import jax.numpy as jnp
from jax import Array

from navix import observations, rewards, terminations, transitions

from ..components import DISCARD_PILE_COORDS, EMPTY_POCKET_ID
from ..entities import Ball, Box, Door, Entities, Key, Player
from ..states import State
from ..grid import (
    open_wall,
    random_colour,
    random_directions,
    random_distinct_positions,
    room,
)
from ..rendering.cache import RenderingCache
from .environment import Environment, Timestep
from .registry import register_env


GRID_SIZE = 19  # fixed regardless of registration - matches MiniGrid's own
NUM_OBJECTS = 12  # matches MiniGrid's own fixed object count


def wall_segments(
    height: int, width: int
) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
    """The wall segments splitting a `height`x`width` grid into a 3x3
    layout of rooms - `(col, row_top, row_bottom)` for each vertical
    segment, `(row, col_left, col_right)` for each horizontal one. A
    fixed function of `height`/`width` alone (only each segment's own
    door offset and colour are randomised per episode, in `_reset`) -
    verified against MiniGrid's actual `PlaygroundEnv._gen_grid`."""
    room_h, room_w = height // 3, width // 3
    vertical: List[Tuple[int, int, int]] = []
    horizontal: List[Tuple[int, int, int]] = []
    for j in range(3):
        for i in range(3):
            x_l, y_t = i * room_w, j * room_h
            x_r, y_b = x_l + room_w, y_t + room_h
            if i + 1 < 3:
                vertical.append((x_r, y_t, y_b))
            if j + 1 < 3:
                horizontal.append((y_b, x_l, x_r))
    return vertical, horizontal


class Playground(Environment):
    def _reset(self, key: Array, cache: Union[RenderingCache, None] = None) -> Timestep:
        vertical_walls, horizontal_walls = wall_segments(self.height, self.width)
        room_size = self.height // 3  # square rooms - == self.width // 3 too

        grid = room(height=self.height, width=self.width)
        for col, row_top, row_bottom in vertical_walls:
            grid = grid.at[row_top:row_bottom, col].set(-1)
        for row, col_left, col_right in horizontal_walls:
            grid = grid.at[row, col_left:col_right].set(-1)

        (
            key,
            k_door_v,
            k_door_h,
            k_door_colour,
            k_positions,
            k_player_dir,
            k_types,
            k_key_colour,
            k_ball_colour,
            k_box_colour,
        ) = jax.random.split(key, 10)

        v_cols = jnp.asarray([w[0] for w in vertical_walls], dtype=jnp.int32)
        v_row_tops = jnp.asarray([w[1] for w in vertical_walls], dtype=jnp.int32)
        v_offsets = jax.random.randint(k_door_v, (len(vertical_walls),), 1, room_size - 1)
        door_v_positions = jnp.stack([v_row_tops + v_offsets, v_cols], axis=1)

        h_rows = jnp.asarray([w[0] for w in horizontal_walls], dtype=jnp.int32)
        h_col_lefts = jnp.asarray([w[1] for w in horizontal_walls], dtype=jnp.int32)
        h_offsets = jax.random.randint(k_door_h, (len(horizontal_walls),), 1, room_size - 1)
        door_h_positions = jnp.stack([h_rows, h_col_lefts + h_offsets], axis=1)

        door_positions = jnp.concatenate([door_v_positions, door_h_positions], axis=0)
        num_doors = door_positions.shape[0]
        for door_pos in door_positions:
            grid = open_wall(grid, door_pos)

        doors = Door.create(
            position=door_positions,
            requires=jnp.full((num_doors,), EMPTY_POCKET_ID, dtype=jnp.int32),
            colour=random_colour(k_door_colour, n=num_doors),
            open=jnp.zeros((num_doors,), dtype=jnp.bool_),
        )

        # player + NUM_OBJECTS objects, all mutually distinct and none on
        # a door cell (door cells are ordinary floor in `grid` itself, so
        # only `exclude` keeps entities off them)
        positions = random_distinct_positions(
            k_positions, grid, n=1 + NUM_OBJECTS, exclude=door_positions
        )
        player = Player.create(
            position=positions[0],
            direction=random_directions(k_player_dir),
            pocket=EMPTY_POCKET_ID,
        )
        object_positions = positions[1:]

        # each object's type drawn independently, 0=Key/1=Ball/2=Box -
        # see module docstring: every slot is allocated for every type,
        # non-matching slots pushed off-grid (GoToObject's own pattern)
        type_idx = jax.random.randint(k_types, (NUM_OBJECTS,), 0, 3)
        is_key, is_ball, is_box = type_idx == 0, type_idx == 1, type_idx == 2
        key_pos = jnp.where(is_key[:, None], object_positions, DISCARD_PILE_COORDS)
        ball_pos = jnp.where(is_ball[:, None], object_positions, DISCARD_PILE_COORDS)
        box_pos = jnp.where(is_box[:, None], object_positions, DISCARD_PILE_COORDS)

        keys = Key.create(
            position=key_pos,
            colour=random_colour(k_key_colour, n=NUM_OBJECTS),
            id=jnp.arange(1, NUM_OBJECTS + 1, dtype=jnp.int32),
        )
        balls = Ball.create(
            position=ball_pos,
            colour=random_colour(k_ball_colour, n=NUM_OBJECTS),
            probability=jnp.ones(NUM_OBJECTS),
            id=jnp.arange(NUM_OBJECTS + 1, 2 * NUM_OBJECTS + 1, dtype=jnp.int32),
        )
        boxes = Box.create(
            position=box_pos,
            colour=random_colour(k_box_colour, n=NUM_OBJECTS),
            id=jnp.arange(2 * NUM_OBJECTS + 1, 3 * NUM_OBJECTS + 1, dtype=jnp.int32),
            pocket=jnp.full((NUM_OBJECTS,), EMPTY_POCKET_ID, dtype=jnp.int32),
        )

        entities = {
            Entities.PLAYER: player[None],
            Entities.DOOR: doors,
            Entities.KEY: keys,
            Entities.BALL: balls,
            Entities.BOX: boxes,
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
    "Navix-Playground-v0",
    lambda *args, **kwargs: Playground.create(
        height=GRID_SIZE,
        width=GRID_SIZE,
        max_steps=kwargs.pop("max_steps", 100),
        observation_fn=kwargs.pop("observation_fn", observations.symbolic),
        reward_fn=kwargs.pop("reward_fn", rewards.free),
        termination_fn=kwargs.pop("termination_fn", terminations.compose()),
        transitions_fn=kwargs.pop("transitions_fn", transitions.deterministic_transition),
        *args,
        **kwargs,
    ),
)
