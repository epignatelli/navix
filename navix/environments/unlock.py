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

"""`Unlock` and `UnlockPickup` - the simplest case of MiniGrid's
`RoomGrid` (`num_rows=1, num_cols=2`, a single door connecting two
equal-sized rooms). See issue #10/#175/#176 - a full `RoomGrid` port
(arbitrary `num_rows`/`num_cols`, issue #174) is deliberately not
needed for either of these two environments, so this file builds the
1x2 layout directly rather than a general room-grid primitive."""

from typing import Dict, Tuple, Union

import jax
import jax.numpy as jnp
from jax import Array

from navix import observations, rewards, terminations

from ..components import EMPTY_POCKET_ID
from ..entities import Box, Door, Entities, Entity, Key, Player
from ..grid import mask_by_coordinates, random_colour, random_directions, random_positions
from ..rendering.cache import RenderingCache
from ..states import State
from . import Environment, Timestep
from .registry import register_env


def two_equal_rooms_with_door(
    key: Array, height: int, width: int
) -> Tuple[Array, Array, int, Dict[str, Entity], Array, Array]:
    """Builds the shared layout `Unlock`/`UnlockPickup` both start from:
    two `height x height`-square rooms side by side (`wall_col =
    height - 1` apart, `width == 2 * (height - 1) + 1`), connected by
    one locked `Door` at a random row, with a matching `Key` and the
    agent both placed randomly in the first (left) room. Mirrors
    MiniGrid's `RoomGrid(room_size=height, num_rows=1, num_cols=2)` -
    `add_door(0, 0, door_idx=0, locked=True)` +
    `add_object(0, 0, "key", door.color)` + `place_agent(0, 0)`.

    Args:
        key (Array): PRNG key.
        height (int): Height of each room (MiniGrid's `room_size`).
        width (int): Must be `2 * (height - 1) + 1`.

    Returns:
        Tuple[Array, ...]: `(key, grid, wall_col, entities_dict, first_room,
        door_colour)` - `entities_dict` has `Entities.PLAYER`/`KEY`/`DOOR`
        already batched (`[None]`-indexed) and ready to merge into a
        `State`; `first_room` (the left room's own `grid`-shaped mask,
        walls everywhere outside it) is returned too, so `UnlockPickup`
        can build its own `second_room` mask from the same `wall_col`
        without recomputing the base grid.
    """
    assert (
        width == 2 * (height - 1) + 1
    ), f"width must be 2 * (height - 1) + 1 for two equal rooms, got height={height}, width={width}"
    # height == 3 degenerates: the left room's interior collapses to a
    # single walkable cell, so the player and key can't both fit -
    # random_positions then places the key outside the room entirely
    # (verified empirically: 20/20 seeds at height=3 misplace the key at
    # a wall/border cell). height >= 4 leaves at least a 2-cell interior.
    assert height >= 4, f"height (room size) must be >= 4, got {height}"

    key, k_color, k_door_row, k_player_pos, k_player_dir, k_key_pos = jax.random.split(
        key, 6
    )

    grid = jnp.zeros((height - 2, width - 2), dtype=jnp.int32)
    grid = jnp.pad(grid, 1, mode="constant", constant_values=-1)
    wall_col = height - 1
    grid = grid.at[:, wall_col].set(-1)

    # one door, at a random row within the shared wall's interior span
    door_row = jax.random.randint(k_door_row, (), 1, height - 1)
    grid = grid.at[door_row, wall_col].set(0)

    door_colour = random_colour(k_color)
    door_pos = jnp.asarray([door_row, wall_col])
    doors = Door.create(
        position=door_pos,
        requires=jnp.asarray(1),
        colour=door_colour,
        open=jnp.asarray(False),
    )

    first_room_mask = mask_by_coordinates(
        grid, (jnp.asarray(height), jnp.asarray(wall_col)), jnp.less
    )
    first_room = jnp.where(first_room_mask, grid, -1)

    player_pos = random_positions(k_player_pos, first_room)
    player_dir = random_directions(k_player_dir)
    player = Player.create(
        position=player_pos, direction=player_dir, pocket=EMPTY_POCKET_ID
    )

    key_pos = random_positions(k_key_pos, first_room, exclude=player_pos)
    keys = Key.create(position=key_pos, id=jnp.asarray(1), colour=door_colour)

    entities = {
        Entities.PLAYER: player[None],
        Entities.KEY: keys[None],
        Entities.DOOR: doors[None],
    }
    return key, grid, wall_col, entities, first_room, door_colour


class Unlock(Environment):
    """The agent has to open a locked door - a matching key is in the
    same (left) room; the other (right) room is otherwise empty.
    Reward + termination on opening the door (see `rewards.on_door_open`/
    `terminations.on_door_open`) - not on reaching any further goal,
    opening the door *is* the success condition."""

    def _reset(self, key: Array, cache: Union[RenderingCache, None] = None) -> Timestep:
        key, grid, _, entities, _, _ = two_equal_rooms_with_door(
            key, self.height, self.width
        )

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


class UnlockPickup(Environment):
    """`Unlock`, plus a `Box` placed in the second (right) room - the
    agent must unlock the door, then pick up the box. Reward +
    termination on picking up the box (see `rewards.on_box_pickup`/
    `terminations.on_box_pickup`)."""

    def _reset(self, key: Array, cache: Union[RenderingCache, None] = None) -> Timestep:
        key, grid, wall_col, entities, _, _ = two_equal_rooms_with_door(
            key, self.height, self.width
        )

        key, k_box_pos, k_box_colour = jax.random.split(key, 3)
        second_room_mask = mask_by_coordinates(
            grid, (jnp.asarray(0), jnp.asarray(wall_col)), jnp.greater
        )
        second_room = jnp.where(second_room_mask, grid, -1)
        box_pos = random_positions(k_box_pos, second_room)
        boxes = Box.create(
            position=box_pos,
            colour=random_colour(k_box_colour),
            id=jnp.asarray(2),
            pocket=jnp.asarray(-1),
        )
        entities[Entities.BOX] = boxes[None]

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
    "Navix-Unlock-v0",
    lambda *args, **kwargs: Unlock.create(
        height=6,
        width=11,
        observation_fn=kwargs.pop("observation_fn", observations.symbolic),
        reward_fn=kwargs.pop("reward_fn", rewards.on_door_open),
        termination_fn=kwargs.pop("termination_fn", terminations.on_door_open),
        *args,
        **kwargs,
    ),
)
register_env(
    "Navix-UnlockPickup-v0",
    lambda *args, **kwargs: UnlockPickup.create(
        height=6,
        width=11,
        observation_fn=kwargs.pop("observation_fn", observations.symbolic),
        reward_fn=kwargs.pop("reward_fn", rewards.on_box_pickup),
        termination_fn=kwargs.pop("termination_fn", terminations.on_box_pickup),
        *args,
        **kwargs,
    ),
)
