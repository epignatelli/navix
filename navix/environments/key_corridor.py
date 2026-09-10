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

"""MiniGrid's KeyCorridor environment - find a key off a corridor to reach a locked room.

See the environment class in this module for the task, layout and
reward/termination details.
"""


from __future__ import annotations

from typing import List, Tuple, Union
import jax
import jax.numpy as jnp
from jax import Array

from navix import observations, rewards, terminations

from ..components import DISCARD_PILE_COORDS, EMPTY_POCKET_ID
from ..rendering.cache import RenderingCache
from ..environments import Environment
from ..entities import Goal, Player, Key, Door
from ..states import State
from ..environments import Timestep
from ..grid import random_directions, random_colour, RoomsGrid
from .registry import register_env


def _room_id(row: int, col: int) -> int:
    return row * 3 + col


class KeyCorridor(Environment):
    """Find a key hidden behind unlocked doors, then use it to open the one
    locked door guarding the goal.

    Rooms form an `n_rows x 3` grid: the middle column is one corridor (its
    inter-row walls are removed), the key lives in a left-column room, and
    the goal lives in a right-column room behind the episode's one locked
    door.

    Every other door follows MiniGrid's `RoomGrid.connect_all`, which draws
    walls uniformly with replacement and puts a closed door on each new one
    that does not touch the locked room, stopping as soon as every room is
    reachable. The order in which distinct eligible walls are first drawn is
    a uniform permutation, so the doors it adds are exactly the shortest
    prefix of a random permutation of eligible walls that connects the rooms
    - walls between rooms that were already connected included. `_reset`
    computes that prefix in one fixed-length pass, counting components with a
    union-find.

    `jax.jit` needs a fixed entity count, so every candidate wall is a `Door`
    entity. The ones `connect_all` never reaches sit at `DISCARD_PILE_COORDS`
    and leave their cell a wall, so they render, encode and block exactly as
    MiniGrid's plain walls do.
    """

    def _reset(self, key: Array, cache: Union[RenderingCache, None] = None) -> Timestep:
        n_rows_config = {3: 1, 5: 2}
        n_rows = n_rows_config.get(self.height, 3)
        room_size = (self.width - 3) // 3
        k1, k2, k3, k4, k5, k6 = jax.random.split(key, num=6)

        # grid of rooms
        grid = RoomsGrid.create(n_rows, 3, (room_size, room_size))

        # key
        key_room_row = jax.random.randint(k1, (), minval=0, maxval=n_rows)
        key_pos = grid.position_in_room(
            key_room_row, jnp.asarray(0, dtype=jnp.int32), key=k1
        )
        key_colour = random_colour(k4)
        key_id = jnp.asarray(1)
        key_obj = Key.create(key_pos, key_colour, key_id)

        # agent
        pk_1, pk_2, pk_3 = jax.random.split(k2, num=3)
        agent_room_row = jax.random.randint(pk_1, (), minval=0, maxval=n_rows)
        agent_pos = grid.position_in_room(agent_room_row, jnp.asarray(1), key=pk_2)
        player = Player.create(
            agent_pos, random_directions(pk_3), pocket=EMPTY_POCKET_ID
        )

        # goal
        goal_room_row = jax.random.randint(k3, (), minval=0, maxval=n_rows)
        goal_pos = grid.position_in_room(goal_room_row, jnp.asarray(2), key=k4)
        goal = Goal.create(goal_pos, probability=jnp.asarray(1.0))

        # Doors: connect_all - see the class docstring. `parent` (union-find)
        # is kept fully flat as an invariant: a union from a flat state only
        # ever strands nodes one hop further off, so a single
        # `parent[parent]` pass after each union always restores it, and a
        # lookup is one gather however many rooms have merged.
        num_rooms = 3 * n_rows
        # (row, u, v, col, side) - `col`/`side` are `position_on_border`'s
        # own args for this wall; `u`/`v` are the two rooms it connects.
        candidates: List[Tuple[int, int, int, int, int]] = []
        for row in range(n_rows):
            candidates.append((row, _room_id(row, 0), _room_id(row, 1), 0, 1))
            candidates.append((row, _room_id(row, 1), _room_id(row, 2), 2, 0))
        for row in range(n_rows - 1):
            candidates.append((row, _room_id(row, 0), _room_id(row + 1, 0), 0, 3))
            candidates.append((row, _room_id(row, 2), _room_id(row + 1, 2), 2, 3))
        num_candidates = len(candidates)

        door_keys = jax.random.split(k5, num=num_candidates + 2)
        positions = jnp.stack(
            [
                grid.position_on_border(row, col, side, key=door_keys[i])
                for i, (row, _, _, col, side) in enumerate(candidates)
            ]
        )
        colours = random_colour(door_keys[num_candidates], num_candidates)
        perm = jax.random.permutation(door_keys[num_candidates + 1], num_candidates)

        row_ids = jnp.asarray([row for row, *_ in candidates])
        u_ids = jnp.asarray([u for _, u, _, _, _ in candidates])
        v_ids = jnp.asarray([v for _, _, v, _, _ in candidates])
        # the (row, col=1)<->(row, col=2) candidate, for whichever row
        # turns out to be the goal row - the one mandatory locked door.
        is_goal_slot = (u_ids % 3 == 1) & (v_ids % 3 == 2) & (row_ids == goal_room_row)

        # The middle column's inter-row walls are unconditionally carved
        # below regardless of doors, so every (row, col=1) room is
        # already one component - point them all directly at row 0's
        # (already flat, no unions/lookups needed to build this).
        parent = jnp.arange(num_rooms)
        corridor_root = _room_id(0, 1)
        col1_ids = jnp.asarray([_room_id(row, 1) for row in range(n_rows)])
        parent = parent.at[col1_ids].set(corridor_root)
        # the mandatory locked door: the goal row's col=2 room joins the
        # (already-flat) corridor component in one hop.
        locked_room = goal_room_row * 3 + 2
        parent = parent.at[locked_room].set(corridor_root)

        eligible = (u_ids != locked_room) & (v_ids != locked_room)
        # the corridor merges `n_rows` rooms into one, the locked door one more
        components = jnp.asarray(num_rooms - n_rows)

        added = jnp.zeros((num_candidates,), dtype=jnp.bool_)
        for i in range(num_candidates):
            idx = perm[i]
            ru, rv = parent[u_ids[idx]], parent[v_ids[idx]]
            add = eligible[idx] & (components > 1)
            merge = add & (ru != rv)
            parent = parent.at[ru].set(jnp.where(merge, rv, ru))
            parent = parent[parent]  # one pointer-doubling pass restores flatness
            components = components - merge
            added = added.at[idx].set(add)

        on_grid = is_goal_slot | added
        door_colours = jnp.where(is_goal_slot, key_colour, colours)
        doors = Door.create(
            position=jnp.where(on_grid[:, None], positions, DISCARD_PILE_COORDS),
            requires=jnp.where(is_goal_slot, key_id, EMPTY_POCKET_ID),
            colour=door_colours,
            open=jnp.zeros((num_candidates,), dtype=jnp.int32),
        )
        # a wall `connect_all` never reached stays a wall: its carve goes to
        # row `self.height`, out of bounds, which `mode="drop"` discards
        carved = jnp.where(on_grid[:, None], positions, jnp.asarray([self.height, 0]))

        entities = {
            "player": player[None],
            "key": key_obj[None],
            "door": doors,
            "goal": goal[None],
        }

        grid = grid.get_grid().at[carved[:, 0], carved[:, 1]].set(0, mode="drop")
        grid = grid.at[
            1 + room_size : self.height - 1 : room_size + 1,
            1 + room_size + 1 : 1 + room_size + 1 + room_size,
        ].set(0)
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
    "Navix-KeyCorridorS3R1-v0",
    lambda *args, **kwargs: KeyCorridor.create(
        height=3,
        width=7,
        observation_fn=kwargs.pop("observation_fn", observations.symbolic),
        reward_fn=kwargs.pop("reward_fn", rewards.on_goal_reached),
        termination_fn=kwargs.pop("termination_fn", terminations.on_goal_reached),
        *args,
        **kwargs,
    ),
)
register_env(
    "Navix-KeyCorridorS3R2-v0",
    lambda *args, **kwargs: KeyCorridor.create(
        height=5,
        width=7,
        observation_fn=kwargs.pop("observation_fn", observations.symbolic),
        reward_fn=kwargs.pop("reward_fn", rewards.on_goal_reached),
        termination_fn=kwargs.pop("termination_fn", terminations.on_goal_reached),
        *args,
        **kwargs,
    ),
)
register_env(
    "Navix-KeyCorridorS3R3-v0",
    lambda *args, **kwargs: KeyCorridor.create(
        height=7,
        width=7,
        observation_fn=kwargs.pop("observation_fn", observations.symbolic),
        reward_fn=kwargs.pop("reward_fn", rewards.on_goal_reached),
        termination_fn=kwargs.pop("termination_fn", terminations.on_goal_reached),
        *args,
        **kwargs,
    ),
)
register_env(
    "Navix-KeyCorridorS4R3-v0",
    lambda *args, **kwargs: KeyCorridor.create(
        height=10,
        width=10,
        observation_fn=kwargs.pop("observation_fn", observations.symbolic),
        reward_fn=kwargs.pop("reward_fn", rewards.on_goal_reached),
        termination_fn=kwargs.pop("termination_fn", terminations.on_goal_reached),
        *args,
        **kwargs,
    ),
)
register_env(
    "Navix-KeyCorridorS5R3-v0",
    lambda *args, **kwargs: KeyCorridor.create(
        height=13,
        width=13,
        observation_fn=kwargs.pop("observation_fn", observations.symbolic),
        reward_fn=kwargs.pop("reward_fn", rewards.on_goal_reached),
        termination_fn=kwargs.pop("termination_fn", terminations.on_goal_reached),
        *args,
        **kwargs,
    ),
)
register_env(
    "Navix-KeyCorridorS6R3-v0",
    lambda *args, **kwargs: KeyCorridor.create(
        height=16,
        width=16,
        observation_fn=kwargs.pop("observation_fn", observations.symbolic),
        reward_fn=kwargs.pop("reward_fn", rewards.on_goal_reached),
        termination_fn=kwargs.pop("termination_fn", terminations.on_goal_reached),
        *args,
        **kwargs,
    ),
)
