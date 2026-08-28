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

from typing import List, Tuple, Union
import jax
import jax.numpy as jnp
from jax import Array

from navix import observations, rewards, terminations

from ..components import EMPTY_POCKET_ID
from ..rendering.cache import RenderingCache
from ..environments import Environment
from ..entities import Goal, Player, Key, Door
from ..states import State
from ..environments import Timestep
from ..grid import random_directions, random_colour, RoomsGrid
from .registry import register_env

# A room stand-in for "no key will ever match this" - distinct from
# EMPTY_POCKET_ID (-1, "no key needed") and from any real key id, so a
# door assigned this permanently fails Door's `open` check. Used for
# candidate connector walls that connect_all decides not to open, so
# they act as ordinary walls (still a Door entity, for a fixed
# per-n_rows entity count, but never passable).
_UNOPENABLE = jnp.asarray(-2, dtype=jnp.int32)


def _room_id(row: int, col: int) -> int:
    return row * 3 + col


class KeyCorridor(Environment):
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

        # Doors: mirrors MiniGrid's RoomGrid.connect_all - one mandatory
        # locked door gating the goal room, and every other connector
        # (including the corridor<->key-room doors MiniGrid also treats
        # as optional) resolved by randomly connecting rooms until every
        # non-goal room is reachable from the agent's start, never
        # touching the goal room a second time. Unlike MiniGrid's
        # sample-with-replacement retry loop, this processes each
        # candidate wall exactly once in a random order and adds it iff
        # it still joins two different components (randomised Kruskal) -
        # same reachability guarantee, no unbounded retries, so it stays
        # `jax.jit`-shaped: a fixed number of doors per `n_rows`, decided
        # by which candidates end up connecting something.
        #
        # `parent` (union-find) is kept fully flat (every entry points
        # directly at its true root) as an invariant, not looked up via
        # a worst-case-length pointer chase: a union starting from a
        # flat array can only ever strand nodes that pointed at the
        # losing root, and those are exactly 1 hop further off, so a
        # single `parent[parent]` pass after each union always restores
        # flatness - no need to scan up to `num_rooms` hops per lookup.
        # This matters on GPU specifically: the difference is a handful
        # of sequential, whole-array gathers per candidate instead of
        # ~2*num_rooms of them, and this loop's per-candidate steps are
        # inherently sequential (each depends on the last), which is
        # exactly the shape that doesn't parallelise there.
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

        active = jnp.zeros((num_candidates,), dtype=jnp.bool_)
        for i in range(num_candidates):
            idx = perm[i]
            u, v, elig = u_ids[idx], v_ids[idx], eligible[idx]
            ru, rv = parent[u], parent[v]  # O(1): parent enters every step flat
            connect = elig & (ru != rv)
            parent = parent.at[ru].set(jnp.where(connect, rv, ru))
            parent = parent[parent]  # one pointer-doubling pass restores flatness
            active = active.at[idx].set(connect)

        requires = jnp.where(
            is_goal_slot, key_id, jnp.where(active, EMPTY_POCKET_ID, _UNOPENABLE)
        )
        door_colours = jnp.where(is_goal_slot, key_colour, colours)
        doors = Door.create(
            position=positions,
            requires=requires,
            colour=door_colours,
            open=jnp.zeros((num_candidates,), dtype=jnp.int32),
        )

        entities = {
            "player": player[None],
            "key": key_obj[None],
            "door": doors,
            "goal": goal[None],
        }

        grid = grid.get_grid(occupied_positions=doors.position)
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
