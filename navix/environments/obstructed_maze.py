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
from typing import List, Tuple, Union

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
    random_distinct_positions,
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

# 3x3 layout, in navix's own (row, col) room indices. MiniGrid indexes
# rooms as (col, row) instead, so its own `side_rooms` list
# [(2,1),(1,2),(0,1),(1,0)] transposes into the one below - which is
# self-confirming, since MiniGrid reaches `side_rooms[i]` from the
# centre through `door_idx=i`, and its door_idx order (0=right, 1=down,
# 2=left, 3=up) is exactly navix's own `Directions` order. So index `i`
# here is both "which side room" and "which wall of the centre".
CENTRE_ROOM = (1, 1)
SIDE_ROOMS = ((1, 2), (2, 1), (1, 0), (0, 1))  # E, S, W, N of centre
# MiniGrid's [(2,0),(2,2),(0,2),(0,0)], transposed the same way, and
# sliced to `num_quarters` before the target ball's corner is drawn
CORNER_ROOMS = ((0, 2), (2, 2), (2, 0), (0, 0))
# one step along each `Directions` side, for backing a blocking ball
# off a door into the room it is approached from
SIDE_DELTAS = ((0, 1), (1, 0), (0, -1), (-1, 0))  # E, S, W, N

# MiniGrid gives each category its own fixed colour - `COLOR_NAMES[0]`
# for the target ball, `[1]` for blocking balls, `[2]` for boxes. Its
# COLOR_NAMES is alphabetical (blue, green, grey, ...), so those are
# blue/green/grey; the indices below are navix's own palette
# (red, green, blue, purple, yellow, grey) for the same three hues.
TARGET_COLOUR = 2  # blue
BLOCKER_COLOUR = 1  # green
BOX_COLOUR = 5  # grey


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


class ObstructedMazeFull(Environment):
    """A 3x3 `room_grid`. The centre room connects to `num_quarters`
    side rooms through *unlocked* doors; each of those side rooms
    connects on to two corner rooms through *locked* ones. Every locked
    door's key lives in the side room it is opened from - optionally
    hidden inside a `Box`, optionally with a `Ball` blocking the door.
    The target ball waits in one of the first `num_quarters` corners.

    Verified against MiniGrid's actual `ObstructedMaze_Full`:
    `room_size=6`, keys placed via `place_in_room(*side_room)`, blocking
    balls at `door_pos - DIR_TO_VEC[door_idx]` (i.e. backed off the door
    into the side room), corners sliced `[:num_quarters]` before one is
    drawn for the target.

    !!! note "This is MiniGrid's `-v1` behaviour, under a `-v0` name"
        MiniGrid ships both `-v0` and `-v1` of `2Dlhb`, `1Q`, `2Q` and
        `Full`. Its `-v1` fixes a placement bug in `-v0`: the blocking
        ball could be dropped onto a cell that already held a key,
        covering it and making the episode unsolvable. navix implements
        only the fixed behaviour and registers it under the plain `-v0`
        name - blocking-ball cells are computed first (they follow
        deterministically from the doors) and then excluded from every
        subsequent player/key/box draw, so a key can never be covered.
        `2Dl`/`2Dlh` have no `-v1` upstream and port across unchanged.

    | navix id     | MiniGrid id     | `num_quarters` | `key_in_box` | `blocked` |
    |--------------|-----------------|----------------|--------------|-----------|
    | `2Dl-v0`     | `2Dl-v0`        | 1              | `False`      | `False`   |
    | `2Dlh-v0`    | `2Dlh-v0`       | 1              | `True`       | `False`   |
    | `2Dlhb-v0`   | `2Dlhb-`**v1**  | 1              | `True`       | `True`    |
    | `1Q-v0`      | `1Q-`**v1**     | 1              | `True`       | `True`    |
    | `2Q-v0`      | `2Q-`**v1**     | 2              | `True`       | `True`    |
    | `Full-v0`    | `Full-`**v1**   | 4              | `True`       | `True`    |

    (`2Dlhb` and `1Q` differ only by `agent_room`: `1Q` starts in the
    centre, `2Dlhb` already inside the side room.)"""

    num_quarters: int = struct.field(pytree_node=False, default=4)
    agent_room: Tuple[int, int] = struct.field(pytree_node=False, default=CENTRE_ROOM)
    key_in_box: bool = struct.field(pytree_node=False, default=True)
    blocked: bool = struct.field(pytree_node=False, default=True)

    def _reset(self, key: Array, cache: Union[RenderingCache, None] = None) -> Timestep:
        quarters = self.num_quarters
        (
            key,
            k_doors,
            k_door_colours,
            k_player_pos,
            k_player_dir,
            k_objects,
            k_corner,
            k_target_pos,
        ) = jax.random.split(key, num=8)

        grid = room_grid(ROOM_SIZE, num_rows=3, num_cols=3)

        # --- doors -------------------------------------------------
        # 1 unlocked (centre -> side) + 2 locked (side -> corners) per
        # quarter. Every loop here runs over static (registration-time)
        # values, so this is plain Python control flow.
        door_keys = jax.random.split(k_doors, 3 * quarters)
        door_colours = random_colour(k_door_colours, n=3 * quarters)
        door_positions: List[Array] = []
        door_requires: List[int] = []
        # per side room: its locked doors, as (position, side, key id)
        locked_by_room: List[Tuple[Tuple[int, int], List[Tuple[Array, int, int]]]] = []

        drawn = 0
        next_key_id = 1
        for quarter in range(quarters):
            side_room = SIDE_ROOMS[quarter]

            position = room_grid_door_position(
                door_keys[drawn], ROOM_SIZE, *CENTRE_ROOM, side=quarter
            )
            drawn += 1
            grid = open_wall(grid, position)
            door_positions.append(position)
            door_requires.append(-1)  # unlocked, but still closed

            locked_here = []
            for offset in (-1, 1):
                side = (quarter + offset) % 4
                position = room_grid_door_position(
                    door_keys[drawn], ROOM_SIZE, *side_room, side=side
                )
                drawn += 1
                grid = open_wall(grid, position)
                door_positions.append(position)
                door_requires.append(next_key_id)
                locked_here.append((position, side, next_key_id))
                next_key_id += 1
            locked_by_room.append((side_room, locked_here))

        doors = Door.create(
            position=jnp.stack(door_positions),
            requires=jnp.asarray(door_requires),
            colour=jnp.asarray(door_colours, dtype=jnp.uint8).reshape(3 * quarters),
            open=jnp.zeros(3 * quarters, dtype=jnp.bool_),
        )

        # --- blocking balls ----------------------------------------
        # deterministic given the doors, so these are fixed first and
        # every later draw avoids them (MiniGrid's -v1 fix, see the
        # class docstring)
        blockers: dict = {}
        for side_room, locked_here in locked_by_room:
            blockers[side_room] = [
                position - jnp.asarray(SIDE_DELTAS[side])
                for position, side, _ in locked_here
            ]

        def room_floor(room: Tuple[int, int]) -> Array:
            return jnp.where(room_mask(grid, ROOM_SIZE, *room), grid, -1)

        def keep_clear(room: Tuple[int, int]) -> Array:
            cells = blockers.get(room, []) if self.blocked else []
            if not cells:
                return DISCARD_PILE_COORDS[None]  # off-grid: excludes nothing
            return jnp.stack(cells)

        # --- player ------------------------------------------------
        player_pos = random_positions(
            k_player_pos, room_floor(self.agent_room), exclude=keep_clear(self.agent_room)
        )
        player = Player.create(
            position=player_pos,
            direction=random_directions(k_player_dir),
            pocket=EMPTY_POCKET_ID,
        )

        # --- keys, and the boxes that may hide them ----------------
        object_keys = jax.random.split(k_objects, quarters)
        key_positions: List[Array] = []
        box_positions: List[Array] = []
        key_ids: List[int] = []
        key_colours: List[Array] = []
        for quarter, (side_room, locked_here) in enumerate(locked_by_room):
            exclude = [keep_clear(side_room)]
            if side_room == self.agent_room:
                exclude.append(player_pos[None])
            # two per side room, mutually distinct - one for each of its
            # locked doors
            drawn_positions = random_distinct_positions(
                object_keys[quarter],
                room_floor(side_room),
                n=2,
                exclude=jnp.concatenate(exclude, axis=0),
            )
            for slot, (_, _, key_id) in enumerate(locked_here):
                key_ids.append(key_id)
                # each key wears its own door's colour, as in MiniGrid
                key_colours.append(doors.colour[3 * quarter + 1 + slot])
                if self.key_in_box:
                    box_positions.append(drawn_positions[slot])
                    key_positions.append(DISCARD_PILE_COORDS)
                else:
                    key_positions.append(drawn_positions[slot])

        keys = Key.create(
            position=jnp.stack(key_positions),
            id=jnp.asarray(key_ids),
            colour=jnp.stack(key_colours),
        )

        # --- the target ball, in one of the reachable corners -------
        corner_keys = jax.random.split(k_target_pos, quarters)
        candidates = jnp.stack(
            [
                random_positions(corner_keys[corner], room_floor(CORNER_ROOMS[corner]))
                for corner in range(quarters)
            ]
        )
        target_pos = candidates[jax.random.randint(k_corner, (), 0, quarters)]

        # --- balls: the blockers, then the target last -------------
        # (the target is always the final entry, the same convention
        # the 1D variants use, so `state.mission` and the tests can
        # find it without a separate index)
        ball_positions: List[Array] = []
        ball_ids: List[int] = []
        ball_colours: List[int] = []
        if self.blocked:
            next_ball_id = next_key_id
            for _, locked_here in locked_by_room:
                for position, side, _ in locked_here:
                    ball_positions.append(position - jnp.asarray(SIDE_DELTAS[side]))
                    ball_ids.append(next_ball_id)
                    ball_colours.append(BLOCKER_COLOUR)
                    next_ball_id += 1
        else:
            next_ball_id = next_key_id
        ball_positions.append(target_pos)
        ball_ids.append(next_ball_id)
        ball_colours.append(TARGET_COLOUR)

        balls = Ball.create(
            position=jnp.stack(ball_positions),
            colour=jnp.asarray(ball_colours, dtype=jnp.uint8),
            probability=jnp.ones(len(ball_positions)),
            id=jnp.asarray(ball_ids),
        )

        entities = {
            Entities.PLAYER: player[None],
            Entities.DOOR: doors,
            Entities.KEY: keys,
            Entities.BALL: balls,
        }
        if self.key_in_box:
            entities[Entities.BOX] = Box.create(
                position=jnp.stack(box_positions),
                colour=jnp.full((len(box_positions),), BOX_COLOUR, dtype=jnp.uint8),
                id=jnp.asarray([100 + n for n in range(len(box_positions))]),
                # box n hides key n - `actions.open_box` matches this
                # against `Key.id`
                pocket=jnp.asarray(key_ids),
            )

        mission = Event(
            position=target_pos,
            colour=jnp.asarray(TARGET_COLOUR, dtype=jnp.uint8),
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


_FULL_HEIGHT, _FULL_WIDTH = room_grid_dims(ROOM_SIZE, num_rows=3, num_cols=3)


def _register_full(
    env_id: str,
    num_quarters: int,
    agent_room: Tuple[int, int],
    key_in_box: bool,
    blocked: bool,
    num_rooms_visited: int,
) -> None:
    register_env(
        env_id,
        lambda *args, **kwargs: ObstructedMazeFull.create(
            height=_FULL_HEIGHT,
            width=_FULL_WIDTH,
            num_quarters=num_quarters,
            agent_room=agent_room,
            key_in_box=key_in_box,
            blocked=blocked,
            # MiniGrid's own formula, with its own per-id
            # `num_rooms_visited` (a hand-tuned exploration budget, not
            # a literal room count - `2Dlhb` and `1Q` share a layout but
            # differ here because they start in different rooms)
            max_steps=kwargs.pop("max_steps", 4 * num_rooms_visited * ROOM_SIZE**2),
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


# `2Dl`/`2Dlh` port MiniGrid's `-v0` directly; the rest carry its `-v1`
# behaviour under a `-v0` name (see ObstructedMazeFull's docstring)
_register_full(
    "Navix-ObstructedMaze-2Dl-v0",
    num_quarters=1,
    agent_room=SIDE_ROOMS[0],
    key_in_box=False,
    blocked=False,
    num_rooms_visited=4,
)
_register_full(
    "Navix-ObstructedMaze-2Dlh-v0",
    num_quarters=1,
    agent_room=SIDE_ROOMS[0],
    key_in_box=True,
    blocked=False,
    num_rooms_visited=4,
)
_register_full(
    "Navix-ObstructedMaze-2Dlhb-v0",
    num_quarters=1,
    agent_room=SIDE_ROOMS[0],
    key_in_box=True,
    blocked=True,
    num_rooms_visited=4,
)
_register_full(
    "Navix-ObstructedMaze-1Q-v0",
    num_quarters=1,
    agent_room=CENTRE_ROOM,
    key_in_box=True,
    blocked=True,
    num_rooms_visited=5,
)
_register_full(
    "Navix-ObstructedMaze-2Q-v0",
    num_quarters=2,
    agent_room=SIDE_ROOMS[0],
    key_in_box=True,
    blocked=True,
    num_rooms_visited=11,
)
_register_full(
    "Navix-ObstructedMaze-Full-v0",
    num_quarters=4,
    agent_room=CENTRE_ROOM,
    key_in_box=True,
    blocked=True,
    num_rooms_visited=25,
)
