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

"""`MultiRoom` (issue #182): a chain of `num_rooms` randomly-sized,
randomly-positioned rooms, each connected to the next by an unlocked
`Door`, winding through the grid in a genuinely random 2D path - a
`Goal` sits in the last room.

MiniGrid's own generation algorithm is retry/backtrack-based: it
repeatedly tries a random room placement and rejects it (retrying with
a new random size/position, up to 8 attempts) if it overlaps an
already-placed room or falls outside the grid. That's inherently
data-dependent, unbounded control flow - doesn't trace under JAX the
way every other navix environment's generation does. This file keeps
the same *algorithm* (bounded to a fixed number of retries per room,
`jax.lax.while_loop` instead of Python recursion) rather than
approximating it with an easier-but-different generation scheme -
verified end to end against MiniGrid's actual `_placeRoom`/`_gen_grid`
(the four wall-relative positioning formulas, the entry/exit wall
bookkeeping, the retry count).

Faithful to MiniGrid in one more respect worth flagging plainly:
`Navix-MultiRoom-N6-v0` uses a 25x25 grid (matching MiniGrid's own
fixed default, used for every registration regardless of room count/
size - its retry logic is what keeps a random layout compact enough to
fit that grid, not a smaller canvas). 625 cells is the biggest grid in
navix by a wide margin (`ObstructedMaze`'s `Full` is 16x16 = 256), and
every per-step operation in this codebase scans the whole grid/entity
set - so `N6` is genuinely, permanently more expensive to run than any
other navix environment, not just slower to compile. Kept anyway,
deliberately, for fidelity - see this session's own discussion on
issue #182 for the (rejected) cheaper alternative (a straight room
chain) and the actual cost numbers.

One consequence worth calling out explicitly: `Environment.step`
embeds a full `Environment.reset` call as its `jax.lax.cond` autoreset
branch, so calling `env.step(...)` *un-jitted* in a Python loop
retraces and recompiles that whole branch - reset included - on every
single call, since eager `lax.cond` doesn't cache across separate
top-level calls the way `jax.jit` does. For most navix environments
that's an unnoticeable cost, because their `reset` is cheap. For
`MultiRoom`, and especially `N6`, `reset` is exactly the heavy nested-
retry search described above, so this pattern is a real footgun: it
was measured to balloon to minutes of compile time and multiple GB of
memory before crashing outright. Always `jax.jit` (or `jax.vmap` over
a jitted function) `env.step`/`env.reset` before calling either in a
loop against any `MultiRoom` variant."""

from __future__ import annotations
from typing import List, Tuple, Union

import jax
import jax.numpy as jnp
from jax import Array
from flax import struct

from navix import observations, rewards, terminations

from ..components import EMPTY_POCKET_ID
from ..entities import Door, Entities, Goal, Player
from ..states import State
from ..grid import coordinates, open_wall, random_colour, random_directions, random_positions
from ..rendering.cache import RenderingCache
from .environment import Environment, Timestep
from .registry import register_env


GRID_SIZE = 25  # fixed for every registration, matching MiniGrid's own
MIN_ROOM_SIZE = 4  # matches MiniGrid's hardcoded minSz
BOUNDARY_MARGIN = MIN_ROOM_SIZE - 1  # see place_room's docstring
MAX_PLACEMENT_TRIES = 64  # MiniGrid itself retries only 8 times per room,
# but a JAX retry (one more jax.lax.while_loop iteration, already-compiled)
# has essentially none of the real cost a Python-level retry does - bumped
# well past MiniGrid's own budget since 8 measurably wasn't enough for
# Navix-MultiRoom-N6-v0's tight packing (6 rooms up to 10x10 in a 25x25
# grid).
MAX_LAYOUT_RESTARTS = 32  # a single room's own retry (MAX_PLACEMENT_TRIES,
# BOUNDARY_MARGIN) still can't always salvage a bad *earlier* room's
# choice - MiniGrid's true backtracking can reconsider previous rooms,
# a single room's retry loop structurally can't (see place_room's
# docstring). Confirmed necessary directly: even with the above two,
# Navix-MultiRoom-N6-v0 PRNGKey(2) still exhausted every retry (room
# 3's door only 4 cells from an edge - no legal room 4 size fits).
#
# 32 was chosen empirically, not just "generously rounded up": pushing
# this much higher (96, then 256) measurably made Navix-MultiRoom-N6-v0
# *harder to even compile* on the real (shared, heavily-loaded)
# machines used for this environment's own development-time
# verification - not a hypothetical concern, repeatedly reproduced.
# At 32, N6's own real success rate is ~90% (3/28 sampled seeds still
# exhausted every restart) - good, not perfect, so a guaranteed-valid
# deterministic fallback (see _reset) covers the remainder instead of
# chasing a higher ceiling that's already shown real compile cost.
# generate_layout reports failure honestly instead of silently
# returning an invalid layout (which previously manifested as e.g. the
# goal defaulting to a wall corner, position (0,0)), and _reset
# restarts the *entire* room chain from scratch with a fresh key,
# rather than trying to locally patch just the one room that failed.


def room_top_left_east(key: Array, entry_row: Array, entry_col: Array, size_h: Array, size_w: Array) -> Array:
    # entered via the new room's own EAST wall -> room extends west
    top_col = entry_col - size_w + 1
    # MiniGrid samples the perpendicular offset from a half-open range
    # `[entry - size + 2, entry)`, deliberately EXCLUDING `entry`
    # itself - landing there would put the door on the new room's own
    # north/south wall too (a corner), cutting it off from the room's
    # interior instead of sitting on a proper interior-adjacent cell
    # of the shared wall. `jax.random.randint`'s `high` is exclusive
    # already, so `high=entry_row` (not `entry_row + 1`) reproduces
    # that range exactly - found via a real seed (N4-S5, PRNGKey(0))
    # whose door connected to a room's exact corner instead of its
    # interior, making the two rooms unreachable from each other.
    top_row = jax.random.randint(key, (), entry_row - size_h + 2, entry_row)
    return jnp.asarray([top_row, top_col])


def room_top_left_south(key: Array, entry_row: Array, entry_col: Array, size_h: Array, size_w: Array) -> Array:
    # entered via the new room's own SOUTH wall -> room extends north
    top_row = entry_row - size_h + 1
    # see room_top_left_east's comment: `high` must exclude entry_col
    top_col = jax.random.randint(key, (), entry_col - size_w + 2, entry_col)
    return jnp.asarray([top_row, top_col])


def room_top_left_west(key: Array, entry_row: Array, entry_col: Array, size_h: Array, size_w: Array) -> Array:
    # entered via the new room's own WEST wall -> room extends east
    top_col = entry_col
    # see room_top_left_east's comment: `high` must exclude entry_row
    top_row = jax.random.randint(key, (), entry_row - size_h + 2, entry_row)
    return jnp.asarray([top_row, top_col])


def room_top_left_north(key: Array, entry_row: Array, entry_col: Array, size_h: Array, size_w: Array) -> Array:
    # entered via the new room's own NORTH wall -> room extends south
    top_row = entry_row
    # see room_top_left_east's comment: `high` must exclude entry_col
    top_col = jax.random.randint(key, (), entry_col - size_w + 2, entry_col)
    return jnp.asarray([top_row, top_col])


def room_top_left(wall: Array, key: Array, entry_row: Array, entry_col: Array, size_h: Array, size_w: Array) -> Array:
    """The new room's top-left corner, positioned so its `wall` side
    touches `(entry_row, entry_col)` and it extends away from there -
    verified against MiniGrid's actual `_placeRoom`'s 4 wall-relative
    formulas. `wall` is a per-episode traced value (unlike every other
    navix room-placement helper's static `side`), hence `jax.lax.
    switch` rather than a plain Python `if`."""
    return jax.lax.switch(
        wall,
        [room_top_left_east, room_top_left_south, room_top_left_west, room_top_left_north],
        key, entry_row, entry_col, size_h, size_w,
    )


def exit_door_position_east(key: Array, top: Array, size: Array) -> Array:
    offset = jax.random.randint(key, (), 1, size[0] - 1)
    return jnp.asarray([top[0] + offset, top[1] + size[1] - 1])


def exit_door_position_south(key: Array, top: Array, size: Array) -> Array:
    offset = jax.random.randint(key, (), 1, size[1] - 1)
    return jnp.asarray([top[0] + size[0] - 1, top[1] + offset])


def exit_door_position_west(key: Array, top: Array, size: Array) -> Array:
    offset = jax.random.randint(key, (), 1, size[0] - 1)
    return jnp.asarray([top[0] + offset, top[1]])


def exit_door_position_north(key: Array, top: Array, size: Array) -> Array:
    offset = jax.random.randint(key, (), 1, size[1] - 1)
    return jnp.asarray([top[0], top[1] + offset])


def exit_door_position(wall: Array, key: Array, top: Array, size: Array) -> Array:
    """A random position on room `top`/`size`'s `wall` side, excluding
    the two corners (matching `grid.room_grid_door_position`'s own
    corner-excluding convention - the same reason: a door needs a real
    interior cell on both sides, not the room's own corner)."""
    return jax.lax.switch(
        wall,
        [exit_door_position_east, exit_door_position_south,
         exit_door_position_west, exit_door_position_north],
        key, top, size,
    )


def interiors_overlap(top_a: Array, size_a: Array, top_b: Array, size_b: Array) -> Array:
    """Whether two rooms' *interiors* (each room's own 1-cell wall
    border excluded) overlap. Interiors, not full bounds: two rooms
    connected by a door are deliberately placed edge-to-edge, sharing
    exactly the 1-cell wall between them - a full-bounds overlap test
    would flag that legitimate adjacency as a collision. Interiors
    never legitimately touch, so this only rejects real collisions."""
    a_row_lo, a_row_hi = top_a[0] + 1, top_a[0] + size_a[0] - 2
    a_col_lo, a_col_hi = top_a[1] + 1, top_a[1] + size_a[1] - 2
    b_row_lo, b_row_hi = top_b[0] + 1, top_b[0] + size_b[0] - 2
    b_col_lo, b_col_hi = top_b[1] + 1, top_b[1] + size_b[1] - 2
    row_overlap = (a_row_lo <= b_row_hi) & (b_row_lo <= a_row_hi)
    col_overlap = (a_col_lo <= b_col_hi) & (b_col_lo <= a_col_hi)
    return row_overlap & col_overlap


def place_room(
    key: Array,
    entry_wall: Array,
    entry_pos: Array,
    size_min: int,
    size_max: int,
    existing_tops: List[Array],
    existing_sizes: List[Array],
) -> Tuple[Array, Array, Array]:
    """Bounded-retry room placement - `jax.lax.while_loop`'s JAX-
    traceable equivalent of MiniGrid's own recursive retry (up to
    `MAX_PLACEMENT_TRIES` random `(size, position)` draws, accepting
    the first that's in-bounds *with a margin* (see below) and doesn't
    overlap any already-placed room). `existing_tops`/`existing_sizes`
    are plain Python lists, not padded arrays - safe here because the
    outer room-by-room loop in `_reset`/`generate_layout` is itself a
    static Python `for` (room count is a registration-time constant),
    so each call site's list length is static, same as
    `ObstructedMazeFull`'s own `door_positions` accumulation.

    `BOUNDARY_MARGIN` keeps every room a few cells clear of the grid
    edges (not just the two relevant to its own entry wall), lowering
    the odds a room lands flush against a boundary - which would make
    the *next* room's placement out-of-bounds by construction,
    unfixable by any amount of retrying here (confirmed directly:
    `Navix-MultiRoom-N6-v0` `PRNGKey(1)`, before this margin existed -
    room 3 landed at column 0, and room 4's own entry-wall formula,
    extend further left from a door already at the boundary, can only
    ever produce a negative column). The margin lowers the odds, but
    doesn't eliminate them - a genuinely tight, unluckily-drawn layout
    can still exhaust every retry (confirmed too: `PRNGKey(2)` still
    failed with the margin in place, room 3's door only 4 cells from
    an edge, too tight for *any* legal room 4 size to fit). That's
    what `found` (the third return value) is for: MiniGrid's true
    retry backtracks across multiple rooms when a placement can't be
    salvaged locally, which a single room's own retry loop structurally
    can't replicate - so on total failure here, this function reports
    it honestly instead of silently returning an invalid placement,
    and `generate_layout`'s own outer retry restarts the *entire*
    chain with a fresh key instead - the closest JAX-traceable
    equivalent of "back up and reconsider an earlier choice", since
    which earlier room to blame isn't something this function can
    know from inside a single room's own placement attempt."""
    entry_row, entry_col = entry_pos[0], entry_pos[1]

    def cond_fn(carry):
        _, attempt, found, _, _ = carry
        return jnp.logical_and(attempt < MAX_PLACEMENT_TRIES, jnp.logical_not(found))

    def body_fn(carry):
        key, attempt, _, _, _ = carry
        key, k_h, k_w, k_pos = jax.random.split(key, 4)
        size_h = jax.random.randint(k_h, (), size_min, size_max + 1)
        size_w = jax.random.randint(k_w, (), size_min, size_max + 1)
        top = room_top_left(entry_wall, k_pos, entry_row, entry_col, size_h, size_w)
        size = jnp.asarray([size_h, size_w])
        in_bounds = (
            (top[0] >= BOUNDARY_MARGIN)
            & (top[1] >= BOUNDARY_MARGIN)
            & (top[0] + size[0] <= GRID_SIZE - BOUNDARY_MARGIN)
            & (top[1] + size[1] <= GRID_SIZE - BOUNDARY_MARGIN)
        )
        no_overlap = jnp.asarray(True)
        for other_top, other_size in zip(existing_tops, existing_sizes):
            no_overlap = no_overlap & jnp.logical_not(
                interiors_overlap(top, size, other_top, other_size)
            )
        valid = in_bounds & no_overlap
        return key, attempt + 1, valid, top, size

    init = (
        key,
        jnp.asarray(0),
        jnp.asarray(False),
        jnp.zeros(2, dtype=jnp.int32),
        jnp.zeros(2, dtype=jnp.int32),
    )
    _, _, found, top, size = jax.lax.while_loop(cond_fn, body_fn, init)
    return top, size, found


def fallback_layout(key: Array, n: int) -> Tuple[Array, Array, Array, Array]:
    """A deterministic straight chain of `n` minimum-size rooms,
    centred as a whole and extending east - always valid by
    construction (`GRID_SIZE=25` comfortably fits the *chain's total
    span*, `MIN_ROOM_SIZE + (n-1) * (MIN_ROOM_SIZE-1)` = 19 for the
    largest real `n=6`, for every real registration), unlike the
    random search `_reset`'s own `jax.lax.while_loop` runs, which only
    *usually* finds a valid layout (see `MAX_LAYOUT_RESTARTS`).
    `_reset` uses this as the guaranteed-safe fallback for the rare
    episode where that random search doesn't resolve within its
    budget, rather than ever using a broken layout. Room *positions*
    are fixed (not random) here, but door colours still are, so it
    isn't the exact same episode every time this path is taken -
    assumes `n >= 2` (true of every real registration; a hypothetical
    `n=1` custom instantiation would need its own, simpler no-doors
    handling, not provided here).

    Centring the *chain's total span*, not just room 0's own position,
    matters: consecutive rooms share one wall cell each, so the chain
    only grows by `MIN_ROOM_SIZE - 1` per additional room, not a full
    `MIN_ROOM_SIZE` - anchoring room 0 alone at the grid centre and
    extending purely eastward from there runs the *last* room off the
    grid's edge for `n=6` (confirmed directly: room 0 at column 10,
    chain reaching column 25 - already outside the valid 0-24 range,
    before that room's own width is even added)."""
    size = jnp.asarray([MIN_ROOM_SIZE, MIN_ROOM_SIZE])
    row0 = GRID_SIZE // 2 - MIN_ROOM_SIZE // 2
    chain_width = MIN_ROOM_SIZE + (n - 1) * (MIN_ROOM_SIZE - 1)
    col0 = (GRID_SIZE - chain_width) // 2

    tops: List[Array] = [jnp.asarray([row0, col0])]
    for i in range(1, n):
        tops.append(tops[-1] + jnp.asarray([0, MIN_ROOM_SIZE - 1]))

    door_row = row0 + 1  # a valid interior row for every room (MIN_ROOM_SIZE=4)
    door_positions = [
        jnp.asarray([door_row, tops[i][1] + MIN_ROOM_SIZE - 1]) for i in range(n - 1)
    ]
    door_colours = [random_colour(k) for k in jax.random.split(key, n - 1)]

    return (
        jnp.stack(tops),
        jnp.stack([size] * n),
        jnp.stack(door_positions),
        jnp.stack(door_colours),
    )


class MultiRoom(Environment):
    """See this module's own docstring for the generation algorithm
    and the `N6` grid-size/cost note. `num_rooms`/`max_room_size` are
    static (`pytree_node=False`) - each registration gets its own
    traced generation graph, same convention as every other navix
    environment's structural parameters."""

    num_rooms: int = struct.field(pytree_node=False, default=2)
    max_room_size: int = struct.field(pytree_node=False, default=4)

    def generate_layout(
        self, key: Array
    ) -> Tuple[Array, Array, Array, Array, Array]:
        """One attempt at placing all `num_rooms` rooms - a pure
        function of `key`, returning fixed-shape stacked arrays (not
        the growing Python lists `place_room`'s own docstring
        describes - those are fine *within* one room's placement,
        where the call site's list length is static per room index,
        but this function's *own* output must be a single fixed-shape
        value so `_reset` can retry it whole via `jax.lax.while_loop`).
        The last return value is `all_valid`: whether every room
        actually found a legal placement (see `place_room`'s
        docstring on why a single room's own retry can't always
        salvage a bad earlier choice) - `_reset` restarts this whole
        function with a fresh key when it's `False`, rather than ever
        using a layout that silently contains an invalid room."""
        n = self.num_rooms
        size_max = self.max_room_size

        key, k_size_h, k_size_w, k_wall = jax.random.split(key, 4)
        size0_h = jax.random.randint(k_size_h, (), MIN_ROOM_SIZE, size_max + 1)
        size0_w = jax.random.randint(k_size_w, (), MIN_ROOM_SIZE, size_max + 1)
        top0 = jnp.asarray([GRID_SIZE // 2 - size0_h // 2, GRID_SIZE // 2 - size0_w // 2])
        size0 = jnp.asarray([size0_h, size0_w])
        exit_wall0 = jax.random.randint(k_wall, (), 0, 4)

        tops: List[Array] = [top0]
        sizes: List[Array] = [size0]
        exit_walls: List[Array] = [exit_wall0]
        door_positions: List[Array] = []
        door_colours: List[Array] = []
        all_valid = jnp.asarray(True)

        for _ in range(1, n):
            key, k_door, k_place, k_exit_offset, k_colour = jax.random.split(key, 5)
            prev_top, prev_size, prev_exit_wall = tops[-1], sizes[-1], exit_walls[-1]

            door_pos = exit_door_position(prev_exit_wall, k_door, prev_top, prev_size)
            door_positions.append(door_pos)

            if door_colours:
                # a random colour distinct from the immediately
                # previous door's - matches MiniGrid's own "exclude
                # the previous door's colour" rule. Adding a random
                # nonzero offset mod 6 always lands on one of the
                # other 5, uniformly - same trick `place_room`'s
                # exit-wall selection uses for "exclude one value".
                offset = jax.random.randint(k_colour, (), 1, 6)
                colour = (door_colours[-1] + offset) % 6
            else:
                colour = random_colour(k_colour)
            door_colours.append(colour)

            entry_wall = (prev_exit_wall + 2) % 4
            top, size, found = place_room(
                k_place, entry_wall, door_pos, MIN_ROOM_SIZE, size_max, tops, sizes
            )
            tops.append(top)
            sizes.append(size)
            all_valid = all_valid & found

            exit_offset = jax.random.randint(k_exit_offset, (), 1, 4)
            exit_walls.append((entry_wall + exit_offset) % 4)

        return (
            jnp.stack(tops),
            jnp.stack(sizes),
            jnp.stack(door_positions),
            jnp.stack(door_colours),
            all_valid,
        )

    def _reset(self, key: Array, cache: Union[RenderingCache, None] = None) -> Timestep:
        n = self.num_rooms
        key, k_layout, k_player_pos, k_player_dir, k_goal = jax.random.split(key, 5)

        def cond_fn(carry):
            _, attempt, valid, *_ = carry
            return jnp.logical_and(attempt < MAX_LAYOUT_RESTARTS, jnp.logical_not(valid))

        def body_fn(carry):
            key, attempt, _, *_ = carry
            tops, sizes, door_positions, door_colours, valid = self.generate_layout(
                jax.random.fold_in(key, attempt)
            )
            return key, attempt + 1, valid, tops, sizes, door_positions, door_colours

        init_tops, init_sizes, init_doors, init_colours, init_valid = self.generate_layout(
            jax.random.fold_in(k_layout, 0)
        )
        _, _, valid, tops, sizes, door_positions, door_colours = jax.lax.while_loop(
            cond_fn,
            body_fn,
            (k_layout, jnp.asarray(1), init_valid, init_tops, init_sizes, init_doors, init_colours),
        )

        # the rare (~10% at this ceiling, see MAX_LAYOUT_RESTARTS) case
        # where random search never found a valid layout - fall back to
        # a deterministic straight chain, guaranteed valid by
        # construction, rather than ever using a broken one
        fallback_tops, fallback_sizes, fallback_doors, fallback_colours = fallback_layout(
            jax.random.fold_in(k_layout, MAX_LAYOUT_RESTARTS), n
        )
        tops = jnp.where(valid, tops, fallback_tops)
        sizes = jnp.where(valid, sizes, fallback_sizes)
        door_positions = jnp.where(valid, door_positions, fallback_doors)
        door_colours = jnp.where(valid, door_colours, fallback_colours)

        # carve every room's interior to floor (the 1-cell borders
        # in between stay wall, punched through only at door cells)
        rows, cols = coordinates(jnp.zeros((GRID_SIZE, GRID_SIZE), dtype=jnp.int32))
        is_floor = jnp.zeros((GRID_SIZE, GRID_SIZE), dtype=jnp.bool_)
        room_floor_masks: List[Array] = []
        for top, size in zip(tops, sizes):
            room_floor = (
                (rows >= top[0] + 1)
                & (rows <= top[0] + size[0] - 2)
                & (cols >= top[1] + 1)
                & (cols <= top[1] + size[1] - 2)
            )
            room_floor_masks.append(room_floor)
            is_floor = is_floor | room_floor
        grid = jnp.where(is_floor, 0, -1).astype(jnp.int32)
        for door_pos in door_positions:
            grid = open_wall(grid, door_pos)

        doors = Door.create(
            position=door_positions,
            requires=jnp.full((n - 1,), -1),
            colour=door_colours.astype(jnp.uint8),
            open=jnp.zeros(n - 1, dtype=jnp.bool_),
        )

        first_room_floor = jnp.where(room_floor_masks[0], 0, -1)
        player_pos = random_positions(k_player_pos, first_room_floor)
        player = Player.create(
            position=player_pos,
            direction=random_directions(k_player_dir),
            pocket=EMPTY_POCKET_ID,
        )

        last_room_floor = jnp.where(room_floor_masks[-1], 0, -1)
        goal_pos = random_positions(k_goal, last_room_floor)
        goal = Goal.create(position=goal_pos, probability=jnp.asarray(1.0))

        entities = {
            Entities.PLAYER: player[None],
            Entities.DOOR: doors,
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
            action=jnp.asarray(0, dtype=jnp.int32),
            reward=jnp.asarray(0.0, dtype=jnp.float32),
            step_type=jnp.asarray(0, dtype=jnp.int32),
            state=state,
        )


def register_multi_room(env_id: str, num_rooms: int, max_room_size: int) -> None:
    register_env(
        env_id,
        lambda *args, **kwargs: MultiRoom.create(
            height=GRID_SIZE,
            width=GRID_SIZE,
            num_rooms=num_rooms,
            max_room_size=max_room_size,
            max_steps=kwargs.pop("max_steps", num_rooms * 20),
            observation_fn=kwargs.pop("observation_fn", observations.symbolic),
            reward_fn=kwargs.pop(
                "reward_fn",
                rewards.compose(rewards.on_goal_reached, rewards.time_cost),
            ),
            termination_fn=kwargs.pop("termination_fn", terminations.DEFAULT_TERMINATION),
            *args,
            **kwargs,
        ),
    )


register_multi_room("Navix-MultiRoom-N2-S4-v0", num_rooms=2, max_room_size=4)
# MiniGrid's own MultiRoom-N4-S5-v0 is a documented legacy bug (kept
# "for backwards compatibility") - it actually uses 6 rooms, not 4;
# MiniGrid's own -v1 fixes this to the 4 its name promises. Per this
# session's v1-becomes-v0 convention, navix implements the fix
# (4 rooms) under the plain -v0 navix id.
register_multi_room("Navix-MultiRoom-N4-S5-v0", num_rooms=4, max_room_size=5)
register_multi_room("Navix-MultiRoom-N6-v0", num_rooms=6, max_room_size=10)
