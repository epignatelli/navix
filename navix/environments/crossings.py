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


from typing import Union
import jax
import jax.numpy as jnp
from jax import Array
from flax import struct

from navix import observations, rewards, terminations

from ..components import EMPTY_POCKET_ID
from ..rendering.cache import RenderingCache
from . import Environment
from ..entities import Player, Goal
from ..states import State
from . import Timestep
from .registry import register_env


def _gaps_from_monotone_path(
    key: Array,
    Hi: int,
    Wi: int,
    odd_rows: Array,
    odd_cols: Array,
    sel_mask_v: Array,
    sel_mask_h: Array,
) -> Array:
    """Return a boolean (Hi, Wi) mask with exactly one gap per selected river,
    placed along a guaranteed-solvable (monotone) path from (0, 0) to
    (Hi - 1, Wi - 1).

    Mirrors MiniGrid's `CrossingEnv._gen_grid` algorithm
    (https://github.com/Farama-Foundation/Minigrid/blob/master/minigrid/envs/crossing.py):

    1. Build a list of moves: ``len(rivers_v)`` horizontal moves +
       ``len(rivers_h)`` vertical moves.
    2. Shuffle the move list.
    3. For each move, place a single gap in the next river to be crossed,
       at a random position within the current "room" bounded by previously
       crossed rivers.

    Implemented with ``jax.lax.scan`` over a fixed-size index sequence so
    the function is JIT-compatible.
    """
    Nv, Nh = odd_cols.size, odd_rows.size

    # Sort selected rivers to the front (ascending), unselected to the end.
    BIG = jnp.int32(10_000)
    v_vals = jnp.where(sel_mask_v, odd_cols, BIG)
    h_vals = jnp.where(sel_mask_h, odd_rows, BIG)
    v_sorted = odd_cols[jnp.argsort(v_vals)]
    h_sorted = odd_rows[jnp.argsort(h_vals)]

    nv = sel_mask_v.sum()
    nh = sel_mask_h.sum()
    n = nv + nh

    # Random interleaving: assign uniform priorities, push unselected past 1.0
    # so they sort to the end of the move sequence.
    key, kv, kh, kscan = jax.random.split(key, 4)
    pv = jax.random.uniform(kv, (Nv,)) + (~sel_mask_v) * 2.0
    ph = jax.random.uniform(kh, (Nh,)) + (~sel_mask_h) * 2.0
    p = jnp.concatenate([pv, ph])
    order = jnp.argsort(p)
    use = jnp.arange(Nv + Nh) < n  # only the first n scan steps are real

    keys = jax.random.split(kscan, Nv + Nh)

    def step(carry, i):
        gv, gh, gaps = carry
        idx = order[i]
        take_i = use[i]
        is_v = idx < Nv
        k = keys[i]

        # Bounds of the current room along each axis (rivers already crossed).
        prev_h = jnp.where(
            gh > 0,
            jax.lax.dynamic_index_in_dim(h_sorted, gh - 1, keepdims=False),
            jnp.int32(-1),
        )
        next_h = jnp.where(
            gh < nh,
            jax.lax.dynamic_index_in_dim(h_sorted, gh, keepdims=False),
            jnp.int32(Hi),
        )
        prev_v = jnp.where(
            gv > 0,
            jax.lax.dynamic_index_in_dim(v_sorted, gv - 1, keepdims=False),
            jnp.int32(-1),
        )
        next_v = jnp.where(
            gv < nv,
            jax.lax.dynamic_index_in_dim(v_sorted, gv, keepdims=False),
            jnp.int32(Wi),
        )

        # The river we are about to cross (column for vertical move,
        # row for horizontal move).
        col = jax.lax.dynamic_index_in_dim(v_sorted, gv, keepdims=False)
        row = jax.lax.dynamic_index_in_dim(h_sorted, gh, keepdims=False)

        # Pick a gap coordinate inside the current room.
        y = jax.random.randint(k, (), prev_h + 1, next_h)  # along vertical river
        x = jax.random.randint(k, (), prev_v + 1, next_v)  # along horizontal river

        gap_r = jnp.where(is_v, y, row)
        gap_c = jnp.where(is_v, col, x)

        # Place the gap. Build a one-hot mask separately so the .at[].set()
        # is unconditional and not nested inside a where (avoids tracer
        # issues with conditional updates).
        add = jnp.zeros((Hi, Wi), dtype=bool).at[gap_r, gap_c].set(True)
        gaps = jax.lax.select(take_i, gaps | add, gaps)

        gv = gv + (take_i & is_v)
        gh = gh + (take_i & ~is_v)
        return (gv, gh, gaps), None

    init = (jnp.int32(0), jnp.int32(0), jnp.zeros((Hi, Wi), dtype=bool))
    (_, _, gaps), _ = jax.lax.scan(step, init, jnp.arange(Nv + Nh))
    return gaps


def _crossings_grid(key: Array, H: int, W: int, n: int) -> Array:
    """Build the interior grid for SimpleCrossing.

    Places ``n`` complete river lines (horizontal or vertical, randomly
    selected from the ``odd_rows`` / ``odd_cols`` candidates) in a
    ``(H - 2, W - 2)`` interior grid, then carves a single gap in each
    selected river along a monotone start-to-goal path.

    Returns a ``jnp.int32`` interior grid where ``-1`` is wall and ``0`` is
    floor; the caller pads it with a wall border.
    """
    Hi, Wi = H - 2, W - 2
    rows = jnp.arange(Hi)
    cols = jnp.arange(Wi)
    odd_rows = jnp.arange(1, Hi, 2)
    odd_cols = jnp.arange(1, Wi, 2)
    Nv, Nh = odd_cols.size, odd_rows.size

    key, k_sel, k_gap = jax.random.split(key, 3)
    all_ids = jnp.arange(Nv + Nh)
    n = jnp.minimum(n, all_ids.size)

    # Select n rivers via permutation (avoids dynamic slicing in JIT).
    perm = jax.random.permutation(k_sel, all_ids)
    rank = jnp.zeros(Nv + Nh, dtype=jnp.int32).at[perm].set(jnp.arange(Nv + Nh))
    sel_mask = rank < n  # boolean over [v_rivers..., h_rivers...]

    # Build the dense river mask without side-effectful writes.
    vcol_onehot = cols[None, :] == odd_cols[:, None]  # (Nv, Wi)
    vmask_cols = jnp.any(vcol_onehot & sel_mask[:Nv, None], axis=0)  # (Wi,)
    vmask = jnp.broadcast_to(vmask_cols, (Hi, Wi))

    rrow_onehot = rows[None, :] == odd_rows[:, None]  # (Nh, Hi)
    hmask_rows = jnp.any(rrow_onehot & sel_mask[Nv:, None], axis=0)  # (Hi,)
    hmask = jnp.broadcast_to(hmask_rows[:, None], (Hi, Wi))

    rivers = vmask | hmask

    gaps = _gaps_from_monotone_path(
        k_gap, Hi, Wi, odd_rows, odd_cols, sel_mask[:Nv], sel_mask[Nv:]
    )

    obstacles = rivers & ~gaps
    return jnp.where(obstacles, -1, 0).astype(jnp.int32)


class Crossings(Environment):
    n_crossings: int = struct.field(pytree_node=False, default=1)

    def _reset(self, key: Array, cache: Union[RenderingCache, None] = None) -> Timestep:
        assert (
            self.height == self.width
        ), f"Crossings are only defined for square grids, got height {self.height} and \
            width {self.width}"
        assert (
            self.height % 2 == 1 and self.width % 2 == 1
        ), "Crossings grid dimensions must be odd"

        key, k1 = jax.random.split(key)

        player_pos = jnp.asarray([1, 1])
        player_dir = jnp.asarray(0)
        player = Player.create(
            position=player_pos, direction=player_dir, pocket=EMPTY_POCKET_ID
        )
        goal_pos = jnp.asarray([self.height - 2, self.width - 2])
        goals = Goal.create(position=goal_pos, probability=jnp.asarray(1.0))

        entities = {
            "player": player[None],
            "goal": goals[None],
        }

        grid_interior = _crossings_grid(k1, self.height, self.width, self.n_crossings)
        grid = jnp.pad(grid_interior, 1, mode="constant", constant_values=-1)

        state = State(
            key=key,
            grid=grid,
            cache=RenderingCache.init(grid),
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
    "Navix-SimpleCrossingS9N1-v0",
    lambda *args, **kwargs: Crossings.create(
        height=9,
        width=9,
        n_crossings=1,
        observation_fn=kwargs.pop("observation_fn", observations.symbolic),
        reward_fn=kwargs.pop("reward_fn", rewards.on_goal_reached),
        termination_fn=kwargs.pop("termination_fn", terminations.on_goal_reached),
        *args,
        **kwargs,
    ),
)
register_env(
    "Navix-SimpleCrossingS9N2-v0",
    lambda *args, **kwargs: Crossings.create(
        height=9,
        width=9,
        n_crossings=2,
        observation_fn=kwargs.pop("observation_fn", observations.symbolic),
        reward_fn=kwargs.pop("reward_fn", rewards.on_goal_reached),
        termination_fn=kwargs.pop("termination_fn", terminations.on_goal_reached),
        *args,
        **kwargs,
    ),
)
register_env(
    "Navix-SimpleCrossingS9N3-v0",
    lambda *args, **kwargs: Crossings.create(
        height=9,
        width=9,
        n_crossings=3,
        observation_fn=kwargs.pop("observation_fn", observations.symbolic),
        reward_fn=kwargs.pop("reward_fn", rewards.on_goal_reached),
        termination_fn=kwargs.pop("termination_fn", terminations.on_goal_reached),
        *args,
        **kwargs,
    ),
)
register_env(
    "Navix-SimpleCrossingS11N5-v0",
    lambda *args, **kwargs: Crossings.create(
        height=11,
        width=11,
        n_crossings=5,
        observation_fn=kwargs.pop("observation_fn", observations.symbolic),
        reward_fn=kwargs.pop("reward_fn", rewards.on_goal_reached),
        termination_fn=kwargs.pop("termination_fn", terminations.on_goal_reached),
        *args,
        **kwargs,
    ),
)
