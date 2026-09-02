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
from typing import Union

import jax
import jax.numpy as jnp
from jax import Array
from flax import struct

from navix import observations, rewards, terminations

from ..components import EMPTY_POCKET_ID
from ..entities import Entities, Goal, Player, Ball
from ..states import State
from ..grid import random_positions, random_directions, room
from ..rendering.cache import RenderingCache
from ..rendering.registry import PALETTE
from .environment import Environment, Timestep
from .registry import register_env


class DynamicObstacles(Environment):
    """A room with `n_obstacles` `Ball`s that each take one random step
    every turn; reach the goal without touching one.

    Deliberately differs from real MiniGrid's `DynamicObstaclesEnv` in
    one place: real MiniGrid removes `pickup`/`drop`/`toggle`/`done`
    from this environment's action space entirely (`Discrete(3)`
    instead of the usual `Discrete(7)`), since none of them mean
    anything here. Navix keeps the full default action set instead, so
    that an agent trained across the whole navix suite sees one
    uniform action interface everywhere - and since `Ball` is
    `Pickable` (needed elsewhere, for `Fetch`/`PutNear`/`GoToObject`/
    `BlockedUnlockPickup`), picking one up here ends the episode the
    same way walking into it already does (`termination_fn` composes
    `on_ball_pickup` alongside `on_ball_hit`), rather than either
    silently removing it from play or being unavailable.

    To instantiate the exact MiniGrid-equivalent `Discrete(3)` action
    space instead, pass `action_set` explicitly:

    ```python
    env = navix.make(
        "Navix-Dynamic-Obstacles-5x5-v0",
        action_set=(navix.actions.rotate_ccw, navix.actions.rotate_cw, navix.actions.forward),
    )
    ```

    (verified directly: `env.action_set`/`env.action_space` both come
    out as length/`Discrete(3)`, matching real MiniGrid, and
    `env.step()` works normally with action indices `0`/`1`/`2`)."""

    random_start: bool = struct.field(pytree_node=False, default=False)
    n_obstacles: int = struct.field(pytree_node=False, default=2)

    def _reset(self, key: Array, cache: Union[RenderingCache, None] = None) -> Timestep:
        key, k1, k2, k3 = jax.random.split(key, 4)

        # map
        grid = room(height=self.height, width=self.width)

        # goal and player
        if self.random_start:
            player_pos = random_positions(k1, grid)
            direction = random_directions(k2, n=1)
        else:
            player_pos = jnp.asarray([1, 1])
            direction = jnp.asarray(0)
        # player
        player = Player.create(
            position=player_pos,
            direction=direction,
            pocket=EMPTY_POCKET_ID,
        )
        # goal
        goal_pos = jnp.asarray([self.height - 2, self.width - 2])
        goal = Goal.create(position=goal_pos, probability=jnp.asarray(1.0))

        # balls
        exclude = jnp.stack([player_pos, goal_pos])
        ball_pos = random_positions(k3, grid, n=self.n_obstacles, exclude=exclude)
        balls = Ball.create(
            position=ball_pos,
            colour=jnp.tile(PALETTE.BLUE, (self.n_obstacles,)),
            probability=jnp.ones(self.n_obstacles),
            # Ball became Pickable so Fetch/PutNear/BlockedUnlockPickup can
            # use it - ids just need to be unique per instance here, they
            # play no role in this env (DynamicObstacles relies on
            # walk-into collision via on_ball_hit, not pickup, for its
            # termination; see PR description for the resulting narrow
            # behavior change: pickup() no longer no-ops on these balls).
            id=jnp.arange(1, self.n_obstacles + 1, dtype=jnp.int32),
        )

        entities = {
            Entities.PLAYER: player[None],
            Entities.GOAL: goal[None],
            Entities.BALL: balls,
        }

        # systems
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


register_env(
    "Navix-Dynamic-Obstacles-5x5-v0",
    lambda *args, **kwargs: DynamicObstacles.create(
        height=5,
        width=5,
        n_obstacles=2,
        random_start=False,
        observation_fn=kwargs.pop("observation_fn", observations.symbolic),
        reward_fn=kwargs.pop("reward_fn", rewards.on_goal_reached),
        # picking up a ball now ends the episode the same way walking
        # into one already does (rather than silently removing it from
        # play), now that Ball is Pickable - see PR #191's review.
        termination_fn=kwargs.pop(
            "termination_fn",
            terminations.compose(
                terminations.on_goal_reached,
                terminations.on_lava_fall,
                terminations.on_ball_hit,
                terminations.on_ball_pickup,
            ),
        ),
        *args,
        **kwargs,
    ),
)
register_env(
    "Navix-Dynamic-Obstacles-5x5-Random-v0",
    lambda *args, **kwargs: DynamicObstacles.create(
        height=5,
        width=5,
        n_obstacles=2,
        random_start=True,
        observation_fn=kwargs.pop("observation_fn", observations.symbolic),
        reward_fn=kwargs.pop("reward_fn", rewards.on_goal_reached),
        # picking up a ball now ends the episode the same way walking
        # into one already does (rather than silently removing it from
        # play), now that Ball is Pickable - see PR #191's review.
        termination_fn=kwargs.pop(
            "termination_fn",
            terminations.compose(
                terminations.on_goal_reached,
                terminations.on_lava_fall,
                terminations.on_ball_hit,
                terminations.on_ball_pickup,
            ),
        ),
        *args,
        **kwargs,
    ),
)
register_env(
    "Navix-Dynamic-Obstacles-6x6-v0",
    lambda *args, **kwargs: DynamicObstacles.create(
        height=6,
        width=6,
        n_obstacles=3,
        random_start=False,
        observation_fn=kwargs.pop("observation_fn", observations.symbolic),
        reward_fn=kwargs.pop("reward_fn", rewards.on_goal_reached),
        # picking up a ball now ends the episode the same way walking
        # into one already does (rather than silently removing it from
        # play), now that Ball is Pickable - see PR #191's review.
        termination_fn=kwargs.pop(
            "termination_fn",
            terminations.compose(
                terminations.on_goal_reached,
                terminations.on_lava_fall,
                terminations.on_ball_hit,
                terminations.on_ball_pickup,
            ),
        ),
        *args,
        **kwargs,
    ),
)
register_env(
    "Navix-Dynamic-Obstacles-6x6-Random-v0",
    lambda *args, **kwargs: DynamicObstacles.create(
        height=6,
        width=6,
        n_obstacles=3,
        random_start=True,
        observation_fn=kwargs.pop("observation_fn", observations.symbolic),
        reward_fn=kwargs.pop("reward_fn", rewards.on_goal_reached),
        # picking up a ball now ends the episode the same way walking
        # into one already does (rather than silently removing it from
        # play), now that Ball is Pickable - see PR #191's review.
        termination_fn=kwargs.pop(
            "termination_fn",
            terminations.compose(
                terminations.on_goal_reached,
                terminations.on_lava_fall,
                terminations.on_ball_hit,
                terminations.on_ball_pickup,
            ),
        ),
        *args,
        **kwargs,
    ),
)
register_env(
    "Navix-Dynamic-Obstacles-8x8-v0",
    lambda *args, **kwargs: DynamicObstacles.create(
        height=8,
        width=8,
        n_obstacles=4,
        random_start=False,
        observation_fn=kwargs.pop("observation_fn", observations.symbolic),
        reward_fn=kwargs.pop("reward_fn", rewards.on_goal_reached),
        # picking up a ball now ends the episode the same way walking
        # into one already does (rather than silently removing it from
        # play), now that Ball is Pickable - see PR #191's review.
        termination_fn=kwargs.pop(
            "termination_fn",
            terminations.compose(
                terminations.on_goal_reached,
                terminations.on_lava_fall,
                terminations.on_ball_hit,
                terminations.on_ball_pickup,
            ),
        ),
        *args,
        **kwargs,
    ),
)
register_env(
    "Navix-Dynamic-Obstacles-16x16-v0",
    lambda *args, **kwargs: DynamicObstacles.create(
        height=16,
        width=16,
        n_obstacles=8,
        random_start=False,
        observation_fn=kwargs.pop("observation_fn", observations.symbolic),
        reward_fn=kwargs.pop("reward_fn", rewards.on_goal_reached),
        # picking up a ball now ends the episode the same way walking
        # into one already does (rather than silently removing it from
        # play), now that Ball is Pickable - see PR #191's review.
        termination_fn=kwargs.pop(
            "termination_fn",
            terminations.compose(
                terminations.on_goal_reached,
                terminations.on_lava_fall,
                terminations.on_ball_hit,
                terminations.on_ball_pickup,
            ),
        ),
        *args,
        **kwargs,
    ),
)
