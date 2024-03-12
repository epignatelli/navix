from typing import Union
import jax
import jax.numpy as jnp
from jax import Array
from flax import struct

from ..components import EMPTY_POCKET_ID
from ..rendering.cache import RenderingCache
from . import Environment
from ..entities import State, Player, Goal, Lava
from . import Timestep
from ..grid import room
from .registry import register_env


class Crossings(Environment):
    n_crossings: int = struct.field(pytree_node=False, default=1)
    is_lava: bool = struct.field(pytree_node=False, default=False)

    def reset(self, key: Array, cache: Union[RenderingCache, None] = None) -> Timestep:
        assert (
            self.height == self.width
        ), f"Crossings are only defined for square grids, got height {self.height} and \
            width {self.width}"
        # check minimum height and width
        key, k1, k2 = jax.random.split(key, num=3)

        grid = jnp.zeros((self.height - 2, self.width - 2), dtype=jnp.int32)

        # player
        player_pos = jnp.asarray([1, 1])
        player_dir = jnp.asarray(0)
        player = Player(
            position=player_pos, direction=player_dir, pocket=EMPTY_POCKET_ID
        )
        # goal
        goal_pos = jnp.asarray([self.height - 2, self.width - 2])
        goals = Goal(position=goal_pos, probability=jnp.asarray(1.0))

        entities = {
            "player": player[None],
            "goal": goals[None],
        }

        # crossings
        obstacles_hor = jnp.mgrid[1 : self.height - 2 : 2, : self.width].transpose(
            1, 2, 0
        )
        obstacles_ver = jnp.mgrid[: self.height, 1 : self.width - 2 : 2].transpose(
            2, 1, 0
        )
        all_obstacles_pos = jnp.concatenate([obstacles_hor, obstacles_ver])
        num_obstacles = min(self.n_crossings, len(all_obstacles_pos))
        obstacles_pos = jax.random.choice(
            k1, all_obstacles_pos, (num_obstacles,), replace=False
        )

        if self.is_lava:
            entities["lava"] = Lava.create(position=obstacles_pos)
        else:
            grid = grid.at[tuple(obstacles_pos.T)].set(-1)

        # path to goal
        def update(direction, start, grid, step_size):
            return jax.lax.cond(
                direction == jnp.asarray(0, dtype=jnp.int32),
                lambda: (
                    start + jnp.asarray([0, step_size]),
                    jax.lax.dynamic_update_slice(
                        grid, jnp.zeros((1, step_size), dtype=jnp.int32), tuple(start.T)
                    ),
                ),
                lambda: (
                    start + jnp.asarray([step_size, 0]),
                    jax.lax.dynamic_update_slice(
                        grid, jnp.zeros((step_size, 1), dtype=jnp.int32), tuple(start.T)
                    ),
                ),
            )

        start = jnp.asarray([0, 0], dtype=jnp.int32)
        step_size = 3
        for i in range(num_obstacles * 2):
            k2, k3 = jax.random.split(k2)
            direction = jax.random.randint(k2, (), minval=0, maxval=2)
            start, grid = update(direction, start, grid, step_size)

        grid = jnp.pad(grid, 1, mode="constant", constant_values=-1)

        state = State(
            key=key,
            grid=grid,
            cache=cache or RenderingCache.init(grid),
            entities=entities,
        )
        return Timestep(
            t=jnp.asarray(0, dtype=jnp.int32),
            observation=self.observation(state),
            action=jnp.asarray(-1, dtype=jnp.int32),
            reward=jnp.asarray(0.0, dtype=jnp.float32),
            step_type=jnp.asarray(0, dtype=jnp.int32),
            state=state,
        )

    def crossing(self, size: int, *, key: Array) -> Array:
        direction = jax.random.randint(key, (), 0, 2)
        row = jax.random.randint(key, (), 2, size - 2)
        col = jax.random.randint(key, (), 2, size - 2)
        return jax.lax.cond(
            direction,
            lambda: jnp.stack(
                [jnp.arange(size), jnp.ones(size, dtype=jnp.int32) * row], axis=1
            ),
            lambda: jnp.stack(
                [jnp.ones(size, dtype=jnp.int32) * col, jnp.arange(size)], axis=1
            ),
        )


register_env(
    "Navix-Crossings-S9N1-v0",
    lambda *args, **kwargs: Crossings(
        *args, **kwargs, height=9, width=9, n_crossings=1
    ),
)
register_env(
    "Navix-Crossings-S9N2-v0",
    lambda *args, **kwargs: Crossings(
        *args, **kwargs, height=9, width=9, n_crossings=2
    ),
)
register_env(
    "Navix-Crossings-S9N3-v0",
    lambda *args, **kwargs: Crossings(
        *args, **kwargs, height=9, width=9, n_crossings=3
    ),
)
register_env(
    "Navix-Crossings-S11N5-v0",
    lambda *args, **kwargs: Crossings(
        *args, **kwargs, height=11, width=11, n_crossings=5
    ),
)
