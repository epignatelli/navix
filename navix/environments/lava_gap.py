from typing import Union
import jax
import jax.numpy as jnp
from jax import Array

from ..components import EMPTY_POCKET_ID
from ..rendering.cache import RenderingCache
from . import Environment
from ..entities import State, Player, Goal, Lava
from . import Timestep
from ..grid import room
from .registry import register_env


class LavaGap(Environment):

    def reset(self, key: Array, cache: Union[RenderingCache, None] = None) -> Timestep:
        # check minimum height and width
        assert (
            self.height > 3
        ), f"Room height must be greater than 3, got {self.height} instead"
        assert (
            self.width > 4
        ), f"Room width must be greater than 5, got {self.width} instead"

        key, k1, k2 = jax.random.split(key, num=3)

        grid = room(height=self.height, width=self.width)

        # player
        player_pos = jnp.asarray([1, 1])
        player_dir = jnp.asarray(0)
        player = Player(
            position=player_pos, direction=player_dir, pocket=EMPTY_POCKET_ID
        )
        # goal
        goal_pos = jnp.asarray([self.height - 2, self.width - 2])
        goals = Goal(position=goal_pos, probability=jnp.asarray(1.0))

        # lava positions
        gap_row = jax.random.randint(k1, (), 1, self.height - 1)  # col

        col = jax.random.randint(k2, (), minval=2, maxval=self.width - 2)
        lava_row = jnp.arange(1, self.height - 1)
        lava_cols = jnp.asarray([col] * (self.height - 2))
        lava_pos = jnp.stack((lava_row, lava_cols), axis=1)
        # remove lava where the door is
        lava_pos = jnp.delete(
            lava_pos, gap_row - 1, axis=0, assume_unique_indices=True
        )
        lavas = Lava(position=lava_pos)

        entities = {
            "player": player[None],
            "goal": goals[None],
            "lava": lavas,
        }

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


register_env(
    "Navix-DoorKey-S5-v0",
    lambda *args, **kwargs: LavaGap(
        *args, **kwargs, height=5, width=5
    ),
)
register_env(
    "Navix-DoorKey-S6-v0",
    lambda *args, **kwargs: LavaGap(
        *args, **kwargs, height=6, width=6
    ),
)
register_env(
    "Navix-DoorKey-S7-v0",
    lambda *args, **kwargs: LavaGap(
        *args, **kwargs, height=7, width=7
    ),
)
