from typing import Union
import jax
import jax.numpy as jnp
from jax import Array
from flax import struct

from ..components import EMPTY_POCKET_ID
from ..rendering.cache import RenderingCache
from ..rendering.registry import PALETTE
from . import Environment
from ..entities import State, Player, Key, Door, Goal, Wall
from . import Timestep
from ..grid import mask_by_coordinates, room, random_positions, random_directions
from .registry import register_env


class KeyDoor(Environment):
    random_start: bool = struct.field(pytree_node=False, default=False)

    def reset(self, key: Array, cache: Union[RenderingCache, None] = None) -> Timestep:
        # check minimum height and width
        assert (
            self.height > 3
        ), f"Room height must be greater than 3, got {self.height} instead"
        assert (
            self.width > 4
        ), f"Room width must be greater than 5, got {self.width} instead"

        key, k1, k2, k3, k4 = jax.random.split(key, 5)

        grid = room(height=self.height, width=self.width)

        # door positions
        # col can be between 1 and height - 2
        door_col = jax.random.randint(k4, (), 2, self.width - 2)  # col
        # row can be between 1 and height - 2
        door_row = jax.random.randint(k3, (), 1, self.height - 1)  # row
        door_pos = jnp.asarray((door_row, door_col))
        doors = Door(
            position=door_pos,
            requires=jnp.asarray(3),
            open=jnp.asarray(False),
            colour=PALETTE.YELLOW,
        )

        # wall positions
        wall_rows = jnp.arange(1, self.height - 1)
        wall_cols = jnp.asarray([door_col] * (self.height - 2))
        wall_pos = jnp.stack((wall_rows, wall_cols), axis=1)
        # remove wall where the door is
        wall_pos = jnp.delete(
            wall_pos, door_row - 1, axis=0, assume_unique_indices=True
        )
        walls = Wall(position=wall_pos)

        # get rooms
        first_room_mask = mask_by_coordinates(
            grid, (jnp.asarray(self.height), door_col), jnp.less
        )
        first_room = jnp.where(first_room_mask, grid, -1)  # put walls where not mask
        second_room_mask = mask_by_coordinates(
            grid, (jnp.asarray(0), door_col), jnp.greater
        )
        second_room = jnp.where(second_room_mask, grid, -1)  # put walls where not mask

        # set player and goal pos
        if self.random_start:
            player_pos = random_positions(k1, first_room)
            player_dir = random_directions(k2)
            goal_pos = random_positions(k2, second_room)
        else:
            player_pos = jnp.asarray([1, 1])
            player_dir = jnp.asarray(0)
            goal_pos = jnp.asarray([self.height - 2, self.width - 2])

        # spawn goal and player
        player = Player(
            position=player_pos, direction=player_dir, pocket=EMPTY_POCKET_ID
        )
        goals = Goal(position=goal_pos, probability=jnp.asarray(1.0))

        # spawn key
        key_pos = random_positions(k2, first_room, exclude=player_pos)
        keys = Key(position=key_pos, id=jnp.asarray(3), colour=PALETTE.YELLOW)

        # remove the wall beneath the door
        grid = grid.at[tuple(door_pos)].set(0)

        entities = {
            "player": player[None],
            "key": keys[None],
            "door": doors[None],
            "goal": goals[None],
            "wall": walls,
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
    "Navix-DoorKey-5x5-v0",
    lambda *args, **kwargs: KeyDoor(
        *args, **kwargs, height=5, width=5, random_start=False
    ),
)
register_env(
    "Navix-DoorKey-6x6-v0",
    lambda *args, **kwargs: KeyDoor(
        *args, **kwargs, height=6, width=6, random_start=False
    ),
)
register_env(
    "Navix-DoorKey-8x8-v0",
    lambda *args, **kwargs: KeyDoor(
        *args, **kwargs, height=8, width=8, random_start=False
    ),
)
register_env(
    "Navix-DoorKey-16x16-v0",
    lambda *args, **kwargs: KeyDoor(
        *args, **kwargs, height=16, width=16, random_start=False
    ),
)
register_env(
    "Navix-DoorKey-Random-5x5-v0",
    lambda *args, **kwargs: KeyDoor(
        *args, **kwargs, height=5, width=5, random_start=True
    ),
)
register_env(
    "Navix-DoorKey-Random-6x6-v0",
    lambda *args, **kwargs: KeyDoor(
        *args, **kwargs, height=6, width=6, random_start=True
    ),
)
register_env(
    "Navix-DoorKey-Random-8x8-v0",
    lambda *args, **kwargs: KeyDoor(
        *args, **kwargs, height=8, width=8, random_start=True
    ),
)
register_env(
    "Navix-DoorKey-Random-16x16-v0",
    lambda *args, **kwargs: KeyDoor(
        *args, **kwargs, height=16, width=16, random_start=True
    ),
)