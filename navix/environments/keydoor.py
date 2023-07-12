import jax
import jax.numpy as jnp
from jax.random import KeyArray
from typing import Union

from ..components import EMPTY_POCKET_ID
from ..graphics import RenderingCache
from ..environments import Environment
from ..entities import State, Player, Key, Door, Goal, Wall
from ..environments import Timestep
from ..grid import (
    room,
    random_positions,
    random_directions,
    mask_by_coordinates,
)


class KeyDoor(Environment):
    def reset(self, key: KeyArray, cache: Union[RenderingCache, None] = None) -> Timestep:  # type: ignore
        key, k1, k2, k3, k4 = jax.random.split(key, 5)

        grid = room(height=self.height, width=self.width)

        # door position
        door_col = jax.random.randint(k4, (), 2, self.width - 2)  # col
        door_row = jax.random.randint(k3, (), 1, self.height - 1)  # row
        door_pos = jnp.asarray((door_row, door_col))
        doors = Door(position=door_pos, requires=jnp.asarray(3), direction=jnp.asarray(0), open=jnp.asarray(False))

        # wall potisions
        wall_rows = jnp.arange(1, self.height - 2)
        wall_cols = jnp.asarray([door_col] * (self.height - 3))
        wall_pos = jnp.stack((wall_rows, wall_cols), axis=1)
        # remove wall where the door is
        wall_pos = jnp.delete(wall_pos, door_row, axis=0, assume_unique_indices=True)
        walls = Wall(position=wall_pos)

        # mask first room
        out_of_bounds = jnp.asarray(self.height)
        first_room_mask = mask_by_coordinates(grid, (out_of_bounds, door_row), jnp.less)
        first_room = jnp.where(first_room_mask, grid, -1)

        # spawn player
        player_pos = random_positions(k1, first_room)
        player_dir = random_directions(k2)
        player = Player(position=player_pos, direction=player_dir, pocket=EMPTY_POCKET_ID)

        # spawn key
        key_pos = random_positions(k2, first_room, exclude=player_pos)
        keys = Key(position=key_pos, id=jnp.asarray(3))

        # mask the second room
        second_room = jnp.where(first_room_mask, -1, grid)

        # spawn goal
        goal_pos = random_positions(k2, second_room)
        goals = Goal(position=goal_pos, probability=jnp.asarray(1.0))

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
            entities=entities
        )
        return Timestep(
            t=jnp.asarray(0, dtype=jnp.int32),
            observation=self.observation(state),
            action=jnp.asarray(-1, dtype=jnp.int32),
            reward=jnp.asarray(0.0, dtype=jnp.float32),
            step_type=jnp.asarray(0, dtype=jnp.int32),
            state=state,
        )
