import jax
import jax.numpy as jnp
from jax.random import KeyArray
from typing import Union

from ..components import EMPTY_POCKET_ID
from ..graphics import RenderingCache
from ..environments import Environment
from ..entities import State, Player, Key, Door, Goal, Wall
from ..environments import Timestep
from ..grid import mask_by_coordinates, room, random_positions, random_directions


class KeyDoor(Environment):
    def reset(self, key: KeyArray, cache: Union[RenderingCache, None] = None) -> Timestep:  # type: ignore
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
            direction=jnp.asarray(0),
            open=jnp.asarray(False),
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

        # spawn player
        player_pos = random_positions(k1, first_room)
        player_dir = random_directions(k2)
        player = Player(
            position=player_pos, direction=player_dir, pocket=EMPTY_POCKET_ID
        )

        # spawn key
        key_pos = random_positions(k2, first_room, exclude=player_pos)
        keys = Key(position=key_pos, id=jnp.asarray(3))

        # mask the second room

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
