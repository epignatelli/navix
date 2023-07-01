import jax
import jax.numpy as jnp
from jax.random import KeyArray

from ..graphics import RenderingCache
from ..environments import Environment
from ..entities import State, Player, Key, Door, Goal, Wall
from ..environments import Timestep
from ..grid import (
    two_rooms,
    random_positions,
    random_directions,
    mask_by_coordinates,
)


class KeyDoor(Environment):
    def reset(self, key: KeyArray, cache: RenderingCache | None = None) -> Timestep:
        key, k1, k2, k3, k4 = jax.random.split(key, 5)

        grid, wall_at = two_rooms(height=self.height, width=self.width, key=k4)

        # add the door
        door_pos = jnp.asarray([jax.random.randint(k3, (), 1, self.height - 1), wall_at])
        doors = Door.create(position=door_pos, requires=jnp.asarray(3))

        # spawn player and key in the first room
        out_of_bounds = jnp.asarray(self.height)
        first_room_mask = mask_by_coordinates(grid, (out_of_bounds, wall_at), jnp.less)
        first_room = jnp.where(first_room_mask, grid, -1)
        # player
        player_pos = random_positions(k1, first_room)
        player_dir = random_directions(k2)
        player = Player.create(position=player_pos, direction=player_dir)
        # key
        key_pos = random_positions(k2, first_room, exclude=player_pos)
        keys = Key.create(position=key_pos, id=jnp.asarray(3))

        # spawn the goal in the second room
        second_room = jnp.where(first_room_mask, -1, grid)
        goal_pos = random_positions(k2, second_room)
        goals = Goal.create(position=goal_pos, probability=jnp.asarray(1.0))

        # wall entities
        rows = jnp.arange(1, self.height - 2)
        cols = jnp.ones(5, dtype=jnp.int32)
        wall_pos = jnp.stack((rows, cols), axis=1)
        walls = Wall(wall_pos)

        # remove the wall beneath the door
        grid = grid.at[tuple(door_pos)].set(0)

        state = State(
            key=key,
            grid=grid,
            players=player,
            goals=goals,
            keys=keys,
            doors=doors,
            cache=cache or RenderingCache.init(grid),
        )
        return Timestep(
            t=jnp.asarray(0, dtype=jnp.int32),
            observation=self.observation(state),
            action=jnp.asarray(-1, dtype=jnp.int32),
            reward=jnp.asarray(0.0, dtype=jnp.float32),
            step_type=jnp.asarray(0, dtype=jnp.int32),
            state=state,
        )
