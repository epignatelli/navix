import jax
import jax.numpy as jnp
from jax.random import KeyArray

from ..graphics import RenderingCache
from ..environments import Environment
from ..components import State, Player, Pickable, Consumable, Goal
from ..environments import Timestep
from ..grid import (
    two_rooms,
    random_positions,
    random_directions,
    mask_by_coordinates,
)


class KeyDoor(Environment):
    def reset(self, key: KeyArray) -> Timestep:
        key, k1, k2, k3, k4 = jax.random.split(key, 5)

        grid, wall_at = two_rooms(height=self.height, width=self.width, key=k4)

        # add the door
        door_pos = jnp.asarray([jax.random.randint(k3, (), 1, self.height - 1), wall_at])
        doors = Consumable(position=door_pos[None], requires=jnp.asarray(3)[None])

        # spawn player and key in the first room
        out_of_bounds = jnp.asarray(self.height)
        first_room_mask = mask_by_coordinates(grid, (out_of_bounds, wall_at), jnp.less)
        first_room = jnp.where(first_room_mask, grid, -1)
        # player
        player_pos = random_positions(k1, first_room)
        player_dir = random_directions(k2)
        player = Player(position=player_pos, direction=player_dir)
        # key
        key_pos = random_positions(k2, first_room, exclude=player_pos[None])
        keys = Pickable(position=key_pos[None], id=jnp.asarray(3)[None])

        # spawn the goal in the second room
        second_room = jnp.where(first_room_mask, -1, grid)
        goal_pos = random_positions(k2, second_room)
        goals = Goal(position=goal_pos[None])


        state = State(
            key=key,
            grid=grid,
            player=player,
            goals=goals,
            keys=keys,
            doors=doors,
            cache=RenderingCache.init(grid),
        )
        return Timestep(
            t=jnp.asarray(0, dtype=jnp.int32),
            observation=self.observation(state),
            action=jnp.asarray(-1, dtype=jnp.int32),
            reward=jnp.asarray(0.0, dtype=jnp.float32),
            step_type=jnp.asarray(0, dtype=jnp.int32),
            state=state,
        )
