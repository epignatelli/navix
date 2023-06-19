import jax
import jax.numpy as jnp

from navix.environments import Environment
from navix.components import State, Player, Pickable, Consumable, Goal
from navix.environments import Timestep
from navix.grid import two_rooms, random_positions, random_directions, mask_by_coordinates


class KeyDoor(Environment):
    def reset(self, key) -> Timestep:
        key, k1, k2, k3, k4 = jax.random.split(key, 5)

        grid, wall_at = two_rooms(height=self.height, width=self.width, key=k4)

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
        goal_pos = random_positions(k2, second_room, exclude=jnp.stack([player_pos, key_pos]))
        goals = Goal(position=goal_pos[None])

        # add the door
        door_coordinates = jnp.asarray(
            [
                jax.random.randint(k3, (), 1, self.height - 1),
                wall_at,
            ]
        )
        doors = Consumable(
            position=door_coordinates[None], requires=jnp.asarray(3)[None]
        )

        state = State(
            key=key,
            grid=grid,
            player=player,
            goals=goals,
            keys=keys,
            doors=doors,
        )
        return Timestep(
            t=jnp.asarray(0, dtype=jnp.int32),
            observation=self.observation(state),
            action=jnp.asarray(-1, dtype=jnp.int32),
            reward=jnp.asarray(0.0, dtype=jnp.float32),
            step_type=jnp.asarray(0, dtype=jnp.int32),
            state=state,
        )
