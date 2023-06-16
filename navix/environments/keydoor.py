import jax
import jax.numpy as jnp

from navix.environments import Environment
from navix.components import State, Timestep, Player, Pickable, Consumable, Goal
from navix.grid import two_rooms, random_positions, random_directions


class KeyDoor(Environment):
    def reset(self, key) -> Timestep:
        key, k1, k2, k3 = jax.random.split(key, 4)

        room = two_rooms(self.width, self.height)

        # spawn player and key in the first room
        first_room = room.at[:, self.width // 2:].set(-1)
        player_pos, key_pos = random_positions(k1, first_room, n=2)
        player_dir = random_directions(k2)
        player = Player(position=player_pos, direction=player_dir)
        keys = Pickable(position=key_pos[None], id=jnp.asarray(3)[None])

        # spawn the goal in the second room
        second_room = room.at[:, :self.width // 2].set(-1)
        goal_pos = random_positions(k2, second_room, n=1)
        goals = Goal(position=goal_pos[None])

        # add the door
        door_coordinates = jnp.asarray([
            jax.random.randint(k3, (), 1, self.height),
            jnp.asarray(self.width // 2),
        ])
        doors = Consumable(position=door_coordinates[None], requires=jnp.asarray(3)[None])

        state = State(
            key=key,
            grid=room,
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
