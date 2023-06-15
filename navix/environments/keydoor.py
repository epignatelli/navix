import jax
import jax.numpy as jnp

from .environment import Environment
from ..components import Component, State, Timestep, Player, Pickable, Consumable, Goal
from ..grid import spawn_entity, place_entity


class KeyDoor(Environment):
    def reset(self, key) -> Timestep:
        key, k1, k2, k3 = jax.random.split(key, 4)

        width, rem = divmod(self.width - 1, 2)
        room_1 = jnp.zeros((self.height, width + rem), dtype=jnp.int32)
        room_1 = jnp.pad(room_1, ((0, 0), (0, 1)), mode="constant", constant_values=-1)
        player_id = 1
        key_id = 2
        room_1 = spawn_entity(room_1, player_id, k1)
        room_1 = spawn_entity(room_1, key_id, k1)

        room_2 = jnp.zeros((self.height, width), dtype=jnp.int32)
        goal_id = 3
        room_2 = spawn_entity(room_2, goal_id, k2)  # goal

        grid = jnp.concatenate([room_1, room_2], axis=1)
        grid = jnp.pad(grid, 1, mode="constant", constant_values=-1)

        door_coordinates = (
            jax.random.randint(k3, (), 1, self.height),
            jnp.asarray(width + 1),
        )
        print(door_coordinates)
        door_id = 4
        grid = place_entity(grid, door_id, door_coordinates)

        entities = {
            "player/0": Player(id=1),
            "key/0": Pickable(id=2),
            "goal/0": Goal(id=3),
            "door/0": Consumable(id=4),
        }

        state = State(key=key, grid=grid, entities=entities)
        return Timestep(
            t=jnp.asarray(0, dtype=jnp.int32),
            observation=self.observation(state),
            action=jnp.asarray(-1, dtype=jnp.int32),
            reward=jnp.asarray(0.0, dtype=jnp.float32),
            step_type=jnp.asarray(0, dtype=jnp.int32),
            state=state,
        )
