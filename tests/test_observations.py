import jax
import jax.numpy as jnp

import navix as nx
from navix.entities import Entities, Player, Goal, Key, Door
from navix.components import EMPTY_POCKET_ID


def test_rgb():
    height = 10
    width = 10
    grid = jnp.zeros((height - 2, width - 2), dtype=jnp.int32)
    grid = jnp.pad(grid, 1, mode="constant", constant_values=-1)

    players = Player(
        position=jnp.asarray((1, 1)), direction=jnp.asarray(0), pocket=EMPTY_POCKET_ID
    )
    goals = Goal(position=jnp.asarray((4, 4)), probability=jnp.asarray(1.0))
    keys = Key(position=jnp.asarray((2, 2)), id=jnp.asarray(0))
    doors = Door(
        position=jnp.asarray([(1, 5), (1, 6)]),
        direction=jnp.asarray((0, 2)),
        requires=jnp.asarray((0, 0)),
        open=jnp.asarray((False, True)),
    )

    entities = {
        Entities.PLAYER.value: players[None],
        Entities.GOAL.value: goals[None],
        Entities.KEY.value: keys[None],
        Entities.DOOR.value: doors,
    }

    state = nx.entities.State(
        key=jax.random.PRNGKey(0),
        grid=grid,
        cache=nx.graphics.RenderingCache.init(grid),
        entities=entities,
    )
    sprites_registry = nx.graphics.SPRITES_REGISTRY

    doors = state.get_doors()
    doors = doors.replace(open=jnp.asarray((False, True)))
    state.entities[Entities.DOOR.value] = doors

    obs = nx.observations.rgb(state)
    expected_obs_shape = (
        height * nx.graphics.TILE_SIZE,
        width * nx.graphics.TILE_SIZE,
        3,
    )
    assert (
        obs.shape == expected_obs_shape
    ), f"Expected observation {expected_obs_shape}, got {obs.shape} instead"

    def get_tile(position):
        x = position[0] * nx.graphics.TILE_SIZE
        y = position[1] * nx.graphics.TILE_SIZE
        return obs[x : x + nx.graphics.TILE_SIZE, y : y + nx.graphics.TILE_SIZE, :]

    player = state.get_player()
    player_tile = get_tile(player.position)
    assert jnp.array_equal(
        player_tile, sprites_registry[Entities.PLAYER.value][player.direction]
    ), player_tile

    goals = state.get_goals()
    goal_tile = get_tile(goals.position[0])
    assert jnp.array_equal(goal_tile, sprites_registry[Entities.GOAL.value]), goal_tile

    keys = state.get_keys()
    key_tile = get_tile(keys.position[0])
    assert jnp.array_equal(key_tile, sprites_registry[Entities.KEY.value]), key_tile

    doors = state.get_doors()
    door_tile = get_tile(doors.position[0])
    direction = doors.direction[0]
    open = jnp.asarray(doors.open[0], dtype=jnp.int32)
    assert jnp.array_equal(
        door_tile, sprites_registry[Entities.DOOR.value][direction, open]
    ), door_tile

    door_tile = get_tile(doors.position[1])
    direction = doors.direction[1]
    open = jnp.asarray(doors.open[1], dtype=jnp.int32)
    assert jnp.array_equal(
        door_tile, sprites_registry[Entities.DOOR.value][direction, open]
    ), door_tile

    return


def test_categorical_first_person():
    height = 10
    width = 10
    grid = jnp.zeros((height - 2, width - 2), dtype=jnp.int32)
    grid = jnp.pad(grid, 1, mode="constant", constant_values=-1)

    players = Player(
        position=jnp.asarray((1, 1)), direction=jnp.asarray(0), pocket=EMPTY_POCKET_ID
    )
    goals = Goal(position=jnp.asarray((4, 4)), probability=jnp.asarray(1.0))
    keys = Key(position=jnp.asarray((2, 2)), id=jnp.asarray(0))
    doors = Door(
        position=jnp.asarray([(1, 5), (1, 6)]),
        direction=jnp.asarray((0, 2)),
        requires=jnp.asarray((0, 0)),
        open=jnp.asarray((False, True)),
    )
    entities = {
        Entities.PLAYER.value: players[None],
        Entities.GOAL.value: goals[None],
        Entities.KEY.value: keys[None],
        Entities.DOOR.value: doors,
    }

    state = nx.entities.State(
        key=jax.random.PRNGKey(0),
        grid=grid,
        cache=nx.graphics.RenderingCache.init(grid),
        entities=entities,
    )

    obs = nx.observations.categorical_first_person(state)
    print(obs)


if __name__ == "__main__":
    test_rgb()
    test_categorical_first_person()
    # jax.jit(test_categorical_first_person)()
