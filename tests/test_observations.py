import jax
import jax.numpy as jnp

import navix as nx
from navix.states import State
from navix.entities import Entities, Player, Goal, Key, Door
from navix.components import EMPTY_POCKET_ID
from navix.rendering.cache import RenderingCache, TILE_SIZE
from navix.rendering.registry import SPRITES_REGISTRY, PALETTE


def test_rgb():
    height = 10
    width = 10
    grid = jnp.zeros((height - 2, width - 2), dtype=jnp.int32)
    grid = jnp.pad(grid, 1, mode="constant", constant_values=-1)

    players = Player(
        position=jnp.asarray((1, 1)), direction=jnp.asarray(0), pocket=EMPTY_POCKET_ID
    )
    goals = Goal.create(position=jnp.asarray((4, 4)), probability=jnp.asarray(1.0))
    keys = Key(position=jnp.asarray((2, 2)), id=jnp.asarray(0), colour=PALETTE.YELLOW)
    doors = Door(
        position=jnp.asarray([(1, 5), (1, 6)]),
        requires=jnp.asarray((0, 0)),
        open=jnp.asarray((False, True)),
        colour=PALETTE.YELLOW[None],
    )

    entities = {
        Entities.PLAYER: players[None],
        Entities.GOAL: goals[None],
        Entities.KEY: keys[None],
        Entities.DOOR: doors,
    }

    state = State(
        key=jax.random.PRNGKey(0),
        grid=grid,
        cache=RenderingCache.init(grid),
        entities=entities,
    )
    sprites_registry = SPRITES_REGISTRY

    doors = state.get_doors()
    doors = doors.replace(open=jnp.asarray((False, True)))
    state.entities[Entities.DOOR] = doors

    obs = nx.observations.rgb(state)
    expected_obs_shape = (
        height * TILE_SIZE,
        width * TILE_SIZE,
        3,
    )
    assert (
        obs.shape == expected_obs_shape
    ), f"Expected observation {expected_obs_shape}, got {obs.shape} instead"

    def get_tile(position):
        x = position[0] * TILE_SIZE
        y = position[1] * TILE_SIZE
        return obs[x : x + TILE_SIZE, y : y + TILE_SIZE, :]

    player = state.get_player()
    player_tile = get_tile(player.position)
    assert jnp.array_equal(
        player_tile, sprites_registry[Entities.PLAYER][player.direction]
    ), player_tile

    goals = state.get_goals()
    goal_tile = get_tile(goals.position[0])
    assert jnp.array_equal(goal_tile, sprites_registry[Entities.GOAL]), goal_tile

    keys = state.get_keys()
    key_tile = get_tile(keys.position[0])
    colour = keys.colour[0]
    assert jnp.array_equal(key_tile, sprites_registry[Entities.KEY][colour]), key_tile

    doors = state.get_doors()
    door = doors[0]
    door_tile = get_tile(door.position)
    colour = door.colour
    idx = jnp.asarray(door.open + 2 * door.locked, dtype=jnp.int32)
    assert jnp.array_equal(
        door_tile, sprites_registry[Entities.DOOR][colour, idx]
    ), door_tile

    door = doors[1]
    door_tile = get_tile(door.position)
    colour = door.colour
    idx = jnp.asarray(door.open + 2 * door.locked, dtype=jnp.int32)
    assert jnp.array_equal(
        door_tile, sprites_registry[Entities.DOOR][colour, idx]
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
    goals = Goal.create(position=jnp.asarray((4, 4)), probability=jnp.asarray(1.0))
    keys = Key(position=jnp.asarray((2, 2)), id=jnp.asarray(0), colour=PALETTE.YELLOW)
    doors = Door(
        position=jnp.asarray([(1, 5), (1, 6)]),
        requires=jnp.asarray((0, 0)),
        open=jnp.asarray((False, True)),
        colour=PALETTE.YELLOW,
    )
    entities = {
        Entities.PLAYER: players[None],
        Entities.GOAL: goals[None],
        Entities.KEY: keys[None],
        Entities.DOOR: doors,
    }

    state = State(
        key=jax.random.PRNGKey(0),
        grid=grid,
        cache=RenderingCache.init(grid),
        entities=entities,
    )

    obs = nx.observations.categorical_first_person(state)
    print(obs)


if __name__ == "__main__":
    test_rgb()
    # test_categorical_first_person()
    # jax.jit(test_categorical_first_person)()
