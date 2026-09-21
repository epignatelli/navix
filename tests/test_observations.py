import pytest

import jax
import jax.numpy as jnp

import navix as nx
from navix.states import State
from navix.entities import Entities, EntityIds, Player, Goal, Key, Door
from navix.components import EMPTY_POCKET_ID
from navix.rendering.cache import RenderingCache, TILE_SIZE
from navix.rendering import minigrid_tiles
from navix.rendering.registry import PALETTE
from navix.environments.environment import MAX_CATEGORICAL_VALUE


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
        colour=jnp.stack([PALETTE.YELLOW, PALETTE.YELLOW]),
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

    # every cell is MiniGrid's tile for its symbol, [agent, highlight, ...]
    table = minigrid_tiles.tiles(TILE_SIZE)
    seen = nx.observations.world_vis(state)

    def expected_tile(position, symbol, agent=0):
        return table[(agent, int(seen[tuple(position)]), *symbol)]

    player = state.get_player()
    player_tile = get_tile(player.position)
    floor = (EntityIds.FLOOR, 0, 0)
    assert jnp.array_equal(
        player_tile, expected_tile(player.position, floor, 1 + player.direction)
    ), player_tile

    goal = state.get_goals().position[0]
    goal_tile = get_tile(goal)
    symbol = (EntityIds.GOAL, PALETTE.GREEN, 0)
    assert jnp.array_equal(goal_tile, expected_tile(goal, symbol)), goal_tile

    key = state.get_keys()[0]
    key_tile = get_tile(key.position)
    symbol = (EntityIds.KEY, key.colour, 0)
    assert jnp.array_equal(key_tile, expected_tile(key.position, symbol)), key_tile

    for door in (doors[0], doors[1]):
        door_tile = get_tile(door.position)
        # MiniGrid's door states: open 0, closed 1, locked 2
        door_state = 0 if door.open else 2 if door.locked else 1
        symbol = (EntityIds.DOOR, door.colour, door_state)
        assert jnp.array_equal(
            door_tile, expected_tile(door.position, symbol)
        ), door_tile


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


def test_categorical_first_person_transparency_matches_rgb():
    # regression test for #112: categorical_first_person negated
    # transparency (`~transparent`) while rgb_first_person did not, so a
    # transparent entity (e.g. a Key, which Entity.transparent documents as
    # True) incorrectly blocked vision under categorical_first_person while
    # correctly letting it through under rgb_first_person - the two
    # observation types disagreed about which entities occlude vision.
    height, width = 10, 10
    grid = jnp.zeros((height - 2, width - 2), dtype=jnp.int32)
    grid = jnp.pad(grid, 1, mode="constant", constant_values=-1)

    player = Player(
        position=jnp.asarray((1, 1)), direction=jnp.asarray(0), pocket=EMPTY_POCKET_ID
    )
    # a transparent entity directly in front of the player...
    key = Key(position=jnp.asarray((1, 2)), id=jnp.asarray(0), colour=PALETTE.YELLOW)
    # ...must not hide what's behind it
    goal = Goal.create(position=jnp.asarray((1, 3)), probability=jnp.asarray(1.0))

    entities = {
        Entities.PLAYER: player[None],
        Entities.KEY: key[None],
        Entities.GOAL: goal[None],
    }
    state = State(
        key=jax.random.PRNGKey(0),
        grid=grid,
        cache=RenderingCache.init(grid),
        entities=entities,
    )

    obs = nx.observations.categorical_first_person(state)
    assert jnp.any(obs == EntityIds.GOAL), (
        "Expected the Goal to be visible through the transparent Key, "
        "got observation:\n{}".format(obs)
    )


def test_rgb_first_person():
    import gymnasium as gym
    import minigrid

    navix_env_id = "Navix-Empty-8x8-v0"
    gym_env_id = navix_env_id.replace("Navix", "MiniGrid")

    env = nx.make(navix_env_id, observation_fn=nx.observations.rgb_first_person)
    timestep = env.reset(jax.random.PRNGKey(0))

    env = gym.make(gym_env_id)
    # navix always occludes; MiniGrid's Empty registers see_through_walls
    env.unwrapped.see_through_walls = False
    env = minigrid.wrappers.RGBImgPartialObsWrapper(env, tile_size=TILE_SIZE)
    obs, _ = env.reset()
    # Empty starts the agent at the same corner, facing east, in both
    assert jnp.array_equal(timestep.observation, obs["image"])


def test_categorical_first_person_stays_inside_its_space():
    # crop() pads off-map cells with 100, which is not an EntityId and
    # sits outside the Discrete(MAX_CATEGORICAL_VALUE) space the
    # observation declares; masking the crop turns them into UNKNOWN.
    # Empty-8x8 is small enough that every pose has off-map cells in view.
    env = nx.make(
        "Navix-Empty-8x8-v0", observation_fn=nx.observations.categorical_first_person
    )
    obs = env.reset(jax.random.PRNGKey(0)).observation
    assert jnp.any(obs == EntityIds.UNKNOWN), obs
    assert int(jnp.max(obs)) < MAX_CATEGORICAL_VALUE, obs


if __name__ == "__main__":
    test_rgb()
    test_categorical_first_person_stays_inside_its_space()
    # test_categorical_first_person()
    # jax.jit(test_categorical_first_person)()


def occluded_room_state(pocket=EMPTY_POCKET_ID):
    """A room split by a wall stub, with a key behind it and one in hand."""
    height, width = 10, 10
    grid = jnp.zeros((height - 2, width - 2), dtype=jnp.int32)
    grid = jnp.pad(grid, 1, mode="constant", constant_values=-1)
    # a wall the player's line of sight cannot round, and a goal behind it
    grid = grid.at[1:5, 3].set(-1)

    player = Player(
        position=jnp.asarray((1, 1)), direction=jnp.asarray(0), pocket=pocket
    )
    goal = Goal.create(position=jnp.asarray((1, 4)), probability=jnp.asarray(1.0))
    key = Key(position=jnp.asarray((5, 5)), id=jnp.asarray(3), colour=PALETTE.YELLOW)
    entities = {
        Entities.PLAYER: player[None],
        Entities.GOAL: goal[None],
        Entities.KEY: key[None],
    }
    return State(
        key=jax.random.PRNGKey(0),
        grid=grid,
        cache=RenderingCache.init(grid),
        entities=entities,
    )


def test_symbolic_first_person_hides_what_the_player_cannot_see():
    # symbolic_first_person applied no visibility mask at all, so the
    # player read the whole cropped window through walls - unlike
    # categorical_first_person and rgb_first_person, which mask with
    # MiniGrid's process_vis.
    state = occluded_room_state()

    obs = nx.observations.symbolic_first_person(state)
    view = nx.observations.first_person_vis(state)

    assert not jnp.any(obs[..., 0] == EntityIds.GOAL), (
        "Expected the goal behind the wall to be hidden, got:\n{}".format(obs[..., 0])
    )
    assert jnp.any(obs[..., 0] == EntityIds.UNKNOWN)
    # every unseen cell, and only those, reads MiniGrid's "unseen" symbol
    assert jnp.array_equal(jnp.all(obs == 0, axis=-1), ~view)


def test_symbolic_first_person_shows_the_pocket():
    # MiniGrid's gen_obs_grid writes the carried object into the player's
    # own cell, and an empty cell when nothing is carried.
    empty = nx.observations.symbolic_first_person(occluded_room_state())
    carrying = nx.observations.symbolic_first_person(
        occluded_room_state(pocket=jnp.asarray(3))
    )
    own_cell = (2 * nx.observations.RADIUS, nx.observations.RADIUS)

    assert jnp.array_equal(
        empty[own_cell], jnp.asarray([EntityIds.FLOOR, 0, 0], dtype=jnp.uint8)
    )
    assert jnp.array_equal(
        carrying[own_cell],
        jnp.asarray([EntityIds.KEY, PALETTE.YELLOW, 0], dtype=jnp.uint8),
    )


def test_categorical_first_person_shows_the_pocket():
    # the player's own cell carries the pocket's tag, PLAYER when empty.
    empty = nx.observations.categorical_first_person(occluded_room_state())
    carrying = nx.observations.categorical_first_person(
        occluded_room_state(pocket=jnp.asarray(3))
    )
    own_cell = (2 * nx.observations.RADIUS, nx.observations.RADIUS)

    assert empty[own_cell] == EntityIds.PLAYER
    assert carrying[own_cell] == EntityIds.KEY
