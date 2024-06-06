from typing import Dict
import jax
import jax.numpy as jnp

import navix as nx
from navix.components import EMPTY_POCKET_ID, DISCARD_PILE_COORDS
from navix.entities import Entities, Entity, Directions
from navix.states import State
from navix.rendering.registry import PALETTE


def test_rotation():
    direction = jnp.asarray(0)

    key = jax.random.PRNGKey(0)
    grid = jnp.zeros((3, 3), dtype=jnp.int32)
    player = nx.entities.Player(
        position=jnp.asarray((1, 1)), direction=direction, pocket=EMPTY_POCKET_ID
    )[None]
    cache = nx.rendering.cache.RenderingCache.init(grid)

    entities: Dict[str, Entity] = {
        Entities.PLAYER: player,
    }

    player.check_ndim(batched=True)

    state = State(
        grid=grid,
        entities=entities,
        cache=cache,
        key=key,
    )

    msg = "Expected direction to be {}, got {}"
    state = nx.actions._rotate(state, -1)
    player = state.get_player()
    player.check_ndim(batched=False)
    assert player.direction == jnp.asarray(3), msg.format(3, player.direction)

    state = nx.actions._rotate(state, 1)
    player = state.get_player()
    player.check_ndim(batched=False)
    assert player.direction == jnp.asarray(0), msg.format(0, player.direction)

    state = nx.actions._rotate(state, 2)
    player = state.get_player()
    player.check_ndim(batched=False)
    assert player.direction == jnp.asarray(2), msg.format(2, player.direction)

    state = nx.actions._rotate(state, 3)
    player = state.get_player()
    player.check_ndim(batched=False)
    assert player.direction == jnp.asarray(1), msg.format(1, player.direction)
    return


def test_move():
    heigh, width = 5, 5
    grid = jnp.zeros((heigh - 2, width - 2), dtype=jnp.int32)
    grid = jnp.pad(grid, pad_width=1, mode="constant", constant_values=1)
    key = jax.random.PRNGKey(0)
    player = nx.entities.Player(
        position=jnp.asarray((1, 1)), direction=jnp.asarray(0), pocket=EMPTY_POCKET_ID
    )
    goals = nx.entities.Goal.create(
        position=jnp.asarray((3, 3)), probability=jnp.asarray(1.0)
    )
    keys = nx.entities.Key(
        position=jnp.asarray((3, 1)), id=jnp.asarray(-1), colour=PALETTE.YELLOW
    )
    doors = nx.entities.Door(
        position=jnp.asarray((2, 2)),
        requires=jnp.asarray(-1),
        open=jnp.asarray(False),
        colour=PALETTE.YELLOW,
    )
    cache = nx.rendering.cache.RenderingCache.init(grid)

    player.check_ndim(batched=False)
    goals.check_ndim(batched=False)
    keys.check_ndim(batched=False)
    doors.check_ndim(batched=False)

    """
    #  #  #  #  #
    #  P  .  . #
    #  .  D  . #
    #  K  .  G #
    #  #  #  #  #
    """
    entities = {
        Entities.PLAYER: player[None],
        Entities.GOAL: goals[None],
        Entities.KEY: keys[None],
        Entities.DOOR: doors[None],
    }
    state = State(
        key=key,
        grid=grid,
        entities=entities,
        cache=cache,
    )

    # check forward
    state = nx.actions.forward(state)
    expected_position = jnp.asarray((1, 2))
    player = state.get_player()
    player.check_ndim(batched=False)
    assert jnp.array_equal(player.position, expected_position)

    # check backward
    state = nx.actions.backward(state)
    expected_position = jnp.asarray((1, 1))
    player = state.get_player()
    player.check_ndim(batched=False)
    assert jnp.array_equal(player.position, expected_position)

    # check right
    state = nx.actions.right(state)
    expected_position = jnp.asarray((2, 1))
    player = state.get_player()
    player.check_ndim(batched=False)
    assert jnp.array_equal(player.position, expected_position)

    # check left
    state = nx.actions.left(state)
    expected_position = jnp.asarray((1, 1))
    player = state.get_player()
    player.check_ndim(batched=False)
    assert jnp.array_equal(player.position, expected_position)

    # check that we can't walk through a closed door
    state = nx.actions.forward(state)
    state = nx.actions.rotate_cw(state)
    state = nx.actions.forward(state)
    expected_position = jnp.asarray((1, 2))
    player = state.get_player()
    player.check_ndim(batched=False)
    assert jnp.array_equal(player.position, expected_position)

    # check that we can walk through an open door
    state.entities["door"] = entities["door"].replace(open=jnp.asarray(True)[None])
    state = nx.actions.forward(state)
    state = nx.actions.forward(state)
    expected_position = jnp.asarray((3, 2))
    player = state.get_player()
    player.check_ndim(batched=False)
    assert jnp.array_equal(player.position, expected_position)

    # check that we can walk through a door backwards
    state = nx.actions.backward(state)
    state = nx.actions.backward(state)
    expected_position = jnp.asarray((1, 2))
    player = state.get_player()
    player.check_ndim(batched=False)
    assert jnp.array_equal(player.position, expected_position)


def test_walkable():
    height, width = 5, 5
    grid = nx.grid.room(height, width)
    key = jax.random.PRNGKey(0)
    player = nx.entities.Player(
        position=jnp.asarray((1, 1)), direction=Directions.EAST, pocket=EMPTY_POCKET_ID
    )
    goals = nx.entities.Goal.create(
        position=jnp.asarray((3, 3)), probability=jnp.asarray(1.0)
    )
    keys = nx.entities.Key(
        position=jnp.asarray((3, 1)), id=jnp.asarray(1), colour=PALETTE.YELLOW
    )
    doors = nx.entities.Door(
        position=jnp.asarray((1, 3)),
        requires=jnp.asarray(1),
        open=jnp.asarray(False),
        colour=PALETTE.YELLOW,
    )
    cache = nx.rendering.cache.RenderingCache.init(grid)

    player.check_ndim(batched=False)
    goals.check_ndim(batched=False)
    keys.check_ndim(batched=False)
    doors.check_ndim(batched=False)

    # Looks like this
    """
    #  #  #  #  #
    #  P  .  D #
    #  .  .  . #
    #  K  .  G #
    #  #  #  #  #
    """
    state = State(
        key=key,
        grid=grid,
        cache=cache,
        entities={
            Entities.PLAYER: player[None],
            Entities.GOAL: goals[None],
            Entities.KEY: keys[None],
            Entities.DOOR: doors[None],
        },
    )

    def check_forward_position(state: State):
        player = state.get_player()
        player.check_ndim(batched=False)
        prev_pos = player.position
        pos = nx.grid.translate_forward(prev_pos, player.direction, jnp.asarray(1))
        walkable, _ = nx.actions._can_walk_there(state, pos)
        assert (
            not walkable
        ), "Expected position {} to be not walkable, since it is a {}".format(
            pos, state.grid[tuple(pos)]
        )

    # should not be able to walk on/through a closed door
    state = nx.actions.forward(state)
    check_forward_position(state)

    # should be able to walk through an open door
    doors = state.get_doors()
    state.entities[Entities.DOOR] = doors.replace(open=jnp.asarray(True)[None])
    state = nx.actions.forward(state)
    state = nx.actions.backward(state)

    # should not be able to walk on/through a key
    state = nx.actions.backward(state)
    state = nx.actions.rotate_cw(state)
    state = nx.actions.forward(state)
    check_forward_position(state)

    # should not be able to walk on/through a wall
    state = nx.actions.rotate_ccw(state)
    state = nx.actions.forward(state)
    state = nx.actions.rotate_cw(state)
    state = nx.actions.forward(state)
    check_forward_position(state)

    # should be able to walk on/through a goal
    state = nx.actions.rotate_ccw(state)

    # but not on/through a wall
    state = nx.actions.forward(state)
    check_forward_position(state)


def test_pickup():
    heigh, width = 5, 5
    grid = jnp.zeros((heigh - 2, width - 2), dtype=jnp.int32)
    grid = jnp.pad(grid, pad_width=1, mode="constant", constant_values=1)
    key = jax.random.PRNGKey(0)
    player = nx.entities.Player(
        position=jnp.asarray((1, 1)), direction=jnp.asarray(1), pocket=EMPTY_POCKET_ID
    )
    goals = nx.entities.Goal.create(
        position=jnp.asarray((3, 3)), probability=jnp.asarray(1.0)
    )
    keys = nx.entities.Key(
        position=jnp.asarray((2, 1)), id=jnp.asarray(1), colour=PALETTE.YELLOW
    )
    doors = nx.entities.Door(
        position=jnp.asarray((1, 3)),
        requires=jnp.asarray(1),
        open=jnp.asarray(False),
        colour=PALETTE.YELLOW,
    )
    cache = nx.rendering.cache.RenderingCache.init(grid)

    # Looks like this
    """
    #  #  #  #  #
    #  P  .  D  #
    #  K  .  .  #
    #  .  .  G  #
    #  #  #  #  #
    """
    entities = {
        Entities.PLAYER: player[None],
        Entities.GOAL: goals[None],
        Entities.KEY: keys[None],
        Entities.DOOR: doors[None],
    }
    state = State(
        key=key,
        grid=grid,
        cache=cache,
        entities=entities,
    )

    # check that the player has no keys
    player = state.get_player()
    player.check_ndim(batched=False)
    assert jnp.array_equal(
        player.pocket, EMPTY_POCKET_ID
    ), "Expected player to have pocket {}, got {}".format(
        EMPTY_POCKET_ID, player.pocket
    )

    # pick up the key
    state = nx.actions.pickup(state)

    # check that the player has the key
    player = state.get_player()
    player.check_ndim(batched=False)
    expected_pocket = jnp.asarray(1)
    assert jnp.array_equal(
        player.pocket, expected_pocket
    ), "Expected player to have key {}, got {}".format(expected_pocket, player.pocket)

    # check that the key is no longer on the grid
    keys = state.get_keys()
    assert jnp.array_equal(
        keys.position[0], DISCARD_PILE_COORDS
    ), "Expected key to be at {}, got {}".format(DISCARD_PILE_COORDS, keys.position)


def test_open():
    heigh, width = 5, 5
    grid = jnp.zeros((heigh - 2, width - 2), dtype=jnp.int32)
    grid = jnp.pad(grid, pad_width=1, mode="constant", constant_values=1)
    key = jax.random.PRNGKey(0)
    player = nx.entities.Player(
        position=jnp.asarray((1, 1)), direction=jnp.asarray(0), pocket=EMPTY_POCKET_ID
    )
    goals = nx.entities.Goal.create(
        position=jnp.asarray((3, 3)), probability=jnp.asarray(1.0)
    )
    keys = nx.entities.Key(
        position=jnp.asarray((3, 1)), id=jnp.asarray(1), colour=PALETTE.YELLOW
    )
    doors = nx.entities.Door(
        position=jnp.asarray((1, 3)),
        requires=jnp.asarray(1),
        open=jnp.asarray(False),
        colour=PALETTE.YELLOW,
    )
    cache = nx.rendering.cache.RenderingCache.init(grid)

    player.check_ndim(batched=False)
    goals.check_ndim(batched=False)
    keys.check_ndim(batched=False)
    doors.check_ndim(batched=False)

    # Looks like this
    # W  W  W  W  W
    # W  P  0  D  W
    # W  0  0  0  W
    # W  K  0  G  W
    # W  W  W  W  W
    entities = {
        Entities.PLAYER: player[None],
        Entities.GOAL: goals[None],
        Entities.KEY: keys[None],
        Entities.DOOR: doors[None],
    }

    state = State(
        key=key,
        grid=grid,
        cache=cache,
        entities=entities,
    )

    # check that the player has no keys
    player = state.get_player()
    player.check_ndim(batched=False)
    expected_pocket = EMPTY_POCKET_ID
    assert jnp.array_equal(
        player.pocket, expected_pocket
    ), "Expected player to have {}, got {}".format(expected_pocket, player.pocket)

    state = nx.actions.forward(state)

    # check that pocket is empty
    player = state.get_player()
    player.check_ndim(batched=False)
    expected_pocket = EMPTY_POCKET_ID
    assert jnp.array_equal(
        player.pocket, expected_pocket
    ), "Expected player to have pocket {}, got {}".format(
        expected_pocket, player.pocket
    )

    # check that we cannot open a door without the required key
    state = nx.actions.open(state)
    # and that the door is still closed
    doors = state.get_doors()
    doors.check_ndim(batched=True)
    expected_open = jnp.asarray(False)[None]
    assert jnp.array_equal(
        doors.open, expected_open
    ), "Expected door open status {}, got {}".format(expected_open, doors.open)

    # artificially put the right key in the player's pocket
    player = state.get_player()
    player.check_ndim(batched=False)
    player = player.replace(pocket=jnp.asarray(1))
    player.check_ndim(batched=False)
    state = state.set_player(player)
    expected_pocket = jnp.asarray(1)
    assert jnp.array_equal(
        player.pocket, expected_pocket
    ), "Expected player to have pocket {}, got {}".format(
        expected_pocket, player.pocket
    )

    # check that we can open the door with the right key
    state = nx.actions.open(state)
    doors = state.get_doors()
    doors.check_ndim(batched=True)
    expected_open = jnp.asarray(True)[None]
    assert jnp.array_equal(
        doors.open, expected_open
    ), "Expected door open status {}, got {}".format(expected_open, doors.open)

    # check that opening an open door keeps it open
    state = nx.actions.open(state)
    doors = state.get_doors()
    doors.check_ndim(batched=True)
    expected_open = jnp.asarray(True)[None]
    assert jnp.array_equal(
        doors.open, expected_open
    ), "Expected door open status {}, got {}".format(expected_open, doors.open)

    # check that we can walk through an open door
    state = nx.actions.forward(state)
    player = state.get_player()
    player.check_ndim(batched=False)
    expected_position = jnp.asarray((1, 3))
    assert jnp.array_equal(
        player.position, expected_position
    ), "Expected player position to be {}, got {}".format(
        expected_position, player.position
    )


if __name__ == "__main__":
    # test_rotation()
    # test_move()
    test_walkable()
    test_pickup()
    # test_open()
