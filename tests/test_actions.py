import jax
import jax.numpy as jnp

import navix as nx
from navix.components import EMPTY_POCKET_ID, DISCARD_PILE_COORDS
from navix.entities import State


def test_rotation():
    direction = jnp.asarray(0)

    key = jax.random.PRNGKey(0)
    grid = jnp.zeros((3, 3), dtype=jnp.int32)
    player = nx.entities.Player.create(position=jnp.asarray((1, 1)), direction=direction)
    cache = nx.graphics.RenderingCache.init(grid)

    state = nx.entities.State(
        grid=grid,
        players=player,
        cache=cache,
        key=key,
    )

    msg = "Expected direction to be {}, got {}"
    state = nx.actions._rotate(state, -1)
    assert state.players.direction == jnp.asarray(3), msg.format(
        3, state.players.direction
    )

    state = nx.actions._rotate(state, 1)
    assert state.players.direction == jnp.asarray(0), msg.format(
        0, state.players.direction
    )

    state = nx.actions._rotate(state, 2)
    assert state.players.direction == jnp.asarray(2), msg.format(
        2, state.players.direction
    )

    state = nx.actions._rotate(state, 3)
    assert state.players.direction == jnp.asarray(1), msg.format(
        1, state.players.direction
    )
    return


def test_move():
    heigh, width = 5, 5
    grid = jnp.zeros((heigh - 2, width - 2), dtype=jnp.int32)
    grid = jnp.pad(grid, pad_width=1, mode="constant", constant_values=1)
    key = jax.random.PRNGKey(0)
    player = nx.entities.Player.create(position=jnp.asarray((1, 1)), direction=jnp.asarray(0))
    goals = nx.entities.Goal.create(position=jnp.asarray((3, 3)), probability=jnp.asarray(1.0)[None])
    keys = nx.entities.Key.create(position=jnp.asarray((3, 1)))
    doors = nx.entities.Door.create(position=jnp.asarray((2, 2)))
    cache = nx.graphics.RenderingCache.init(grid)
    """
    #  #  #  #  #
    #  P  .  . #
    #  .  D  . #
    #  K  .  G #
    #  #  #  #  #
    """

    state = nx.entities.State(
        grid=grid,
        players=player,
        goals=goals,
        keys=keys,
        doors=doors,
        cache=cache,
        key=key,
    )

    # check forward
    state = nx.actions.forward(state)
    expected_position = jnp.asarray((1, 2))
    assert jnp.array_equal(state.players.position, expected_position)

    # check backward
    state = nx.actions.backward(state)
    expected_position = jnp.asarray((1, 1))
    assert jnp.array_equal(state.players.position, expected_position)

    # check right
    state = nx.actions.right(state)
    expected_position = jnp.asarray((2, 1))
    assert jnp.array_equal(state.players.position, expected_position)

    # check left
    state = nx.actions.left(state)
    expected_position = jnp.asarray((1, 1))
    assert jnp.array_equal(state.players.position, expected_position)

    # check that we can't walk through a closed door
    state = nx.actions.forward(state)
    state = nx.actions.rotate_cw(state)
    state = nx.actions.forward(state)
    expected_position = jnp.asarray((1, 2))
    assert jnp.array_equal(state.players.position, expected_position)

    # check that we can walk through an open door
    state = state.replace(doors=state.doors.replace(open=jnp.asarray(True)[None]))
    state = nx.actions.forward(state)
    state = nx.actions.forward(state)
    expected_position = jnp.asarray((3, 2))
    assert jnp.array_equal(state.players.position, expected_position)

    # check that we can walk through a door backwards
    state = nx.actions.backward(state)
    state = nx.actions.backward(state)
    expected_position = jnp.asarray((1, 2))
    assert jnp.array_equal(state.players.position, expected_position)


def test_walkable():
    heigh, width = 5, 5
    grid = jnp.zeros((heigh - 2, width - 2), dtype=jnp.int32)
    grid = jnp.pad(grid, pad_width=1, mode="constant", constant_values=1)
    key = jax.random.PRNGKey(0)
    player = nx.entities.Player.create(position=jnp.asarray((1, 1)), direction=jnp.asarray(0))
    goals = nx.entities.Goal.create(position=jnp.asarray((3, 3)), probability=jnp.asarray(1.0)[None])
    keys = nx.entities.Key.create(position=jnp.asarray((3, 1)))
    doors = nx.entities.Door.create(position=jnp.asarray((1, 3)))
    cache = nx.graphics.RenderingCache.init(grid)
    # Looks like this
    """
    #  #  #  #  #
    #  P  .  D #
    #  .  .  . #
    #  K  .  G #
    #  #  #  #  #
    """
    state = nx.entities.State(
        grid=grid,
        players=player,
        goals=goals,
        keys=keys,
        doors=doors,
        cache=cache,
        key=key,
    )

    def check_forward_position(state: State):
        prev_pos = state.players.position
        pos = nx.grid.translate_forward(prev_pos, state.players.direction, jnp.asarray(1))
        assert not nx.actions._walkable(state, pos), (
            "Expected position {} to be not walkable, since it is a {}".format(pos, state.grid[tuple(pos)])
        )

    # should not be able to walk on/through a closed door
    state = nx.actions.forward(state)
    check_forward_position(state)

    # should be able to walk through an open door
    state = state.replace(doors=state.doors.replace(open=jnp.asarray(True)[None]))
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
    player = nx.entities.Player.create(position=jnp.asarray((1, 1)), direction=jnp.asarray(1))
    goals = nx.entities.Goal.create(position=jnp.asarray((3, 3)), probability=jnp.asarray(1.0)[None])
    keys = nx.entities.Key.create(position=jnp.asarray((2, 1)), id=jnp.asarray(1)[None][None])
    doors = nx.entities.Door.create(position=jnp.asarray((1, 3)), requires=jnp.asarray(1)[None])
    cache = nx.graphics.RenderingCache.init(grid)

    # Looks like this
    """
    #  #  #  #  #
    #  P  .  D  #
    #  K  .  .  #
    #  .  .  G  #
    #  #  #  #  #
    """
    state = nx.entities.State(
        grid=grid,
        players=player,
        goals=goals,
        keys=keys,
        doors=doors,
        cache=cache,
        key=key,
    )

    # check that the player has no keys
    assert jnp.array_equal(state.players.pocket, EMPTY_POCKET_ID), (
        "Expected player to have pocket {}, got {}".format(EMPTY_POCKET_ID, state.players.pocket)
    )

    # pick up the key
    state = nx.actions.pickup(state)

    # check that the player has the key
    expected_pocket = jnp.asarray(1)
    assert jnp.array_equal(state.players.pocket, expected_pocket), (
        "Expected player to have key {}, got {}".format(expected_pocket, state.players.pocket)
    )

    # check that the key is no longer on the grid
    assert jnp.array_equal(state.keys.position[0], DISCARD_PILE_COORDS), (
        "Expected key to be at {}, got {}".format(DISCARD_PILE_COORDS, state.keys.position)
    )


def test_open():
    heigh, width = 5, 5
    grid = jnp.zeros((heigh - 2, width - 2), dtype=jnp.int32)
    grid = jnp.pad(grid, pad_width=1, mode="constant", constant_values=1)
    key = jax.random.PRNGKey(0)
    player = nx.entities.Player.create(position=jnp.asarray((1, 1)), direction=jnp.asarray(0))
    goals = nx.entities.Goal.create(position=jnp.asarray((3, 3))[None], probability=jnp.asarray(1.0)[None])
    keys = nx.entities.Key.create(position=jnp.asarray((3, 1))[None], id=jnp.asarray(1)[None])
    doors = nx.entities.Door.create(position=jnp.asarray((1, 3))[None], requires=jnp.asarray(1)[None])
    cache = nx.graphics.RenderingCache.init(grid)

    # Looks like this
    # W  W  W  W  W
    # W  P  0  D  W
    # W  0  0  0  W
    # W  K  0  G  W
    # W  W  W  W  W

    state = nx.entities.State(
        grid=grid,
        players=player,
        goals=goals,
        keys=keys,
        doors=doors,
        cache=cache,
        key=key,
    )

    # check that the player has no keys
    expected_pocket = EMPTY_POCKET_ID
    assert jnp.array_equal(state.players.pocket, expected_pocket), (
        "Expected player to have {}, got {}".format(expected_pocket, state.players.pocket)
    )

    state = nx.actions.forward(state)

    # check that pocket is empty
    expected_pocket = EMPTY_POCKET_ID
    assert jnp.array_equal(state.players.pocket, expected_pocket), (
        "Expected player to have pocket {}, got {}".format(expected_pocket, state.players.pocket)
    )

    # check that we cannot open a door without the required key
    state = nx.actions.open(state)
    # and that the door is still closed
    expected_open = jnp.asarray(False)[None]
    assert jnp.array_equal(state.doors.open, expected_open), (
        "Expected door open status {}, got {}".format(expected_open, state.doors.open)
    )

    # artificially put the right key in the player's pocket
    state = state.replace(players=state.players.replace(pocket=jnp.asarray(1)))
    expected_pocket = jnp.asarray(1)
    assert jnp.array_equal(state.players.pocket, expected_pocket), (
        "Expected player to have pocket {}, got {}".format(expected_pocket, state.players.pocket)
    )

    # check that we can open the door with the right key
    state = nx.actions.open(state)
    expected_open = jnp.asarray(True)[None]
    assert jnp.array_equal(state.doors.open, expected_open), (
        "Expected door open status {}, got {}".format(expected_open, state.doors.open)
    )

    # check that opening an open door keeps it open
    state = nx.actions.open(state)
    expected_open = jnp.asarray(True)[None]
    assert jnp.array_equal(state.doors.open, expected_open), (
        "Expected door open status {}, got {}".format(expected_open, state.doors.open)
    )

    # check that we can walk through an open door
    state = nx.actions.forward(state)
    expected_position = jnp.asarray((1, 3))
    assert jnp.array_equal(state.players.position, expected_position), (
        "Expected player position to be {}, got {}".format(expected_position, state.players.position)
    )


if __name__ == "__main__":
    # test_rotation()
    test_walkable()
    # test_pickup()
