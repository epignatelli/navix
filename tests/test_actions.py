import jax
import jax.numpy as jnp

import navix as nx


def test_rotation():
    direction = jnp.asarray(0)

    key = jax.random.PRNGKey(0)
    grid = jnp.zeros((3, 3), dtype=jnp.int32)
    player = nx.components.Player(position=jnp.asarray((1, 1)), direction=direction)
    cache = nx.graphics.RenderingCache(grid)

    state = nx.components.State(
        grid=grid,
        player=player,
        cache=cache,
        key=key,
    )

    msg = "Expected direction to be {}, got {}"
    state = nx.actions._rotate(state, -1)
    assert state.player.direction == jnp.asarray(3), msg.format(
        3, state.player.direction
    )

    state = nx.actions._rotate(state, 1)
    assert state.player.direction == jnp.asarray(0), msg.format(
        0, state.player.direction
    )

    state = nx.actions._rotate(state, 2)
    assert state.player.direction == jnp.asarray(2), msg.format(
        2, state.player.direction
    )

    state = nx.actions._rotate(state, 3)
    assert state.player.direction == jnp.asarray(1), msg.format(
        1, state.player.direction
    )
    return


def test_walkable():
    heigh, width = 5, 5
    grid = jnp.zeros((heigh - 2, width - 2), dtype=jnp.int32)
    grid = jnp.pad(grid, pad_width=1, mode="constant", constant_values=1)
    key = jax.random.PRNGKey(0)
    player = nx.components.Player(position=jnp.asarray((1, 1)), direction=jnp.asarray(0))
    goals = nx.components.Goal(position=jnp.asarray((3, 3))[None])
    keys = nx.components.Key(position=jnp.asarray((3, 1))[None])
    doors = nx.components.Door(position=jnp.asarray((1, 3))[None])
    cache = nx.graphics.RenderingCache(grid)
    # Looks like this
    # -1 -1 -1 -1 -1
    # -1  P  0  D -1
    # -1  0  0  0 -1
    # -1  K  0  G -1
    # -1 -1 -1 -1 -1

    state = nx.components.State(
        grid=grid,
        player=player,
        goals=goals,
        keys=keys,
        doors=doors,
        cache=cache,
        key=key,
    )

    def check_forward_position(state):
        prev_pos = state.player.position
        pos = nx.grid.translate_forward(prev_pos, state.player.direction, jnp.asarray(1))
        assert not nx.actions._walkable(state, pos), (
            "Expected position {} to be not walkable, since it is a {}".format(pos, state.grid[tuple(pos)])
        )

    state = nx.actions.forward(state)

    # should not be able to walk on/through a door
    check_forward_position(state)

    state = nx.actions.backward(state)
    state = nx.actions.rotate_cw(state)
    state = nx.actions.forward(state)

    # should not be able to walk on/through a key
    check_forward_position(state)

    state = nx.actions.rotate_ccw(state)
    state = nx.actions.forward(state)
    state = nx.actions.rotate_cw(state)
    state = nx.actions.forward(state)

    # should not be able to walk on/through a wall
    check_forward_position(state)

    state = nx.actions.rotate_ccw(state)

    # should be able to walk on/through a goal
    state = nx.actions.forward(state)
    # but not on/through a wall
    check_forward_position(state)


def test_pickup():
    raise NotImplementedError


def test_open():
    raise NotImplementedError


if __name__ == "__main__":
    # test_rotation()
    test_walkable()
