import jax
import jax.numpy as jnp

import navix as nx


def test_rotation():
    direction = jnp.asarray(0)

    state = nx.components.State(
        jax.random.PRNGKey(0),
        grid=jnp.zeros((3, 3), dtype=jnp.int32),
        player=nx.components.Player(position=jnp.asarray((1, 1)), direction=direction),
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


if __name__ == "__main__":
    test_rotation()
