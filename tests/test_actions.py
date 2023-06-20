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


def test_walkable():
    height = 6
    width = 18
    env = nx.environments.KeyDoor(
        height=height,
        width=width,
        max_steps=100,
        observation_fn=nx.observations.categorical,
    )

    key = jax.random.PRNGKey(0)
    timestep = env.reset(key)
    actions = (
        2,
        3, # in front of key after this
    )
    actions_stuck = (
        3, # should not be able to move forward
        3, # should not be able to move forward
        2, # rotate towards the wall
        3, # should not be able to move forward
        3, # should not be able to move forward
    )
    for action in actions:
        timestep = env.step(timestep, jnp.asarray(action))
        print(timestep.state.player.position)

    for action in actions_stuck:
        next_timestep = env.step(timestep, jnp.asarray(action))
        print(timestep.state.player.position)
        assert jnp.array_equal(timestep.state.player.position, next_timestep.state.player.position)
        timestep = next_timestep

if __name__ == "__main__":
    # test_rotation()
    test_walkable()
