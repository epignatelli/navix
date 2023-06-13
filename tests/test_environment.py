import jax
import jax.numpy as jnp
import navix as nx


def test_env():
    env = nx.environments.Room(3, 3, 8)
    key = jax.random.PRNGKey(4)
    timestep = env.reset(key)
    actions = (
        0,  # noop sanity check
        1,  # rotate_cw
        2,  # rotate_ccw
        3,  # forward
        5,  # backward
        4,  # right
        6,  # left
        0,  # noop
        0,  # noop
        0,  # noop
        0,  # noop
    )
    print(timestep)
    for action in actions:
        timestep = env.step(timestep, jnp.asarray(action))
        print()
        print(nx.actions.ACTIONS[action])
        print(timestep)


def test_jit_env():
    env = nx.environments.Room(3, 3, 8)
    key = jax.random.PRNGKey(4)
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    timestep = reset(key)
    actions = (
        0,  # noop sanity check
        1,  # rotate_cw
        2,  # rotate_ccw
        3,  # forward
        5,  # backward
        4,  # right
        6,  # left
        0,  # noop
        0,  # noop
        0,  # noop
        0,  # noop
    )
    print(timestep)
    for action in actions:
        timestep = step(timestep, jnp.asarray(action))
        print()
        print(nx.actions.ACTIONS[action])
        print(timestep)


if __name__ == "__main__":
    test_jit_env()
