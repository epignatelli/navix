import jax
import jax.numpy as jnp
import navix as nx


def test_room():
    def f():
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
        print()
        for action in actions:
            timestep = step(timestep, jnp.asarray(action))
            print()
            print(nx.actions.ACTIONS[action])
            print(timestep)
        return timestep

    # f()
    jax.jit(f)()


def test_keydoor():
    def f():
        env = nx.environments.KeyDoor(10, 5, 8)
        key = jax.random.PRNGKey(1)
        reset = jax.jit(env.reset)
        step = jax.jit(env.step)
        timestep = reset(key)
        #  these are optimal actions for navigation + action_cost
        actions = (
            2,  # rotate_ccw
            3,  # forward
            3,  # forward
            3,  # forward
            2,  # rotate_ccw
            7,  # pick-up
            2,  # rotate_ccw
            2,  # rotate_ccw
            3,  # forward
            3,  # forward
            1,  # rotate_cw
            3,  # forward
            2,  # rotate_ccw
            8,  # open
            3,  # forward
            3,  # forward
        )
        print(timestep)
        for action in actions:
            timestep = step(timestep, jnp.asarray(action))
            print()
            print(nx.actions.ACTIONS[action])
            print(timestep)
        return timestep

    jax.jit(f)()


if __name__ == "__main__":
    test_room()
    # test_keydoor()
