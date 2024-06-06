import jax
import jax.numpy as jnp
import navix as nx


def test_room():
    def f():
        env = nx.environments.Room.create(height=3, width=3, max_steps=8)
        key = jax.random.PRNGKey(4)
        reset = jax.jit(env._reset)
        step = jax.jit(env.step)
        timestep = reset(key)
        # these are optimal actios for navigation + action_cost
        actions = (
            0,  # noop sanity check
            2,  # rotate_ccw
            3,  # forward
            3,  # forward
            2,  # rotate_ccw
            3,  # forward
        )
        print(timestep)
        print()
        for action in actions:
            timestep = step(timestep, jnp.asarray(action))
            print()
            print(nx.actions.DEFAULT_ACTION_SET[action])
            print(timestep)
        return timestep

    f()
    timestep = jax.jit(f)()
    print(timestep)


def test_keydoor():
    def f():
        env = nx.environments.DoorKey.create(height=5, width=10, max_steps=8)
        key = jax.random.PRNGKey(1)
        reset = jax.jit(env._reset)
        step = jax.jit(env.step)
        timestep = reset(key)
        #  these are optimal actions for navigation + action_cost
        actions = (
            0,  # rotate_ccw
            2,  # forward
            2,  # forward
            2,  # forward
            0,  # rotate_ccw
            3,  # pick-up
            0,  # rotate_ccw
            0,  # rotate_ccw
            2,  # forward
            2,  # forward
            1,  # rotate_cw
            2,  # forward
            0,  # rotate_ccw
            5,  # open
            2,  # forward
            2,  # forward
        )
        print(timestep)
        for action in actions:
            timestep = step(timestep, jnp.asarray(action))
            print()
            print(nx.actions.DEFAULT_ACTION_SET[action])
            print(timestep)
        return timestep

    f()
    jax.jit(f)()


def test_keydoor2():
    env = nx.environments.DoorKey.create(5, 7, 100, observation_fn=nx.observations.rgb)

    key = jax.random.PRNGKey(1)
    timestep = env._reset(key)
    return


if __name__ == "__main__":
    # test_room()
    # jax.jit(test_room)()
    test_keydoor()
    # test_keydoor2()
