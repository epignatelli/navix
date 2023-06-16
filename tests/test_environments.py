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
        # these are optimal actions for navigation + action_cost
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
            print(nx.actions.ACTIONS[action])
            print(timestep)
        return timestep

    f()
    timestep = jax.jit(f)()
    print(timestep)


def test_termination():
    def f():
        env = nx.environments.Room(3, 3, 100)
        key = jax.random.PRNGKey(4)
        reset = jax.jit(env.reset)
        step = jax.jit(env.step)
        timestep = reset(key)
        print(timestep)
        print()
        # these are optimal actions for navigation + action_cost
        actions = (
            0,  # noop sanity check
            2,  # rotate_ccw
            3,  # forward
            3,  # forward
            2,  # rotate_ccw
            3,  # forward
        )
        for action in actions:
            timestep = step(timestep, jnp.asarray(action))
            print(nx.actions.ACTIONS[action])
            print(timestep)
            print()
        return timestep

    timestep = f()
    msg = f"Timestep should be terminataed ({jnp.asarray(2)}), got {timestep.step_type} instead"
    assert timestep.step_type == jnp.asarray(2), msg


def test_truncation():
    def f():
        env = nx.environments.Room(3, 3, 4)
        key = jax.random.PRNGKey(4)
        reset = jax.jit(env.reset)
        step = jax.jit(env.step)
        timestep = reset(key)
        # these are optimal actions for navigation + action_cost
        actions = (
            0,  # t_0 noop sanity check
            2,  # t_1 rotate_ccw
            0,  # t_2 noop
            0,  # t_3 noop - should be truncated
        )
        for action in actions:
            timestep = step(timestep, jnp.asarray(action))
        return timestep

    f()
    timestep = jax.jit(f)()
    print(timestep)
    msg = f"Timestep should be truncated ({jnp.asarray(1)}), got {timestep.step_type} instead"
    assert timestep.step_type == jnp.asarray(1), msg


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

    f()
    jax.jit(f)()


if __name__ == "__main__":
    # test_room()
    # test_termination()
    # test_truncation()
    test_keydoor()
