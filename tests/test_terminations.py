import jax
import jax.numpy as jnp

import navix as nx


def test_termination():
    def f():
        env = nx.environments.Room(height=3, width=3, max_steps=100)
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
        env = nx.environments.Room(height=3, width=3, max_steps=4)
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


if __name__ == "__main__":
    test_termination()
    test_truncation()