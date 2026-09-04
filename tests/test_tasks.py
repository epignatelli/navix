import jax
import jax.numpy as jnp
import pytest

import navix as nx
from navix.states import State
from navix.entities import Entities, Player, Goal, Key, Door
from navix.components import EMPTY_POCKET_ID
from navix.rendering.registry import PALETTE


def test_navigation():
    """Unittest for https://github.com/epignatelli/navix/pull/47"""
    height = 10
    width = 10
    grid = jnp.zeros((height - 2, width - 2), dtype=jnp.int32)
    grid = jnp.pad(grid, 1, mode="constant", constant_values=-1)

    players = Player(
        position=jnp.asarray((1, 1)), direction=jnp.asarray(0), pocket=EMPTY_POCKET_ID
    )
    goals = Goal.create(
        position=jnp.asarray([(1, 1), (1, 1)]), probability=jnp.asarray([0.0, 0.0])
    )
    keys = Key.create(position=jnp.asarray((2, 2)), id=jnp.asarray(0), colour=PALETTE.YELLOW)
    doors = Door.create(
        position=jnp.asarray([(1, 5), (1, 6)]),
        requires=jnp.asarray((0, 0)),
        open=jnp.asarray((False, True)),
        colour=PALETTE.YELLOW,
    )

    entities = {
        Entities.PLAYER: players[None],
        Entities.GOAL: goals,
        Entities.KEY: keys[None],
        Entities.DOOR: doors,
    }

    state = State(
        key=jax.random.PRNGKey(0),
        grid=grid,
        cache=nx.rendering.cache.RenderingCache.init(grid),
        entities=entities,
    )
    action = jnp.asarray(0)
    reward = nx.rewards.on_goal_reached(state, action, state)
    assert jnp.array_equal(reward, jnp.asarray(0.0))


def test_tasks_composition():
    reward_fn = nx.rewards.compose(
        nx.rewards.on_goal_reached,
        nx.rewards.time_cost,
        nx.rewards.wall_hit_cost,
    )

    env = nx.environments.Room.create(height=3, width=3, max_steps=8, reward_fn=reward_fn)
    key = jax.random.PRNGKey(0)

    def _test():
        timestep = env._reset(key)
        for _ in range(10):
            timestep = env.step(timestep, jax.random.randint(key, (), 0, 7))
        return timestep

    print(jax.jit(_test)())


def test_default_task_is_unshaped_on_goal_reached():
    # https://github.com/epignatelli/navix/issues/211 - DEFAULT_TASK no
    # longer folds in a per-step cost; it is exactly `on_goal_reached`.
    assert nx.rewards.DEFAULT_TASK is nx.rewards.on_goal_reached

    env = nx.make("Navix-Empty-5x5-v0", reward_fn=nx.rewards.DEFAULT_TASK)
    ts = env.reset(jax.random.PRNGKey(0))
    # first (non-goal) step: no shaping, reward is exactly 0
    ts = env.step(ts, jnp.asarray(0))  # rotate - definitely not on the goal
    assert float(ts.reward) == 0.0


def test_multiroom_keeps_its_per_step_cost():
    # MultiRoom rode on the old shaped DEFAULT_TASK; it now pins its own
    # `compose(on_goal_reached, time_cost)` so its reward is unchanged.
    env = nx.make("Navix-MultiRoom-N2-S4-v0")
    assert env.reward_fn is not nx.rewards.DEFAULT_TASK
    ts = env.reset(jax.random.PRNGKey(0))
    ts = env.step(ts, jnp.asarray(0))  # non-goal step
    assert abs(float(ts.reward) - (-0.01)) < 1e-6


def test_action_cost_is_a_deprecated_alias_of_time_cost():
    # https://github.com/epignatelli/navix/issues/211 - the index-6
    # `done` exemption was action-set-dependent; every action now costs
    # `cost`, i.e. action_cost == time_cost.
    with pytest.warns(DeprecationWarning, match="time_cost"):
        a = nx.rewards.action_cost(None, jnp.asarray(6), None)
    t = nx.rewards.time_cost(None, jnp.asarray(6), None)
    assert float(a) == float(t)
    assert abs(float(a) - (-0.01)) < 1e-6


if __name__ == "__main__":
    # test_tasks_composition()
    test_navigation()
