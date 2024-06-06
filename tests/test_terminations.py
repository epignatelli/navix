import jax
import jax.numpy as jnp

import navix as nx
from navix.entities import Directions
from navix.states import State
from navix.components import EMPTY_POCKET_ID


def test_on_navigation_completion():
    grid = jnp.zeros((5, 5), dtype=jnp.int32)

    players = nx.entities.Player(
        position=jnp.asarray((1, 1)), direction=Directions.EAST, pocket=EMPTY_POCKET_ID
    )
    goals = nx.entities.Goal.create(position=jnp.asarray((1, 2)), probability=jnp.asarray(1))
    entities = {
        nx.entities.Entities.PLAYER: players[None],
        nx.entities.Entities.GOAL: goals[None],
    }

    state = State(
        key=jax.random.PRNGKey(0),
        grid=grid,
        cache=nx.rendering.cache.RenderingCache.init(grid),
        entities=entities,
    )
    # should not terminate
    termination = nx.terminations.on_goal_reached(state, jnp.asarray(0), state)
    assert not termination, f"Should not terminate, got {termination} instead"

    # move forward
    new_state = nx.actions.forward(state)
    termination = nx.terminations.on_goal_reached(state, jnp.asarray(0), new_state)
    assert termination, f"Should terminate, got {termination} instead"


def test_check_truncation():
    terminated = jnp.asarray(False)
    truncated = jnp.asarray(False)
    assert nx.terminations.check_truncation(terminated, truncated) == jnp.asarray(
        0, dtype=jnp.int32
    )

    terminated = jnp.asarray(True)
    truncated = jnp.asarray(False)
    assert nx.terminations.check_truncation(terminated, truncated) == jnp.asarray(
        2, dtype=jnp.int32
    )

    terminated = jnp.asarray(False)
    truncated = jnp.asarray(True)
    assert nx.terminations.check_truncation(terminated, truncated) == jnp.asarray(
        1, dtype=jnp.int32
    )

    terminated = jnp.asarray(True)
    truncated = jnp.asarray(True)
    assert nx.terminations.check_truncation(terminated, truncated) == jnp.asarray(
        2, dtype=jnp.int32
    )


if __name__ == "__main__":
    test_on_navigation_completion()
    test_check_truncation()
