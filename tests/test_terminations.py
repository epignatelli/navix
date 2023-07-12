import jax
import jax.numpy as jnp

import navix as nx
from navix.components import EMPTY_POCKET_ID


def test_on_navigation_completion():
    grid = jnp.zeros((5, 5), dtype=jnp.int32)

    players = nx.entities.Player(
        position=jnp.asarray((1, 1)), direction=jnp.asarray(0), pocket=EMPTY_POCKET_ID
    )
    goals = nx.entities.Goal(position=jnp.asarray((3, 3)), probability=jnp.asarray(1))
    entities = {
        nx.entities.Entities.PLAYER.value: players[None],
        nx.entities.Entities.GOAL.value: goals[None],
    }

    state = nx.entities.State(
        key=jax.random.PRNGKey(0),
        grid=grid,
        cache=nx.entities.RenderingCache.init(grid),
        entities=entities,
    )
    # shpuld not terminate
    termination = nx.terminations.on_navigation_completion(state, jnp.asarray(0), state)
    assert not termination, f"Should not terminate, got {termination} instead"

    # artificially put agent on goal
    player = state.get_player()
    goals = state.get_goals()
    new_state = state.set_player(player.replace(position=goals.position[0]))
    termination = nx.terminations.on_navigation_completion(
        state, jnp.asarray(0), new_state
    )
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
