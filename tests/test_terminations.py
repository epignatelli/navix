import jax
import jax.numpy as jnp

import navix as nx


def test_on_navigation_completion():
    grid = jnp.zeros((5, 5), dtype=jnp.int32)
    state = nx.entities.State(
        key=jax.random.PRNGKey(0),
        grid=grid,
        cache=nx.entities.RenderingCache.init(grid),
        players=nx.entities.Player.create(jnp.asarray((1, 1)), jnp.asarray(0)),
        goals=nx.entities.Goal.create(jnp.asarray((3, 3)), jnp.asarray(1)),
    )
    # shpuld not terminate
    termination = nx.terminations.on_navigation_completion(state, jnp.asarray(0), state)
    assert not termination, f"Should not terminate, got {termination} instead"

    # artificially put agent on goal
    new_state = state.replace(players=state.players.replace(position=state.goals.position))
    termination = nx.terminations.on_navigation_completion(state, jnp.asarray(0), new_state)
    assert termination, f"Should terminate, got {termination} instead"


def test_check_truncation():
    terminated = jnp.asarray(False)
    truncated = jnp.asarray(False)
    assert nx.terminations.check_truncation(terminated, truncated) == jnp.asarray(0, dtype=jnp.int32)

    terminated = jnp.asarray(True)
    truncated = jnp.asarray(False)
    assert nx.terminations.check_truncation(terminated, truncated) == jnp.asarray(2, dtype=jnp.int32)

    terminated = jnp.asarray(False)
    truncated = jnp.asarray(True)
    assert nx.terminations.check_truncation(terminated, truncated) == jnp.asarray(1, dtype=jnp.int32)

    terminated = jnp.asarray(True)
    truncated = jnp.asarray(True)
    assert nx.terminations.check_truncation(terminated, truncated) == jnp.asarray(2, dtype=jnp.int32)


if __name__ == "__main__":
    test_on_navigation_completion()
    test_check_truncation()
