import jax
import navix as nx


def test_issue_100():
    env = nx.make(
        "Navix-KeyCorridorS6R3-v0", observation_fn=nx.observations.rgb_first_person
    )
    timestep = env.reset(jax.random.PRNGKey(0))
    timestep = env.step(timestep, jax.numpy.asarray(1))
