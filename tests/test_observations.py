import jax
import navix as nx


def test_rgb():
    env = nx.environments.Room.create(
        height=10, width=5, max_steps=100, observation_fn=nx.observations.rgb
    )
    key = jax.random.PRNGKey(4)
    timestep = env.reset(key)

    actions = jax.random.randint(key, (100,), 0, 6)
    timestep = jax.lax.scan(lambda c, x: (env.step(c, x), ()), timestep, actions)
    return timestep


if __name__ == "__main__":
    jax.jit(test_rgb)()
