import time
from timeit import repeat

import jax
import jax.numpy as jnp
import navix as nx


N_TIMEIT_LOOPS = 10
N_REPEAT = 30
N_TIMESTEPS = 1000
N_SEEDS = 10


def test_observation(observation_fn):
    def test(seed):
        env = nx.environments.KeyDoor(
            height=5, width=10, max_steps=100, gamma=1.0, observation_fn=observation_fn
        )
        key = jax.random.PRNGKey(seed)
        timestep = env.reset(key)

        actions = jax.random.randint(key, (100,), 0, 6)
        timestep = jax.lax.scan(lambda c, x: (env.step(c, x), ()), timestep, actions)[0]
        return timestep

    # profile navix scanned
    print(
        "Profiling observation {}, N_SEEDS = {}, N_TIMESTEPS = {}".format(
            observation_fn, N_SEEDS, N_TIMESTEPS
        )
    )

    seeds = jnp.arange(N_SEEDS)

    print(f"\tCompiling {observation_fn}...")
    start = time.time()
    test_jit = jax.jit(jax.vmap(test)).lower(seeds).compile()
    print("\tCompiled in {:.2f}s".format(time.time() - start))

    print(f"\tRunning {observation_fn}...")
    res = repeat(
        lambda: test_jit(seeds).observation.block_until_ready(),
        number=N_TIMEIT_LOOPS,
        repeat=N_REPEAT,
    )
    res = jnp.asarray(res)
    print(f"\t {jnp.mean(res)} ± {jnp.std(res)}")


if __name__ == "__main__":
    test_observation(nx.observations.none)
    test_observation(nx.observations.categorical)
    test_observation(nx.observations.rgb)
    test_observation(nx.observations.categorical_first_person)
    test_observation(nx.observations.rgb_first_person)
