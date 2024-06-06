import time
from timeit import repeat

import jax
import jax.numpy as jnp
import navix as nx


N_TIMEIT_LOOPS = 10
N_REPEAT = 100
N_TIMESTEPS = 1000
N_SEEDS = 100


def test_observation():
    def test(seed):
        env = nx.environments.Room.create(
            height=10, width=5, max_steps=100, observation_fn=nx.observations.none
        )
        key = jax.random.PRNGKey(seed)
        timestep = env._reset(key)

        actions = jax.random.randint(key, (100,), 0, 6)
        timestep = jax.lax.scan(lambda c, x: (env.step(c, x), ()), timestep, actions)[0]
        return timestep

    # profile navix scanned
    print("Profiling, N_SEEDS = {}, N_TIMESTEPS = {}".format(N_SEEDS, N_TIMESTEPS))

    seeds = jnp.arange(N_SEEDS)

    print(f"\tCompiling {test}...")
    start = time.time()
    test_jit = jax.jit(jax.vmap(test)).lower(seeds).compile()
    print("\tCompiled in {:.2f}s".format(time.time() - start))

    print("\tRunning ...")
    res = repeat(
        lambda: test_jit(seeds).observation.block_until_ready(),
        number=N_TIMEIT_LOOPS,
        repeat=N_REPEAT,
    )
    res = jnp.asarray(res)
    print(f"\t {jnp.mean(res)} ± {jnp.std(res)}")


if __name__ == "__main__":
    test_observation()
