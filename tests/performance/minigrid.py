import jax
import jax.numpy as jnp
import navix as nx

import gymnasium
import random
import time

from timeit import repeat

N_TIMEIT_LOOPS = 3
N_REPEAT = 5
N_TIMESTEPS = 1000
N_SEEDS = 10_000


def profile_navix_scan(seed):
    env = nx.environments.Room.create(
        height=10, width=5, max_steps=100, observation_fn=nx.observations.categorical
    )
    key = jax.random.PRNGKey(4)
    timestep = env._reset(key)
    actions = jax.random.randint(key, (N_TIMESTEPS,), 0, 6)

    timestep = jax.lax.scan(
        lambda carry, x: (env.step(carry, x), ()), timestep, actions
    )[0]

    return timestep


def profile_minigrid(seed):
    env = gymnasium.make("MiniGrid-Empty-16x16-v0", render_mode=None)
    observation, info = env.reset(seed=42)
    for _ in range(N_TIMESTEPS):
        action = random.randint(0, 4)
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()
    env.close()
    return observation


if __name__ == "__main__":
    # profile navix scanned
    print(
        "Profiling navix with `scan`, N_SEEDS = {}, N_TIMESTEPS = {}".format(
            N_SEEDS, N_TIMESTEPS
        )
    )
    seeds = jnp.arange(N_SEEDS)

    print(f"\tCompiling {profile_navix_scan}...")
    start = time.time()
    f_scan = jax.jit(jax.vmap(profile_navix_scan)).lower(seeds).compile()
    print("\tCompiled in {:.2f}s".format(time.time() - start))

    print("\tRunning ...")
    res_navix = repeat(
        lambda: f_scan(seeds).observation.block_until_ready(),
        number=N_TIMEIT_LOOPS,
        repeat=N_REPEAT,
    )
    res_navix = jnp.asarray(res_navix)
    print(f"\t {jnp.mean(res_navix)} ± {jnp.std(res_navix)}")

    # profile minigrid
    print("Profiling minigrid, N_SEEDS = 1, N_TIMESTEPS = {}".format(N_TIMESTEPS))
    res_minigrid = repeat(
        lambda: profile_minigrid(0), number=N_TIMEIT_LOOPS, repeat=N_REPEAT
    )
    res_minigrid = jnp.asarray(res_minigrid)
    print(f"\t {jnp.mean(res_minigrid)} ± {jnp.std(res_minigrid)}")
