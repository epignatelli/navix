import jax
import jax.numpy as jnp
import navix as nx

import gymnasium as gym
import minigrid
import random
import time

from timeit import timeit


N_TIMEIT_LOOPS = 5
N_TIMESTEPS = 100
N_SEEDS = 10


def profile_navix(seed):
    env = nx.environments.Room(16, 16, 8)
    key = jax.random.PRNGKey(seed)
    timestep = env.reset(key)
    actions = jax.random.randint(key, (N_TIMESTEPS,), 0, 6)

    for i in range(N_TIMESTEPS):
        timestep = env.step(timestep, actions[i])

    return timestep


def profile_minigrid(seed):
    env = gym.make("MiniGrid-Empty-16x16-v0", render_mode=None)
    observation, info = env.reset(seed=42)
    for _ in range(N_TIMESTEPS):
        action = random.randint(0, 4)
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()
    env.close()
    return observation


if __name__ == "__main__":
    # profile navix
    print("Profiling navix, N_SEEDS = {}, N_TIMESTEPS = {}".format(N_SEEDS, N_TIMESTEPS))
    seeds = jnp.arange(N_SEEDS)

    print("\tCompiling...")
    start = time.time()
    f = jax.jit(jax.vmap(profile_navix)).lower(seeds).compile()
    print("\tCompiled in {:.2f}s".format(time.time() - start))

    print("\tRunning ...")
    res_navix = timeit(lambda: f(seeds).state.grid.block_until_ready(), number=N_TIMEIT_LOOPS)
    print(res_navix)

    # profile minigrid
    print("Profiling minigrid, N_SEEDS = 1, N_TIMESTEPS = {}".format(N_TIMESTEPS))
    res_minigrid = timeit(lambda: profile_minigrid(0), number=N_TIMEIT_LOOPS)
    print(res_minigrid)
