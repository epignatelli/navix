import jax
import jax.numpy as jnp
import navix as nx

import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper
import random
import time

from timeit import timeit


N_TIMEIT_LOOPS = 5
N_TIMESTEPS = 10
N_SEEDS = 10_000


def profile_navix(seed):
    env = nx.make("Navix-Empty-5x5-v0", max_steps=100)
    key = jax.random.PRNGKey(seed)
    timestep = env._reset(key)
    actions = jax.random.randint(key, (N_TIMESTEPS,), 0, 6)

    # for loop
    for i in range(N_TIMESTEPS):
        timestep = env.step(timestep, actions[i])

    return timestep


def profile_minigrid(seed):
    num_envs = N_SEEDS // 1000
    env = gym.vector.make(
        "MiniGrid-Empty-16x16-v0",
        wrappers=ImgObsWrapper,
        num_envs=num_envs,
        render_mode=None,
        asynchronous=True,
    )
    observation, info = env.reset(seed=42)
    for _ in range(N_TIMESTEPS):
        action = random.randint(0, 4)
        timestep = env.step([action] * num_envs)

    env.close()
    return observation


if __name__ == "__main__":
    # profile navix
    print(
        "Profiling navix, N_SEEDS = {}, N_TIMESTEPS = {}".format(N_SEEDS, N_TIMESTEPS)
    )
    seeds = jnp.arange(N_SEEDS)

    print("\tCompiling...")
    start = time.time()
    n_devices = jax.local_device_count()
    seeds = seeds.reshape(n_devices, N_SEEDS // n_devices)
    f = jax.vmap(profile_navix, axis_name="batch")
    f = jax.pmap(f, axis_name="device")
    f = f.lower(seeds).compile()
    print("\tCompiled in {:.2f}s".format(time.time() - start))

    print("\tRunning ...")
    res_navix = timeit(
        lambda: f(seeds).state.grid.block_until_ready(), number=N_TIMEIT_LOOPS
    )
    print(res_navix)

    # profile minigrid
    print(
        "Profiling minigrid, N_SEEDS = {}, N_TIMESTEPS = {}".format(
            N_TIMESTEPS, N_SEEDS // 1000
        )
    )
    res_minigrid = timeit(lambda: profile_minigrid(0), number=N_TIMEIT_LOOPS)
    print(res_minigrid)
