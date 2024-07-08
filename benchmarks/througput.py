import os
import time
import timeit
import json
import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import navix as nx


jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 4)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1)


def run_minigrid(env_id: str, num_envs: int, num_steps: int, num_runs: int):
    print("Running MiniGrid...")

    def _run():
        env = gym.make_vec(
            env_id,
            num_envs=num_envs,
            # max_episode_steps=num_steps,
            wrappers=[ImgObsWrapper],
        )
        env.reset()

        for _ in range(num_steps):
            action = env.action_space.sample()
            timestep = env.step(action)
        return timestep

    times = timeit.repeat(_run, number=1, repeat=num_runs)
    print(f"Time taken for one run: {times} seconds")
    return times


def run_navix_jit_loop(env_id: str, num_envs: int, num_steps: int, num_runs: int):
    print("Running Navix JIT loop...")

    def _run(key):
        env = nx.make(env_id, max_steps=num_steps)  # Create the environment
        timestep = env.reset(key)
        actions = jax.random.randint(key, (num_steps,), 0, env.action_space.n)

        def body_fun(state, _):
            timestep, i = state
            timestep = env.step(timestep, actions[i])  # Update the environment state
            return (timestep, i + 1), ()

        return jax.lax.scan(body_fun, (timestep, 0), length=num_steps, unroll=20)[0][0]

    key = jax.random.split(jax.random.PRNGKey(0), num_envs)
    _run = jax.jit(jax.vmap(_run)).lower(key).compile()
    times = timeit.repeat(
        lambda: _run(key).t.block_until_ready(), number=1, repeat=num_runs
    )
    print(f"Time taken for one run: {times} seconds")
    return times


def benchmark_throughput():
    navix_env_id = "Navix-Empty-8x8-v0"
    minigrid_env_id = "MiniGrid-Empty-8x8-v0"
    num_steps = 1_000
    num_runs = 5

    def save(results):
        with open(
            os.path.join(os.path.dirname(__file__), "throughput_envs.json"), "w"
        ) as f:
            json.dump(results, f)

    results = {"MiniGrid": {}, "NAVIX": {}}
    i = 1
    while True:
        try:
            num_envs = 2**i
            print(f"Number of environments: {num_envs}")
            results["MiniGrid"][num_envs] = run_minigrid(
                minigrid_env_id, num_envs, num_steps, num_runs
            )
            i += 1
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print("Max number of environments reached.")
            print(repr(e))
            break
        save(results)

    i = 1
    while True:
        try:
            num_envs = 2**i
            print(f"Number of environments: {num_envs}")
            results["NAVIX"][num_envs] = run_navix_jit_loop(
                navix_env_id, num_envs, num_steps, num_runs
            )
            i += 1
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print("Max number of environments reached.")
            print(repr(e))
            break
        save(results)


def plot_throughput():
    with open(
        os.path.join(os.path.dirname(__file__), "throughput_envs.json"), "r"
    ) as f:
        results = json.load(f)

    minigrid_times = results["MiniGrid"]
    navix_times = results["NAVIX"]
    fig, ax = plt.subplots(figsize=(11, 3), dpi=150)
    xs_minigrid = [int(x) for x in minigrid_times.keys()]
    ys_minigrid = jnp.asarray(list(minigrid_times.values()))
    print(ys_minigrid)
    ax.errorbar(
        xs_minigrid,
        ys_minigrid.mean(axis=-1),
        label="MiniGrid",
        yerr=ys_minigrid.std(axis=-1),
        color="black",
        marker="o",
    )
    xs_navix = [int(x) for x in navix_times.keys()]
    ys_navix = jnp.asarray(list(navix_times.values()))
    ax.errorbar(
        xs_navix,
        ys_navix.mean(axis=-1),
        yerr=ys_navix.std(axis=-1),
        label="NAVIX",
        color="red",
        marker="s",
    )
    ax.set_title("Batch mode throughput", fontsize=14)
    ax.set_xlabel("Number of environments", fontsize=12)
    ax.set_ylabel("Time (s)", fontsize=12)
    ax.set_yscale("log")
    ax.set_xscale("log", base=2)
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.grid(axis="y", linestyle=(0, (6, 8)), alpha=0.6)
    legend = fig.legend(
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.53, -0.2),  # Adjust the y-coordinate to add more white space
        shadow=False,
        frameon=False,
    )
    fig.savefig(
        os.path.join(os.path.dirname(__file__), "throughput_envs.png"),
        bbox_extra_artists=(legend,),
        bbox_inches="tight",
    )


def plot_throughput_ppo():
    with open(os.path.join(os.path.dirname(__file__), "throughput_ppo.json"), "r") as f:
        navix_times = json.load(f)

    fig, ax = plt.subplots(figsize=(11, 3), dpi=150)
    xs_navix = [int(x) for x in navix_times.keys()]
    ys_navix = jnp.asarray(list(navix_times.values()))
    ax.errorbar(
        xs_navix,
        ys_navix.mean(axis=-1),
        yerr=ys_navix.std(axis=-1),
        label="NAVIX",
        color="red",
        marker="s",
    )
    ax.hlines(
        248.0,
        0,
        4096,
        colors=["black"],
        linestyles=(0, (5, 5)),  # type: ignore
        linewidth=1,
        label="MiniGrid",
    )
    ax.set_title("Training throughput (PPO)", fontsize=14)
    ax.set_xlabel("Number of agents [#]", fontsize=12)
    ax.set_ylabel("Time (s)", fontsize=12)
    ax.set_yscale("log")
    ax.set_xscale("log", base=2)
    ax.set_xlim(0, 4096)
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.grid(axis="y", linestyle=(0, (6, 8)), alpha=0.6)
    legend = fig.legend(
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.53, -0.2),  # Adjust the y-coordinate to add more white space
        shadow=False,
        frameon=False,
    )
    fig.savefig(
        os.path.join(os.path.dirname(__file__), "throughput_ppo.png"),
        bbox_extra_artists=(legend,),
        bbox_inches="tight",
    )


if __name__ == "__main__":
    # benchmark_throughput()
    plot_throughput()
    plot_throughput_ppo()
