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

NUM_ENVS = 1


def run_minigrid(env_id: str, num_steps: int, num_runs: int):
    print("Running MiniGrid...")

    def _run():
        if NUM_ENVS == 1:
            env = gym.make(env_id, max_episode_steps=num_steps)
        else:
            env = gym.make_vec(env_id, num_envs=NUM_ENVS, wrappers=[ImgObsWrapper])
        env.reset()

        for _ in range(num_steps):
            action = env.action_space.sample()
            timestep = env.step(action)
        return timestep

    times = timeit.repeat(_run, number=1, repeat=num_runs)
    _run()
    print(f"Time taken for one run: {times} seconds")
    return times


def run_navix(env_id: str, num_steps: int, num_runs: int):
    print("Running Navix JIT loop...")

    def _run(key):
        env = nx.make(env_id, max_steps=num_steps)  # Create the environment
        timestep = env.reset(key)
        actions = jax.random.randint(key, (num_steps,), 0, env.action_space.n)

        def body_fun(timestep, action):
            timestep = env.step(timestep, action)  # Update the environment state
            return timestep, ()

        return jax.lax.scan(body_fun, timestep, actions, unroll=20)[0]

    key = jax.random.split(jax.random.PRNGKey(0), num=NUM_ENVS)
    _run = jax.jit(jax.vmap(_run)).lower(key).compile()
    times = timeit.repeat(
        lambda: _run(key).t.block_until_ready(), number=1, repeat=num_runs
    )
    print(f"Time taken for one run: {times} seconds")
    return times


def speedup_by_num_steps():
    print("*" * 80)
    print("Running speedup by num steps")
    print("*" * 80)
    NUM_RUNS = 5
    ENV_ID = "Navix-Empty-8x8-v0"
    gym_env_id = ENV_ID.replace("Navix", "MiniGrid")

    results = {}
    for order in range(1, 7):
        num_steps = 10**order
        print(num_steps)
        results[num_steps] = {
            "minigrid": run_minigrid(gym_env_id, num_steps, NUM_RUNS),
            "navix_jit_loop": run_navix(ENV_ID, num_steps, NUM_RUNS),
        }
        with open(
            os.path.join(os.path.dirname(__file__), "speedup_num_steps.json"), "w"
        ) as f:
            json.dump(results, f, indent=2)


def speedup_by_env():
    print("*" * 80)
    print("Running speedup by env...")
    print("*" * 80)
    NUM_STEPS = 1_000
    NUM_RUNS = 5

    results = {}
    for env_id in nx.registry():
        try:
            gym_env_id = env_id.replace("Navix", "MiniGrid")
            print(env_id)
            results[env_id] = {
                "minigrid": run_minigrid(gym_env_id, NUM_STEPS, NUM_RUNS),
                "navix_jit_loop": run_navix(env_id, NUM_STEPS, NUM_RUNS),
            }
            with open(
                os.path.join(os.path.dirname(__file__), "speedup_env.json"), "w"
            ) as f:
                json.dump(results, f, indent=2)
        except KeyboardInterrupt:
            return
        except Exception as e:
            print(f"Error in {env_id}: {repr(e)}")


def plot_speedup_by_num_steps():
    with open(
        os.path.join(os.path.dirname(__file__), "speedup_num_steps.json"), "r"
    ) as f:
        results = json.load(f)
    minigrid_times = {k: v["minigrid"] for k, v in results.items()}
    navix_times = {k: v["navix_jit_loop"] for k, v in results.items()}
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
    ax.set_xlabel("Number of steps", fontsize=12)
    ax.set_ylabel("Time (s)", fontsize=12)
    ax.set_title("Speed up by number of steps", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.grid(axis="y", linestyle=(0, (6, 8)), alpha=0.6)
    legend = fig.legend(
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.53, -0.2),  # Adjust the y-coordinate to add more white space
        shadow=False,
        frameon=False,
    )
    fig.savefig(
        os.path.join(os.path.dirname(__file__), "speedup_num_steps.png"),
        bbox_extra_artists=(legend,),
        bbox_inches="tight",
    )


def plot_speedup_by_env():
    with open(os.path.join(os.path.dirname(__file__), "speedup_env.json"), "r") as f:
        results = json.load(f)
    minigrid_times = {k: v["minigrid"] for k, v in results.items()}
    navix_times = {k: v["navix_jit_loop"] for k, v in results.items()}
    fig, ax = plt.subplots(figsize=(11, 3), dpi=150)
    xs = range(len(minigrid_times))
    ys_minigrid = jnp.asarray(list(minigrid_times.values()))
    ys_navix = jnp.asarray(list(navix_times.values()))
    ax.bar(
        [x - 0.2 for x in xs],
        ys_minigrid.mean(axis=-1),
        yerr=ys_minigrid.std(axis=-1),
        label="MiniGrid",
        color="black",
        width=0.4,
    )
    ax.bar(
        [x + 0.2 for x in xs],
        ys_navix.mean(axis=-1),
        yerr=ys_navix.std(axis=-1),
        label="NAVIX",
        color="red",
        alpha=0.7,
        width=0.4,
    )
    ax.set_xlabel("Environment", fontsize=12)
    ax.set_ylabel("Time (s)", fontsize=12)
    ax.set_title("Speed up by environment", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.set_xticks(xs)
    # ax.set_yscale("log")
    ax.grid(axis="y", linestyle=(0, (6, 8)), alpha=0.6)
    legend = fig.legend(
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.53, -0.1),
        shadow=False,
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(os.path.dirname(__file__), "speedup_env.png"),
        bbox_extra_artists=(legend,),
        bbox_inches="tight",
    )
    print(ys_navix, ys_minigrid, ys_minigrid / ys_navix)
    mean = jnp.mean(ys_minigrid / ys_navix)
    std = jnp.mean(ys_minigrid / ys_navix, axis=-1).std()
    print(mean, std)


if __name__ == "__main__":
    # speedup_by_num_steps()
    # speedup_by_env()
    plot_speedup_by_num_steps()
    plot_speedup_by_env()
