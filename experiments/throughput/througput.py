import os
import sys
import time
import timeit
import json
import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
from navix import observations
from navix.agents.models import ActorCritic
from navix.agents.ppo import PPO, PPOHparams
import numpy as np
from navix.environments.environment import Environment
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
        return timestep  # type: ignore

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


def benchmark_throughput_ppo():
    print("Running throughput benchmark for PPO...")
    ENV_ID = "Navix-Empty-Random-5x5-v0"

    def FlattenObsWrapper(env: Environment):
        flatten_obs_fn = lambda x: jnp.ravel(env.observation_fn(x))
        flatten_obs_shape = (int(np.prod(env.observation_space.shape)),)
        return env.replace(
            observation_fn=flatten_obs_fn,
            observation_space=env.observation_space.replace(shape=flatten_obs_shape),
        )

    class HiddenPrints:
        def __enter__(self):
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")

        def __exit__(self, exc_type, exc_val, exc_tb):
            sys.stdout.close()
            sys.stdout = self._original_stdout

    def save(results):
        with open(
            os.path.join(os.path.dirname(__file__), "throughput_ppo.json"), "w"
        ) as f:
            json.dump(results, f)

    def run(n_agents):
        env = nx.make(
            ENV_ID,
            observation_fn=observations.symbolic_first_person,
        )
        env = FlattenObsWrapper(env)

        agent = PPO(
            hparams=PPOHparams(),
            network=ActorCritic(
                action_dim=len(env.action_set),
            ),
            env=env,
        )

        experiment = nx.Experiment(
            name="throughput_ppo",
            agent=agent,
            env=env,
            env_id=ENV_ID,
            seeds=tuple(range(n_agents)),
        )
        with HiddenPrints():
            train_state, _ = experiment.run(do_log=False)
        return train_state

    results = {}
    n_agents = 1
    while True:
        try:
            print(f"Running with {n_agents} agents...")
            times = timeit.repeat(lambda: run(n_agents), number=1, repeat=5)
            results[n_agents] = times
            print(f"Time taken for {n_agents} agents: {times} seconds")
            n_agents *= 2
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
    fig, ax = plt.subplots(figsize=(6, 3), dpi=150)
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

    fig, ax = plt.subplots(figsize=(6, 3), dpi=150)
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


def compute_stats(values):
    values = jnp.asarray(values)
    mean = jnp.mean(values)
    std = jnp.std(values)
    return f"{mean:.3f} ± {std:.3f}"


def table_throughput():
    with open("throughput_envs.json", "r") as f:
        data = json.load(f)

    # Collect all env counts across all backends
    all_env_counts = sorted(
        {int(k) for backend in data.values() for k in backend.keys()}
    )
    env_cols = [str(k) for k in all_env_counts]

    rows = []

    for backend in reversed(sorted(data.keys())):
        row = [backend]
        for env in env_cols:
            if env in data[backend]:
                row.append(compute_stats(data[backend][env]))
            else:
                row.append("-")
        rows.append(row)

    # Header row and formatting
    header = ["Backend"] + env_cols
    separator = ["---"] * len(header)

    def format_row(row):
        return "| " + " | ".join(row) + " |"

    table = [format_row(header), format_row(separator)]
    table += [format_row(row) for row in rows]

    table = "\n".join(table)
    print(table)

    with open("throughput_envs.md", "w") as f:
        f.write(table)


def table_throughput_ppo():
    with open("throughput_ppo.json", "r") as f:
        data = json.load(f)

    # Sort keys numerically
    env_counts = sorted(int(k) for k in data.keys())
    env_cols = [str(k) for k in env_counts]

    header = ["Throughpu/Agents"] + env_cols
    separator = ["---"] * len(header)

    # Only one row: NAVIX throughput
    row = ["NAVIX"]
    for key in env_cols:
        row.append(compute_stats(data[key]))

    def format_row(row):
        return "| " + " | ".join(row) + " |"

    table = [format_row(header), format_row(separator), format_row(row)]

    table = "\n".join(table)
    print(table)

    with open("throughput_ppo.md", "w") as f:
        f.write(table)


if __name__ == "__main__":
    # benchmark_throughput()
    # benchmark_throughput_ppo()
    # plot_throughput()
    # plot_throughput_ppo()
    table_throughput()
    table_throughput_ppo()
