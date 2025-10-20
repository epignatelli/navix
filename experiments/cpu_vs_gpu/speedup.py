import os
import timeit
import json
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt

import navix as nx


jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 4)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1)

NUM_ENVS = 1


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

    # warm up
    _run(key).t.block_until_ready()
    print("Warm up done, running the benchmark...")

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

    results = {}
    for order in range(1, 7):
        num_steps = 10**order
        print(num_steps)
        with jax.default_device(jax.devices("cpu")[0]):
            cpu_times = run_navix(ENV_ID, num_steps, NUM_RUNS)
        with jax.default_device(jax.devices("gpu")[0]):
            gpu_times = run_navix(ENV_ID, num_steps, NUM_RUNS)
        results[num_steps] = {
            "CPU": cpu_times,
            "GPU": gpu_times,
        }

        with open(
            os.path.join(
                os.path.dirname(__file__), "speedup_num_steps_cpu_vs_gpu.json"
            ),
            "w",
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
            print(env_id)
            with jax.default_device(jax.devices("cpu")[0]):
                cpu_times = run_navix(env_id, NUM_STEPS, NUM_RUNS)
            with jax.default_device(jax.devices("gpu")[0]):
                gpu_times = run_navix(env_id, NUM_STEPS, NUM_RUNS)
            results[env_id] = {
                "CPU": cpu_times,
                "GPU": gpu_times,
            }
            with open(
                os.path.join(os.path.dirname(__file__), "speedup_env_cpu_vs_gpu.json"),
                "w",
            ) as f:
                json.dump(results, f, indent=2)
        except KeyboardInterrupt:
            return
        except Exception as e:
            print(f"Error in {env_id}: {repr(e)}")


def plot_speedup_by_num_steps():
    with open(
        os.path.join(os.path.dirname(__file__), "speedup_num_steps_cpu_vs_gpu.json"),
        "r",
    ) as f:
        results = json.load(f)

    minigrid_times = {k: v["CPU"] for k, v in results.items()}
    navix_times = {k: v["GPU"] for k, v in results.items()}

    # Standardized figure size
    fig, ax = plt.subplots(figsize=(5, 3), dpi=150)

    xs_minigrid = [int(x) for x in minigrid_times.keys()]
    ys_minigrid = jnp.asarray(list(minigrid_times.values()))

    ax.errorbar(
        xs_minigrid,
        ys_minigrid.mean(axis=-1),
        label="CPU",
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
        label="GPU",
        color="red",
        marker="s",
    )

    # Ensure consistent font sizes
    font_size = 12
    ax.set_xlabel("Number of steps", fontsize=font_size)
    ax.set_ylabel("Time (s)", fontsize=font_size)
    ax.set_title("Speed up by number of steps", fontsize=font_size + 2)
    ax.tick_params(axis="both", which="major", labelsize=font_size - 2)

    # Ensure consistent log scale if needed
    ax.set_yscale("log")
    ax.set_xscale("log")

    ax.grid(axis="y", linestyle=(0, (6, 8)), alpha=0.6)

    legend = fig.legend(
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.2),  # Center-aligned
        shadow=False,
        frameon=False,
        fontsize=font_size - 2,  # Standardized font size
    )

    fig.savefig(
        os.path.join(os.path.dirname(__file__), "speedup_num_steps_cpu_vs_gpu.png"),
        bbox_extra_artists=(legend,),
        bbox_inches="tight",
    )


def plot_speedup_by_env():
    with open(
        os.path.join(os.path.dirname(__file__), "speedup_env_cpu_vs_cpu.json"), "r"
    ) as f:
        results = json.load(f)

    minigrid_times = {k: v["CPU"] for k, v in results.items()}
    navix_times = {k: v["GPU"] for k, v in results.items()}

    # Standardized figure size
    fig, ax = plt.subplots(figsize=(7, 3), dpi=150)

    xs = range(len(minigrid_times))
    ys_minigrid = jnp.asarray(list(minigrid_times.values()))
    ys_navix = jnp.asarray(list(navix_times.values()))

    ax.bar(
        [x - 0.2 for x in xs],
        ys_minigrid.mean(axis=-1),
        yerr=ys_minigrid.std(axis=-1),
        label="CPU",
        color="black",
        width=0.4,
    )

    ax.bar(
        [x + 0.2 for x in xs],
        ys_navix.mean(axis=-1),
        yerr=ys_navix.std(axis=-1),
        label="GPU",
        color="red",
        alpha=0.7,
        width=0.4,
    )

    # Ensure consistent font sizes
    font_size = 12
    ax.set_xlabel("Environment", fontsize=font_size)
    ax.set_ylabel("Time (s)", fontsize=font_size)
    ax.set_title("Speed up by environment", fontsize=font_size + 2)
    ax.tick_params(axis="both", which="major", labelsize=font_size - 2)

    # Make sure y-axis scaling matches the first plot
    ax.set_yscale("log")  # This a y-axis comparable to the previous plot

    ax.grid(axis="y", linestyle=(0, (6, 8)), alpha=0.6)

    legend = fig.legend(
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.2),  # Center-aligned
        shadow=False,
        frameon=False,
        fontsize=font_size - 2,  # Standardized font size
    )

    fig.savefig(
        os.path.join(os.path.dirname(__file__), "speedup_env_cpu_vs_gpu.png"),
        bbox_extra_artists=(legend,),
        bbox_inches="tight",
    )


def compute_stats(values):
    """Compute mean ± std string."""
    values = jnp.asarray(values)
    mean = jnp.mean(values)
    std = jnp.std(values)
    return mean, std, f"{mean:.5f} ± {std:.5f}"


def table_speedup_by_env():
    """Markdown table for speedup by environment.
    It has as many columnts as all the environments in the results file,
    and two rows: one for MiniGrid and one for Navix.
    Saves the table as a string."""
    with open("speedup_env_cpu_vs_gpu.json", "r") as f:
        data = json.load(f)

    envs = list(data.keys())
    navix_row = ["GPU"]
    minigrid_row = ["CPU"]
    speedup_row = ["Speedup"]

    for env in envs:
        minigrid_mean, minigrid_std, minigrid_stats = compute_stats(data[env]["CPU"])
        navix_mean, navix_std, navix_stats = compute_stats(data[env]["GPU"])

        navix_row.append(navix_stats)
        minigrid_row.append(minigrid_stats)

        speedup = minigrid_mean / navix_mean
        rel_err = jnp.sqrt(
            (minigrid_std / minigrid_mean) ** 2 + (navix_std / navix_mean) ** 2
        )
        speedup_std = speedup * rel_err

        speedup_row.append(f"{speedup:.1f} ± {speedup_std:.1f}×")

    # Build header
    header = ["Model"] + envs
    separator = ["---"] * len(header)

    def format_row(row):
        return "| " + " | ".join(row) + " |"

    table = [
        format_row(header),
        format_row(separator),
        format_row(navix_row),
        format_row(minigrid_row),
        format_row(speedup_row),
    ]

    table = "\n".join(table)
    print(table)
    with open("speedup_env_cpu_vs_gpu.md", "w") as f:
        f.write(table)


def table_speedup_by_num_steps():
    """Markdown table for speedup by number of steps.
    It has as many columns as the number of steps in the results file,
    and two rows: one for MiniGrid and one for Navix.
    Saves the table as a string."""
    with open("speedup_num_steps_cpu_vs_gpu.json", "r") as f:
        data = json.load(f)

    steps = list(data.keys())
    navix_row = ["GPU"]
    minigrid_row = ["CPU"]
    speedup_row = ["Speedup"]

    for step in steps:
        minigrid_mean, minigrid_std, minigrid_stats = compute_stats(data[step]["CPU"])
        navix_mean, navix_std, navix_stats = compute_stats(data[step]["GPU"])

        navix_row.append(navix_stats)
        minigrid_row.append(minigrid_stats)

        speedup = minigrid_mean / navix_mean
        rel_err = jnp.sqrt(
            (minigrid_std / minigrid_mean) ** 2 + (navix_std / navix_mean) ** 2
        )
        speedup_std = speedup * rel_err

        speedup_row.append(f"{speedup:.1f} ± {speedup_std:.1f}×")

    # Build header
    header = ["Steps"] + steps
    separator = ["---"] * len(header)

    def format_row(row):
        return "| " + " | ".join(row) + " |"

    table = [
        format_row(header),
        format_row(separator),
        format_row(navix_row),
        format_row(minigrid_row),
        format_row(speedup_row),
    ]

    table = "\n".join(table)
    print(table)
    with open("speedup_num_steps_cpu_vs_gpu.md", "w") as f:
        f.write(table)


if __name__ == "__main__":
    speedup_by_num_steps()
    plot_speedup_by_num_steps()
    table_speedup_by_num_steps()
    
    speedup_by_env()
    plot_speedup_by_env()
    table_speedup_by_env()
