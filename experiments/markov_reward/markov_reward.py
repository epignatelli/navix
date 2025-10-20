from dataclasses import dataclass, field
import os
import json

import tyro
import numpy as np
import matplotlib.pyplot as plt
import jax

# set persistent compilation cache directory
jax.config.update("jax_compilation_cache_dir", "/tmp/jax-cache/")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
import jax.numpy as jnp

import navix as nx
from navix import observations, rewards
from navix.agents import PPO, PPOHparams, ActorCritic
from navix.environments.environment import Environment

# set persistent compilation cache directory
jax.config.update("jax_compilation_cache_dir", "/tmp/jax-cache/")


@dataclass
class Args:
    project_name = "navix-markov-reward"
    seeds_offset: int = 0
    n_seeds: int = 32
    ppo_config: PPOHparams = field(default_factory=PPOHparams)


def FlattenObsWrapper(env: Environment):
    flatten_obs_fn = lambda x: jnp.ravel(env.observation_fn(x))
    flatten_obs_shape = (int(np.prod(env.observation_space.shape)),)
    return env.replace(
        observation_fn=flatten_obs_fn,
        observation_space=env.observation_space.replace(shape=flatten_obs_shape),
    )


def train(env_id: str, args: Args, markov: bool = False):
    on_goal_reached = rewards.compose(rewards.on_goal_reached, rewards.action_cost)
    on_door_done = rewards.compose(rewards.on_door_done, rewards.action_cost)
    original_reward_fns = {
        "Navix-Empty-6x6-v0": on_goal_reached,
        "Navix-Empty-Random-6x6-v0": on_goal_reached,
        "Navix-Empty-16x16-v0": on_goal_reached,
        "Navix-Dynamic-Obstacles-6x6-Random-v0": on_goal_reached,
        "Navix-DoorKey-5x5-v0": on_goal_reached,
        "Navix-DoorKey-16x16-v0": on_goal_reached,
        "Navix-DistShift2-v0": on_goal_reached,
        "Navix-GoToDoor-6x6-v0": on_door_done,
        "Navix-GoToDoor-8x8-v0": on_goal_reached,
        "Navix-FourRooms-v0": on_goal_reached,
        "Navix-LavaGapS6-v0": on_goal_reached,
        "Navix-LavaGapS7-v0": on_goal_reached,
    }

    def non_markov_reward_fn(prev_state, action, state, timestep):
        reward = original_reward_fns[env_id](prev_state, action, state, timestep)
        return jax.lax.cond(
            reward > 0,
            lambda reward: reward - 0.9 * (timestep.t / env.max_steps),
            lambda reward: reward,
            reward,
        )

    env = nx.make(
        env_id,
        observation_fn=observations.symbolic,
        reward_fn=original_reward_fns[env_id] if markov else non_markov_reward_fn,
    )
    env = FlattenObsWrapper(env)
    agent = PPO(
        hparams=args.ppo_config,
        network=ActorCritic(
            action_dim=len(env.action_set),
        ),
        env=env,
    )

    experiment = nx.Experiment(
        name=args.project_name,
        agent=agent,
        env=env,
        env_id=env_id,
        seeds=tuple(range(args.seeds_offset, args.seeds_offset + args.n_seeds)),
    )
    train_state, logs = experiment.run(do_log=False)

    assert "returns" in logs, "Returns not found in logs"
    assert "done_mask" in logs, "Done mask not found in logs"
    is_terminal = jnp.asarray(
        logs.pop("done_mask"), dtype=jnp.bool
    )  # (Seeds, Iters, Time, Envs)
    returns = logs.pop("returns")  # (Seeds, Iters, Time, Envs)
    # success_mask = is_terminal * (returns > 0)  # (Seeds, Iters, Time, Envs)
    # success = jnp.sum(success_mask, axis=(-1, -2))  # (Seeds, Iters)
    # non_success = jnp.sum(jnp.logical_not(success_mask), axis=(-1, -2))  # (Seeds, Iters)
    # success_rate = success / (success + non_success)
    successes = jnp.sum(is_terminal * (returns > 0), axis=(-1, -2))  # (Seeds, Iters)
    non_successes = jnp.sum(
        is_terminal * (returns <= 0), axis=(-1, -2)
    )  # (Seeds, Iters)
    success_rate = successes / (successes + non_successes)
    return success_rate


def plot():
    with open("markov_reward.json", "r") as f:
        results = json.load(f)

    fig, ax = plt.subplots(4, 3, figsize=(11, 9), dpi=150)
    i = 0
    for env_id in results:
        r, c = i // 3, i % 3
        colours = ["black", "red"]
        for j, key in enumerate(["non_markov", "markov"]):
            returns = jnp.array(list(results[env_id][key].values()))
            returns_avg = jnp.mean(returns, axis=0)  # (n_iters,)
            returns_5 = jnp.percentile(returns, 5, axis=0)
            returns_95 = jnp.percentile(returns, 95, axis=0)
            xs = jnp.linspace(0, 1_000_000, len(returns_avg))

            ax[r, c].plot(
                xs,
                returns_avg,
                label=key if i == 0 else "",
                color=colours[j],
            )
            ax[r, c].fill_between(
                xs,
                returns_5,
                returns_95,
                alpha=0.2,
                color=colours[j],
            )
        i += 1

        ax[r, c].grid(axis="y", linestyle=(0, (6, 8)), alpha=0.6)
        ax[r, c].set_xlabel("Number of steps", fontsize=12)
        ax[r, c].set_ylabel("Success rate", fontsize=12)
        ax[r, c].set_title(env_id[6:], fontsize=14)
        ax[r, c].tick_params(axis="both", which="major", labelsize=10)
        ax[r, c].set_xlim(0, 1_000_000)
        ax[r, c].set_ylim(0, 1)
    legend = fig.legend(
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.53, -0.05),
        shadow=False,
        frameon=False,
        handles=ax[0, 0].get_lines(),
        labels=["Non-Markov", "Markov"],
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(os.path.dirname(__file__), "markov_reward.png"),
        bbox_extra_artists=(legend,),
        bbox_inches="tight",
    )


def compute_stats(values):
    """Compute mean ± std string."""
    values = jnp.asarray([v[-1] for k, v in values.items()])
    mean = jnp.mean(values)
    std = jnp.std(values)
    return f"{mean:.3f} ± {std:.3f}"


def table():
    with open("markov_reward.json", "r") as f:
        data = json.load(f)

    envs = list(data.keys())
    markov_row = ["Markov"]
    non_markov_row = ["Non-Markov"]

    for env in envs:
        markov_stats = compute_stats(data[env]["markov"])
        non_markov_stats = compute_stats(
            data[env]["non_markov"]
        )

        markov_row.append(markov_stats)
        non_markov_row.append(non_markov_stats)

    # Build header
    header = [""] + envs
    separator = ["---"] * len(header)

    def format_row(row):
        return "| " + " | ".join(row) + " |"

    table = [
        format_row(header),
        format_row(separator),
        format_row(markov_row),
        format_row(non_markov_row),
    ]

    table = "\n".join(table)
    print(table)
    with open("speedup_env.md", "w") as f:
        f.write(table)


def main(args: Args):
    results = {}
    envs = [
        "Navix-Empty-6x6-v0",
        "Navix-Empty-Random-6x6-v0",
        "Navix-Empty-16x16-v0",
        "Navix-Dynamic-Obstacles-6x6-Random-v0",
        "Navix-DoorKey-5x5-v0",
        "Navix-DoorKey-16x16-v0",
        "Navix-DistShift2-v0",
        "Navix-GoToDoor-6x6-v0",
        "Navix-GoToDoor-8x8-v0",
        "Navix-FourRooms-v0",
        "Navix-LavaGapS6-v0",
        "Navix-LavaGapS7-v0",
    ]
    for env_id in envs:
        results[env_id] = {"markov": {}, "non_markov": {}}
        print(f"Training on environment: {env_id}")
        return_hist_markov = train(env_id, args, markov=True)
        return_hist_non_markov = train(env_id, args, markov=False)
        print(f"Completed training for environment: {env_id}")
        # save log to disk

        for i, seed in enumerate(args.seeds_offset + jnp.arange(args.n_seeds)):
            results[env_id]["markov"][seed.item()] = return_hist_markov[i].tolist()
            results[env_id]["non_markov"][seed.item()] = return_hist_non_markov[
                i
            ].tolist()

        with open("markov_reward.json", "w") as f:
            json.dump(results, f)


if __name__ == "__main__":
    args = tyro.cli(Args)
    # main(args)
    # plot()
    table()

