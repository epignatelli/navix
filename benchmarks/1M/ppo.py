"""Scores navix's own PPO against the Navix1M benchmark preset."""
from dataclasses import dataclass
import subprocess

import numpy as np
import jax.numpy as jnp
import tyro

from navix.agents import PPO, PPOHparams, ActorCritic
from navix.benchmarks import AlgorithmEntry, Navix1M
from navix.environments.environment import Environment


def _flatten_obs(env: Environment) -> Environment:
    flatten_obs_fn = lambda x: jnp.ravel(env.observation_fn(x))
    flatten_shape = (int(np.prod(env.observation_space.shape)),)
    return env.replace(
        observation_fn=flatten_obs_fn,
        observation_space=env.observation_space.replace(shape=flatten_shape),
    )


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def make_ppo(env: Environment, hparams: PPOHparams) -> PPO:
    env = _flatten_obs(env)
    return PPO(hparams=hparams, network=ActorCritic(action_dim=len(env.action_set)), env=env)


@dataclass
class Args:
    author: str = "navix"
    """Author of this implementation, for the AlgorithmEntry's provenance."""
    log_to_wandb: bool = True
    ppo: PPOHparams = PPOHparams()


if __name__ == "__main__":
    args = tyro.cli(Args)
    commit_sha = _git_sha()

    entry = AlgorithmEntry(
        name="PPO",
        author=args.author,
        paper_url="https://arxiv.org/abs/1707.06347",
        commit_sha=commit_sha,
        requirements_url=(
            f"https://raw.githubusercontent.com/epignatelli/navix/{commit_sha}/requirements.txt"
        ),
        agent_factory=lambda env: make_ppo(env, args.ppo),
    )

    result = Navix1M(entry).run(log_to_wandb=args.log_to_wandb)
    print(f"{Navix1M.name} results:")
    print(f"  success_rate:   {result.success_rate}")
    print(f"  returns:        {result.returns}")
    print(f"  episode_length: {result.episode_length}")
    print(f"  fps:            {result.fps}")
    print(f"  wall_time:      {result.wall_time}")
