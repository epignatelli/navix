"""Scores navix's own PPO against the Navix1K benchmark preset.

Reproduction: `python run.py` from any directory, no arguments - this
entry's static metadata lives in the sibling `config.yml`, its pinned
dependencies in the sibling `requirements.txt`."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import yaml

from benchmarks._common import flatten_obs
from navix.agents import PPO, PPOHparams, ActorCritic
from navix.benchmarks import AlgorithmEntry, Navix1K
from navix.environments.environment import Environment

HERE = Path(__file__).resolve().parent


def make_ppo(env: Environment, hparams: PPOHparams) -> PPO:
    env = flatten_obs(env)
    return PPO(hparams=hparams, network=ActorCritic(action_dim=len(env.action_set)), env=env)


if __name__ == "__main__":
    config = yaml.safe_load((HERE / "config.yml").read_text())

    entry = AlgorithmEntry(
        name=config["name"],
        author=config["author"],
        paper_url=config["paper_url"],
        navix_commit_url=config["navix_commit_url"],
        algorithm_commit_url=config["algorithm_commit_url"],
        agent_factory=lambda env: make_ppo(env, PPOHparams()),
    )

    result = Navix1K(entry).run(log_to_wandb=True)
    print(f"{Navix1K.name} / {entry.name} results:")
    print(f"  success_rate:   {result.success_rate}")
    print(f"  returns:        {result.returns}")
    print(f"  episode_length: {result.episode_length}")
    print(f"  fps:            {result.fps}")
    print(f"  wall_time:      {result.wall_time}")
