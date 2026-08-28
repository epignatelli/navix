"""Scores navix's own Dreamer against the Navix100K benchmark preset.

Reproduction: `python run.py` from any directory, no arguments - this
entry's static metadata lives in the sibling `config.yml`, its pinned
dependencies in the sibling `requirements.txt`."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import jax
import yaml

from navix.agents import Dreamer, DreamerHparams, WorldModel, DreamerActor, DreamerCritic
from navix.benchmark import AlgorithmEntry, Navix100K, last_percent_mean
from navix.environments.environment import Environment

HERE = Path(__file__).resolve().parent


def make_dreamer(env: Environment, hparams: DreamerHparams) -> Dreamer:
    # Unlike PPO's ActorCritic (which needs a pre-flattened observation),
    # Dreamer's world model flattens whatever shape env.observation_fn
    # returns internally - obs_dim just needs to match the flattened
    # size of that raw shape.
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = len(env.action_set)
    return Dreamer(
        hparams=hparams,
        env=env,
        world=WorldModel(obs_dim=obs_dim, act_dim=act_dim, hparams=hparams),
        actor=DreamerActor(act_dim=act_dim, hidden=hparams.hidden_size),
        critic=DreamerCritic(
            hidden=hparams.hidden_size,
            bins=hparams.bins,
            low=hparams.bins_low,
            high=hparams.bins_high,
        ),
    )


if __name__ == "__main__":
    config = yaml.safe_load((HERE / "config.yml").read_text())

    entry = AlgorithmEntry(
        name=config["name"],
        author=config["author"],
        paper_url=config["paper_url"],
        navix_commit_url=config["navix_commit_url"],
        algorithm_commit_url=config["algorithm_commit_url"],
        agent_factory=lambda env: make_dreamer(env, DreamerHparams()),
    )

    raw = Navix100K.run(entry)
    summary = Navix100K.summary(raw)
    print(f"{Navix100K.__name__} / {entry.name} summary:")
    print(f"  returns:              {summary.returns}")
    print(f"  episode_length:       {summary.episode_length}")
    print(f"  flops:                {summary.flops}")
    print(f"  memory_bytes:         {summary.memory_bytes}")
    print(f"  compile_time_seconds: {summary.compile_time_seconds}")
    print(f"  fps:                  {summary.fps}")
    print(f"  wall_time:            {summary.wall_time}")
    for env_id, result in raw.items():
        m = jax.tree.map(last_percent_mean, result)
        print(f"  {env_id}: returns={m.returns} episode_length={m.episode_length}")
