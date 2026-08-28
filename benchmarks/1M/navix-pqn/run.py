"""Scores navix's own PQN against the Navix1M benchmark preset.

Reproduction: `python run.py` from any directory, no arguments - this
entry's static metadata lives in the sibling `config.yml`, its pinned
dependencies in the sibling `requirements.txt`."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import jax
import yaml

from navix.agents import PQN, PQNHparams
from navix.agents.models import QNetwork
from navix.benchmark import AlgorithmEntry, Navix1M, last_percent_mean
from navix.environments.environment import Environment

HERE = Path(__file__).resolve().parent


def make_pqn(env: Environment, hparams: PQNHparams) -> PQN:
    # QNetwork flattens whatever shape env.observation_fn returns
    # internally - no FlattenObsWrapper needed, same ergonomics as
    # Dreamer's world model.
    network = QNetwork(action_dim=len(env.action_set), hidden_size=hparams.hidden_size)
    return PQN(hparams=hparams, network=network, env=env)


if __name__ == "__main__":
    config = yaml.safe_load((HERE / "config.yml").read_text())

    entry = AlgorithmEntry(
        name=config["name"],
        author=config["author"],
        paper_url=config["paper_url"],
        navix_commit_url=config["navix_commit_url"],
        algorithm_commit_url=config["algorithm_commit_url"],
        agent_factory=lambda env: make_pqn(env, PQNHparams()),
    )

    raw = Navix1M.run(entry)
    summary = Navix1M.summary(raw)
    print(f"{Navix1M.__name__} / {entry.name} summary:")
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
