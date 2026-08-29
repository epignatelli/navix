"""Scores navix's own PQN against the Navix1M benchmark preset.

Reproduction: `python run.py` from any directory, no arguments - this
entry's static metadata lives in the sibling `config.yml`, its pinned
dependencies in the sibling `requirements.txt`."""
import sys
from pathlib import Path
from typing import Dict, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import jax
import yaml

from navix.agents import PQN, PQNHparams
from navix.agents.agent import Agent
from navix.agents.models import QNetwork
from navix.benchmarks import AlgorithmEntry, Navix1M
from navix.environments.registry import make

HERE = Path(__file__).resolve().parent


class PQNEntry(AlgorithmEntry):
    def train(self, env_id: str, rng: jax.Array) -> Tuple[Agent, Dict[str, jax.Array]]:
        env = make(env_id)
        hparams = PQNHparams(budget=self.budget)
        # QNetwork flattens whatever shape env.observation_fn returns
        # internally - no FlattenObsWrapper needed, same ergonomics as
        # Dreamer's world model.
        network = QNetwork(action_dim=len(env.action_set), hidden_size=hparams.hidden_size)
        agent = PQN(hparams=hparams, network=network, env=env)
        return agent.train(rng)


if __name__ == "__main__":
    config = yaml.safe_load((HERE / "config.yml").read_text())

    entry = PQNEntry(
        name=config["name"],
        author=config["author"],
        paper_url=config["paper_url"],
        navix_commit_url=config["navix_commit_url"],
        algorithm_commit_url=config["algorithm_commit_url"],
    )

    benchmark = Navix1M()
    raw = benchmark.run(entry)
    summary = benchmark.summary(raw)
    print(f"{type(benchmark).__name__} / {entry.name} summary:")
    print(f"  episodic_returns:     {summary['episodic_returns']}")
    print(f"  flops:                {summary['flops']}")
    print(f"  memory_bytes:         {summary['memory_bytes']}")
    print(f"  compile_time_seconds: {summary['compile_time_seconds']}")
    print(f"  fps:                  {summary['fps']}")
    print(f"  wall_time:            {summary['wall_time']}")
    print(f"  returns_variance:            {summary['returns_variance']}")
    print(f"  returns_convergence_rate:    {summary['returns_convergence_rate']}")
    per_env = raw.last_percent_mean()
    for i, env_id in enumerate(raw.info["env_ids"]):
        print(f"  {env_id}: episodic_returns={per_env.episodic_returns[i]} length={per_env.length[i]}")
