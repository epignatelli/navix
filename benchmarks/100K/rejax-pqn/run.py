"""Scores rejax's PQN against the Navix100K benchmark preset.

Reproduction: `python run.py` from any directory, no arguments - this
entry's static metadata lives in the sibling `config.yml`, its pinned
dependencies in the sibling `requirements.txt`.

rejax ships a first-class navix integration (`rejax[compat]`,
`rejax.compat.navix2gymnax`) - `env="navix/<env_id>"` resolves through
it directly, using navix's own `wrappers.ToGymnax` internally. No
manual env adapter needed here, unlike a library without built-in
navix support would require."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import rejax
import yaml

from navix.benchmarks import AlgorithmEntry, Navix100K, TrainingCurve

HERE = Path(__file__).resolve().parent


class RejaxPQNEntry(AlgorithmEntry):
    def train(self, env_id: str, budget: int, rng: jax.Array) -> TrainingCurve:
        algo = rejax.PQN.create(env=f"navix/{env_id}", total_timesteps=budget)
        _, (lengths, returns) = algo.train(rng)
        return TrainingCurve(
            episodic_returns=jnp.mean(returns, axis=-1),
            lengths=jnp.mean(lengths, axis=-1),
        )


if __name__ == "__main__":
    config = yaml.safe_load((HERE / "config.yml").read_text())

    entry = RejaxPQNEntry(
        name=config["name"],
        author=config["author"],
        paper_url=config["paper_url"],
        navix_commit_url=config["navix_commit_url"],
        algorithm_commit_url=config["algorithm_commit_url"],
    )

    benchmark = Navix100K()
    raw = benchmark.run(entry)
    summary = benchmark.summary(raw)
    details = benchmark.details(raw)
    benchmark.submit_entry(entry, raw)
    print(f"{type(benchmark).__name__} / {entry.name} summary:")
    print(f"  episodic_returns:     {summary['episodic_returns']}")
    print(f"  flops:                {summary['flops']}")
    print(f"  memory_bytes:         {summary['memory_bytes']}")
    print(f"  compile_time_seconds: {summary['compile_time_seconds']}")
    print(f"  fps:                  {summary['fps']}")
    print(f"  wall_time:            {summary['wall_time']}")
    print(f"  returns_variance:            {summary['returns_variance']}")
    print(f"  returns_convergence_rate:    {summary['returns_convergence_rate']}")
    for i, env_id in enumerate(details["env_ids"]):
        print(f"  {env_id}: episodic_returns={details['episodic_returns'][i]} length={details['length'][i]}")
