"""Scores rejax's IQN against the Navix1M benchmark preset.

Reproduction: `python run.py` from any directory, no arguments - this
entry's static metadata lives in the sibling `config.yml`, its pinned
dependencies in the sibling `requirements.txt`.

rejax ships a first-class navix integration (`rejax[compat]`,
`rejax.compat.navix2gymnax`) - `env="navix/<env_id>"` resolves through
it directly, using navix's own `wrappers.ToGymnax` internally. No
manual env adapter needed here, unlike a library without built-in
navix support would require.

Each environment gets its own Evolution-Strategies hyperparameter
search (`navix.benchmarks.search.search_hparams`) at a reduced budget
before the real, full-budget scoring run - see `HPARAMS_DISTR`/
`SEARCH_*` below. This is intentionally not a `Benchmark`/
`AlgorithmEntry` feature (see `navix/benchmarks/search.py`'s module
docstring for why) - it's just this entry's own `run.py` using
`search_hparams` as a library, then baking the per-env results into
`RejaxIQNEntry.hparams` before the real run."""
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import distrax
import jax
import jax.numpy as jnp
import rejax
import yaml

from navix.benchmarks import AlgorithmEntry, Navix1M, TrainingCurve
from navix.benchmarks.search import search_hparams

HERE = Path(__file__).resolve().parent

HPARAMS_DISTR = {
    "learning_rate": distrax.Uniform(low=1e-5, high=1e-2),
    "max_grad_norm": distrax.Uniform(low=0.5, high=20.0),
    "eps_start": distrax.Uniform(low=0.5, high=1.0),
    "eps_end": distrax.Uniform(low=0.01, high=0.2),
    "polyak": distrax.Uniform(low=0.9, high=0.999),
    "kappa": distrax.Uniform(low=0.5, high=2.0),
}
"""rejax.IQN's own continuous hyperparameters - excludes buffer_size/
fill_buffer/batch_size/target_update_freq/num_epochs/num_tau_samples/
num_tau_prime_samples (batch-shape/architecture knobs, not really
"tuning" in the same sense), gamma (more a task-horizon property than
something to search), and exploration_fraction (EpsilonGreedyMixin
declares it pytree_node=False - rejax's own epsilon_schedule does
int(exploration_fraction * total_timesteps), which requires a concrete
Python value, not a vmapped/traced one - structurally unsearchable via
ES, same boundary Experiment.run_hparam_search enforces for navix's
own pytree_node=False fields)."""

SEARCH_SEEDS = (0, 1, 2, 3)
SEARCH_POP_SIZE = 8
SEARCH_NUM_GENERATIONS = 10
SEARCH_BUDGET_FRACTION = 0.1
"""Search trains at this fraction of the real scoring budget per
candidate - enough to differentiate good from bad hyperparameters
without paying full training cost for every one of pop_size *
num_generations candidates."""


def train_with_hparams(hparams: Dict[str, float], env_id: str, budget: int, rng: jax.Array) -> TrainingCurve:
    algo = rejax.IQN.create(env=f"navix/{env_id}", total_timesteps=budget, **hparams)
    _, (lengths, returns) = algo.train(rng)
    return TrainingCurve(
        episodic_returns=jnp.mean(returns, axis=-1),
        lengths=jnp.mean(lengths, axis=-1),
    )


@dataclass
class RejaxIQNEntry(AlgorithmEntry):
    hparams: Dict[str, Dict[str, float]] = field(default_factory=dict)
    """env_id -> per-field hyperparameter overrides (see `run.py`'s
    module docstring) - looked up per env_id in `train`, empty (rejax's
    own defaults) for any env_id not present."""

    def train(self, env_id: str, budget: int, rng: jax.Array) -> TrainingCurve:
        return train_with_hparams(self.hparams.get(env_id, {}), env_id, budget, rng)


if __name__ == "__main__":
    config = yaml.safe_load((HERE / "config.yml").read_text())

    entry = RejaxIQNEntry(
        name=config["name"],
        author=config["author"],
        paper_url=config["paper_url"],
        navix_commit_url=config["navix_commit_url"],
        algorithm_commit_url=config["algorithm_commit_url"],
    )

    benchmark = Navix1M()
    search_budget = max(1, int(benchmark.budget * SEARCH_BUDGET_FRACTION))

    tuned_hparams = {}
    for env_id in benchmark.env_ids:
        print(f"Searching hyperparameters for {env_id} (budget={search_budget})...")
        best_hparams, best_fitness = search_hparams(
            trainable=lambda hp, rng, env_id=env_id: train_with_hparams(hp, env_id, search_budget, rng),
            hparams_distr=HPARAMS_DISTR,
            seeds=SEARCH_SEEDS,
            pop_size=SEARCH_POP_SIZE,
            num_generations=SEARCH_NUM_GENERATIONS,
        )
        print(f"{env_id}: best hparams {best_hparams} (fitness {best_fitness})")
        tuned_hparams[env_id] = best_hparams

    entry = replace(entry, hparams=tuned_hparams)

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
