"""Scores navix's own PQN against the Navix100K benchmark preset.

Reproduction: `python run.py` from any directory, no arguments - this
entry's static metadata lives in the sibling `config.yml`, its pinned
dependencies in the sibling `requirements.txt`.

Each environment gets its own Evolution-Strategies hyperparameter
search (`navix.benchmarks.search.search_hparams`) at a reduced budget
before the real, full-budget scoring run - see `HPARAMS_DISTR`/
`SEARCH_*` below. This is intentionally not a `Benchmark`/
`AlgorithmEntry` feature (see `navix/benchmarks/search.py`'s module
docstring for why) - it's just this entry's own `run.py` using
`search_hparams` as a library, then baking the per-env results into
`PQNEntry.hparams` before the real run.

MDP (fully observable) only, written to an `mdp/` subdirectory
(`Benchmark.submit_entry`'s `subdir` argument) for consistency with
`navix-ppo`'s entry, which also has a `pomdp/` arm - `QNetwork` always
flattens its input internally (see `navix/agents/models.py`), no
pluggable CNN encoder the way `ActorCritic` has, so handing it raw
partially-observable pixels would just flatten them into a plain
Dense/LayerNorm stack - not a meaningful visual-RL comparison, so it's
skipped here rather than forced."""
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import distrax
import jax
import jax.numpy as jnp
import yaml

from navix.agents import PQN, PQNHparams
from navix.agents.agent import masked_mean
from navix.agents.models import QNetwork
from navix.benchmarks import AlgorithmEntry, Navix100K, TrainingCurve
from navix.benchmarks.search import search_hparams
from navix.environments.registry import make

HERE = Path(__file__).resolve().parent

HPARAMS_DISTR = {
    "lr": distrax.Uniform(low=1e-5, high=1e-2),
    "max_grad_norm": distrax.Uniform(low=0.5, high=20.0),
    "q_lambda": distrax.Uniform(low=0.5, high=0.99),
    "start_e": distrax.Uniform(low=0.5, high=1.0),
    "end_e": distrax.Uniform(low=0.01, high=0.2),
    "exploration_fraction": distrax.Uniform(low=0.05, high=0.5),
}
"""PQNHparams' own continuous (pytree_node=True) fields - excludes
budget/num_envs/num_steps/num_minibatches/num_epochs/anneal_lr (all
pytree_node=False - batch-shape/flag knobs, structurally unsearchable
via ES) and hidden_size (network width, not sensible to perturb with
Gaussian noise)."""

SEARCH_SEEDS = (0, 1, 2, 3)
SEARCH_POP_SIZE = 8
SEARCH_NUM_GENERATIONS = 10
SEARCH_BUDGET_FRACTION = 0.1
"""Search trains at this fraction of the real scoring budget per
candidate - enough to differentiate good from bad hyperparameters
without paying full training cost for every one of pop_size *
num_generations candidates."""


def train_with_hparams(hparams: Dict[str, float], env_id: str, budget: int, rng: jax.Array) -> TrainingCurve:
    env = make(env_id)
    hp = PQNHparams(budget=budget).replace(**hparams)
    # QNetwork flattens whatever shape env.observation_fn returns
    # internally - no FlattenObsWrapper needed, same ergonomics as
    # Dreamer's world model.
    network = QNetwork(action_dim=len(env.action_set), hidden_size=hp.hidden_size)
    agent = PQN(hparams=hp, network=network, env=env)
    _, logs = agent.train(rng)
    mask = jnp.asarray(logs["done_mask"], dtype=jnp.bool_)
    return TrainingCurve(
        episodic_returns=masked_mean(logs["returns"], mask, axis=(-2, -1)),
        lengths=masked_mean(logs["lengths"], mask, axis=(-2, -1)),
    )


@dataclass
class PQNEntry(AlgorithmEntry):
    hparams: Dict[str, Dict[str, float]] = field(default_factory=dict)
    """env_id -> per-field hyperparameter overrides (see `run.py`'s
    module docstring) - looked up per env_id in `train`, empty
    (PQNHparams' own defaults) for any env_id not present."""

    def train(self, env_id: str, budget: int, rng: jax.Array) -> TrainingCurve:
        return train_with_hparams(self.hparams.get(env_id, {}), env_id, budget, rng)


if __name__ == "__main__":
    config = yaml.safe_load((HERE / "config.yml").read_text())

    entry = PQNEntry(
        name=config["name"],
        author=config["author"],
        paper_url=config["paper_url"],
        navix_commit_url=config["navix_commit_url"],
        algorithm_commit_url=config["algorithm_commit_url"],
    )

    benchmark = Navix100K()
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
    benchmark.submit_entry(entry, raw, subdir="mdp")
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
