"""Scores rejax's PPO against the Navix1M benchmark preset.

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
`RejaxPPOEntry.hparams` before the real run.

MDP (fully observable) only, written to an `mdp/` subdirectory
(`Benchmark.submit_entry`'s `subdir` argument) for consistency with
`navix-ppo`'s entry, which also has a `pomdp/` arm - every rejax
network (`DiscretePolicy`, `VNetwork`, ...) is MLP-only, no pluggable
CNN encoder, so handing it raw partially-observable pixels would just
flatten them into a plain MLP - not a meaningful visual-RL comparison,
so it's skipped here rather than forced."""
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
from navix.es import LogUniform

HERE = Path(__file__).resolve().parent

HPARAMS_DISTR = {
    "learning_rate": LogUniform(low=1e-5, high=1e-2),
    "gae_lambda": distrax.Uniform(low=0.8, high=0.99),
    "clip_eps": distrax.Uniform(low=0.1, high=0.3),
    "vf_coef": distrax.Uniform(low=0.1, high=1.0),
    "ent_coef": distrax.Uniform(low=0.0, high=0.05),
    "max_grad_norm": distrax.Uniform(low=0.1, high=1.0),
}
"""rejax.PPO's own continuous hyperparameters - excludes num_envs/
num_steps/num_minibatches/num_epochs (batch-shape knobs, not really
"tuning" in the same sense) and gamma (more a task-horizon property
than something to search)."""

SEARCH_SEEDS = (0, 1, 2, 3)
SEARCH_POP_SIZE = 8
SEARCH_NUM_GENERATIONS = 10
SEARCH_BUDGET_FRACTION = 0.1
"""Search trains at this fraction of the real scoring budget per
candidate - enough to differentiate good from bad hyperparameters
without paying full training cost for every one of pop_size *
num_generations candidates."""


def train_with_hparams(hparams: Dict[str, float], env_id: str, budget: int, rng: jax.Array) -> TrainingCurve:
    """Trains `rejax.PPO` on `env_id` for `budget` timesteps with
    `hparams` overriding its own defaults, and reduces the result to a
    `TrainingCurve`. The shared trainable both `search_hparams` (at a
    reduced budget) and `RejaxPPOEntry.train` (at the real budget)
    call - only `budget` and `hparams` differ between the two calls.

    Args:
        hparams (Dict[str, float]): `rejax.PPO`'s own hyperparameter
            overrides (see `HPARAMS_DISTR` above) - empty uses
            `rejax.PPO`'s own defaults.
        env_id (str): The environment to train on.
        budget (int): Training budget in timesteps.
        rng (jax.Array): PRNG key for this single training run (the
            caller vmaps over seeds, this function never does).

    Returns:
        TrainingCurve: `episodic_returns`/`lengths`, meaned over
        `rejax`'s own per-episode axis (rejax's `train` already
        returns one point per evaluation, unlike navix's masked-mean
        over a raw per-step stream - no `diagnostics` here, `rejax`'s
        `train` doesn't expose per-update loss terms the way navix's
        own agents' `logs` do)."""
    algo = rejax.PPO.create(env=f"navix/{env_id}", total_timesteps=budget, **hparams)
    _, (lengths, returns) = algo.train(rng)
    return TrainingCurve(
        episodic_returns=jnp.mean(returns, axis=-1),
        lengths=jnp.mean(lengths, axis=-1),
    )


@dataclass
class RejaxPPOEntry(AlgorithmEntry):
    """`rejax`'s PPO, wired into `Benchmark`'s `AlgorithmEntry`
    protocol - see this module's docstring for why no vendored code is
    needed (rejax ships a first-class navix integration) and the
    per-env hparam search this entry scores under."""

    hparams: Dict[str, Dict[str, float]] = field(default_factory=dict)
    """env_id -> per-field hyperparameter overrides (see `run.py`'s
    module docstring) - looked up per env_id in `train`, empty (rejax's
    own defaults) for any env_id not present."""

    def train(self, env_id: str, budget: int, rng: jax.Array) -> TrainingCurve:
        """`AlgorithmEntry.train`, delegating to `train_with_hparams`
        with this entry's own `hparams`."""
        return train_with_hparams(self.hparams.get(env_id, {}), env_id, budget, rng)


if __name__ == "__main__":
    config = yaml.safe_load((HERE / "config.yml").read_text())

    entry = RejaxPPOEntry(
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
