"""Scores navix's own Dreamer against the Navix1M benchmark preset.

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
`DreamerEntry.hparams` before the real run."""
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import distrax
import jax
import jax.numpy as jnp
import numpy as np
import yaml

from navix.agents import Dreamer, DreamerHparams, WorldModel, DreamerActor, DreamerCritic
from navix.agents.agent import masked_mean
from navix.benchmarks import AlgorithmEntry, Navix1M, TrainingCurve
from navix.benchmarks.search import search_hparams
from navix.environments.registry import make

HERE = Path(__file__).resolve().parent

HPARAMS_DISTR = {
    "model_lr": distrax.Uniform(low=1e-5, high=1e-2),
    "actor_lr": distrax.Uniform(low=1e-5, high=1e-2),
    "critic_lr": distrax.Uniform(low=1e-5, high=1e-2),
    "max_grad_norm": distrax.Uniform(low=10.0, high=200.0),
    "actor_entropy": distrax.Uniform(low=0.0, high=1e-2),
    "lam": distrax.Uniform(low=0.8, high=0.99),
}
"""A deliberately modest subset of DreamerHparams' many continuous
(pytree_node=True) fields - the standard "optimizer + entropy" knobs.
DreamerHparams has ~20 more continuous fields (world-model KL
balancing, symexp-twohot bins, slow-critic EMA rates, ...) left at
their own defaults here - searching all of them at once is a much
higher-dimensional problem than this entry takes on."""

SEARCH_SEEDS = (0, 1, 2, 3)
SEARCH_POP_SIZE = 8
SEARCH_NUM_GENERATIONS = 10
SEARCH_BUDGET_FRACTION = 0.1
"""Search trains at this fraction of the real scoring budget per
candidate - enough to differentiate good from bad hyperparameters
without paying full training cost for every one of pop_size *
num_generations candidates."""


def train_with_hparams(hparams: Dict[str, float], env_id: str, budget: int, rng: jax.Array) -> TrainingCurve:
    """Builds `env_id`'s env/world model/actor/critic, trains Dreamer
    on it for `budget` frames with `hparams` overriding `DreamerHparams`'
    defaults, and reduces the result to a `TrainingCurve`. The shared
    trainable both `search_hparams` (at a reduced budget) and
    `DreamerEntry.train` (at the real budget) call - only `budget` and
    `hparams` differ between the two calls.

    Args:
        hparams (Dict[str, float]): `DreamerHparams` field overrides
            (see `HPARAMS_DISTR` above) - empty uses `DreamerHparams`'
            own defaults.
        env_id (str): The environment to train on.
        budget (int): Training budget in frames.
        rng (jax.Array): PRNG key for this single training run (the
            caller vmaps over seeds, this function never does).

    Returns:
        TrainingCurve: `episodic_returns`/`lengths` (masked-mean over
        completed episodes), plus every `loss/*`/`agent/*` entry
        Dreamer's own training loop already computes per update as
        `diagnostics` (world-model KL/reconstruction losses, actor/
        critic losses, imagined-rollout statistics)."""
    env = make(env_id)
    hp = DreamerHparams(budget=budget).replace(**hparams)
    # Unlike PPO's ActorCritic (which needs a pre-flattened observation),
    # Dreamer's world model flattens whatever shape env.observation_fn
    # returns internally - obs_dim just needs to match the flattened
    # size of that raw shape.
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = len(env.action_set)
    agent = Dreamer(
        hparams=hp,
        env=env,
        world=WorldModel(obs_dim=obs_dim, act_dim=act_dim, hparams=hp),
        actor=DreamerActor(act_dim=act_dim, hidden=hp.hidden_size),
        critic=DreamerCritic(
            hidden=hp.hidden_size,
            bins=hp.bins,
            low=hp.bins_low,
            high=hp.bins_high,
        ),
    )
    _, logs = agent.train(rng)
    mask = jnp.asarray(logs["done_mask"], dtype=jnp.bool_)
    # Dreamer's own training loop already reduces every loss/*/agent/*
    # entry to one scalar per training update - already the exact
    # per-update-curve shape TrainingCurve.diagnostics wants, no
    # further reduction needed (same reasoning as navix-ppo/navix-pqn's
    # run.py).
    diagnostics = {key: value for key, value in logs.items() if key.startswith("loss/") or key.startswith("agent/")}
    return TrainingCurve(
        episodic_returns=masked_mean(logs["returns"], mask, axis=(-2, -1)),
        lengths=masked_mean(logs["lengths"], mask, axis=(-2, -1)),
        diagnostics=diagnostics,
    )


@dataclass
class DreamerEntry(AlgorithmEntry):
    """navix's own Dreamer, wired into `Benchmark`'s `AlgorithmEntry`
    protocol - see this module's docstring for the per-env hparam
    search this entry scores under."""

    hparams: Dict[str, Dict[str, float]] = field(default_factory=dict)
    """env_id -> per-field hyperparameter overrides (see `run.py`'s
    module docstring) - looked up per env_id in `train`, empty
    (DreamerHparams' own defaults) for any env_id not present."""

    def train(self, env_id: str, budget: int, rng: jax.Array) -> TrainingCurve:
        """`AlgorithmEntry.train`, delegating to `train_with_hparams`
        with this entry's own `hparams`."""
        return train_with_hparams(self.hparams.get(env_id, {}), env_id, budget, rng)


if __name__ == "__main__":
    config = yaml.safe_load((HERE / "config.yml").read_text())

    entry = DreamerEntry(
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
