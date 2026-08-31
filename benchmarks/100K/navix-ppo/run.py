"""Scores navix's own PPO against the Navix100K benchmark preset, under
two observation configurations: `mdp` (fully observable, `observations.
symbolic`, flattened into `ActorCritic`'s default `MLPEncoder`) and
`pomdp` (partially observable, `observations.rgb_first_person`, fed to
`ActorCritic`'s `ConvEncoder` instead - raw pixels need a convolutional
encoder to make architectural sense, see `navix/agents/models.py`).
Each writes its own `summary.json`/`details.json`/`diagnostics.npz`
into a `mdp/`/`pomdp/` subdirectory of this entry's folder
(`Benchmark.submit_entry`'s `subdir` argument) - `PPOEntry.hparams` is
the same per-env-override mechanism either way, `observation_mode` is
just another (fixed-at-construction, not per-env) field on the entry,
same reasoning as `navix/benchmarks/search.py`'s module docstring for
why this isn't a `Benchmark`/`AlgorithmEntry` feature: what's
configurable here is inherently entry-specific (not every algorithm
even has a pluggable encoder to swap - every rejax network always
flattens internally, no CNN option at all, so this two-observation-mode
setup is specific to navix's own agents, not a general `Benchmark`
capability).

Reproduction: `python run.py` from any directory runs both modes
sequentially; `python run.py mdp` or `python run.py pomdp` runs just
one (lets each mode be put on its own GPU as a separate process) - this
entry's static metadata lives in the sibling `config.yml`, its pinned
dependencies in the sibling `requirements.txt`.

Each (observation mode, environment) pair gets its own Evolution-
Strategies hyperparameter search (`navix.benchmarks.search.
search_hparams`) at a reduced budget before the real, full-budget
scoring run - see `HPARAMS_DISTR`/`SEARCH_*` below."""
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Type

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import yaml

from navix import observations
from navix.agents import PPO, PPOHparams, ActorCritic
from navix.agents.agent import masked_mean
from navix.agents.models import ConvEncoder, MLPEncoder
from navix.benchmarks import AlgorithmEntry, Navix100K, TrainingCurve
from navix.benchmarks.search import search_hparams
from navix.environments.environment import Environment
from navix.environments.registry import make
from navix.es import LogUniform

HERE = Path(__file__).resolve().parent

OBSERVATION_MODES = ("mdp", "pomdp")

HPARAMS_DISTR = {
    "lr": LogUniform(low=1e-5, high=1e-2),
    "gae_lambda": distrax.Uniform(low=0.8, high=0.99),
    "clip_eps": distrax.Uniform(low=0.1, high=0.3),
    "vf_coef": distrax.Uniform(low=0.1, high=1.0),
    "ent_coef": distrax.Uniform(low=0.0, high=0.05),
    "max_grad_norm": distrax.Uniform(low=0.1, high=1.0),
}
"""PPOHparams' own continuous (pytree_node=True) fields - excludes
budget/num_envs/num_steps/num_minibatches/num_epochs/anneal_lr/
normalise_advantage/clip_value_loss (all pytree_node=False - batch-
shape/architecture/flag knobs, structurally unsearchable via ES, same
boundary `Experiment.run_hparam_search` already enforces). Shared
across both observation modes - it's the same PPOHparams either way,
only the network/env wiring differs."""

SEARCH_SEEDS = (0, 1, 2, 3)
SEARCH_POP_SIZE = 8
SEARCH_NUM_GENERATIONS = 10
SEARCH_BUDGET_FRACTION = 0.1
"""Search trains at this fraction of the real scoring budget per
candidate - enough to differentiate good from bad hyperparameters
without paying full training cost for every one of pop_size *
num_generations candidates."""


def flatten_obs(env: Environment) -> Environment:
    """PPO's `MLPEncoder` needs a pre-flattened observation - only used
    for `mdp` mode. `pomdp` mode's `ConvEncoder` wants the raw `(H, W,
    3)` image `observations.rgb_first_person` already returns."""
    flatten_obs_fn = lambda x: jnp.ravel(env.observation_fn(x))
    flatten_shape = (int(np.prod(env.observation_space.shape)),)
    return env.replace(
        observation_fn=flatten_obs_fn,
        observation_space=env.observation_space.replace(shape=flatten_shape),
    )


def train_with_hparams(
    hparams: Dict[str, float], env_id: str, budget: int, rng: jax.Array, observation_mode: str
) -> TrainingCurve:
    encoder_cls: Type[nn.Module]
    if observation_mode == "mdp":
        env = flatten_obs(make(env_id, observation_fn=observations.symbolic))
        encoder_cls = MLPEncoder
    elif observation_mode == "pomdp":
        env = make(env_id, observation_fn=observations.rgb_first_person)
        encoder_cls = ConvEncoder
    else:
        raise ValueError(f"Unknown observation_mode {observation_mode!r}, expected one of {OBSERVATION_MODES}.")

    hp = PPOHparams(budget=budget).replace(**hparams)
    network = ActorCritic(action_dim=len(env.action_set), actor_encoder=encoder_cls(), critic_encoder=encoder_cls())
    agent = PPO(hparams=hp, network=network, env=env)
    _, logs = agent.train(rng)
    mask = jnp.asarray(logs["done_mask"], dtype=jnp.bool_)
    return TrainingCurve(
        episodic_returns=masked_mean(logs["returns"], mask, axis=(-2, -1)),
        lengths=masked_mean(logs["lengths"], mask, axis=(-2, -1)),
    )


@dataclass
class PPOEntry(AlgorithmEntry):
    hparams: Dict[str, Dict[str, float]] = field(default_factory=dict)
    """env_id -> per-field hyperparameter overrides (see `run.py`'s
    module docstring) - looked up per env_id in `train`, empty
    (PPOHparams' own defaults) for any env_id not present."""
    observation_mode: str = "mdp"
    """`"mdp"` or `"pomdp"` - fixed for the whole entry (unlike
    `hparams`, doesn't vary per env_id)."""

    def train(self, env_id: str, budget: int, rng: jax.Array) -> TrainingCurve:
        return train_with_hparams(self.hparams.get(env_id, {}), env_id, budget, rng, self.observation_mode)


if __name__ == "__main__":
    config = yaml.safe_load((HERE / "config.yml").read_text())

    # `python run.py` (no arguments) runs every mode in `OBSERVATION_MODES`
    # sequentially in one process, as documented above. `python run.py
    # <mode> [<mode> ...]` runs only the given mode(s) - lets a caller (e.g.
    # a GPU-queue script) put each mode on its own GPU as a separate
    # process instead, without changing what a bare `python run.py` does.
    modes_to_run = sys.argv[1:] or list(OBSERVATION_MODES)
    for m in modes_to_run:
        if m not in OBSERVATION_MODES:
            raise ValueError(f"Unknown observation_mode {m!r} on the command line, expected one of {OBSERVATION_MODES}.")

    benchmark = Navix100K()
    search_budget = max(1, int(benchmark.budget * SEARCH_BUDGET_FRACTION))

    for observation_mode in modes_to_run:
        print(f"\n{'=' * 20} observation_mode={observation_mode} {'=' * 20}")
        entry = PPOEntry(
            name=config["name"],
            author=config["author"],
            paper_url=config["paper_url"],
            navix_commit_url=config["navix_commit_url"],
            algorithm_commit_url=config["algorithm_commit_url"],
            observation_mode=observation_mode,
        )

        tuned_hparams = {}
        for env_id in benchmark.env_ids:
            print(f"Searching hyperparameters for {env_id} (budget={search_budget})...")
            best_hparams, best_fitness = search_hparams(
                trainable=lambda hp, rng, env_id=env_id: train_with_hparams(
                    hp, env_id, search_budget, rng, observation_mode
                ),
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
        benchmark.submit_entry(entry, raw, subdir=observation_mode)
        print(f"{type(benchmark).__name__} / {entry.name} [{observation_mode}] summary:")
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
