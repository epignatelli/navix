"""Scores navix's own PQN against the Navix100K benchmark preset, under
two observation configurations: `mdp` (fully observable, `observations.
symbolic`, flattened into `QNetwork`'s default `QMLPEncoder`) and
`pomdp` (partially observable, `observations.rgb_first_person`, fed to
`QNetwork`'s `QConvEncoder` instead - raw pixels need a convolutional
encoder to make architectural sense, see `navix/agents/models.py`).
Each writes its own `summary.json`/`details.json`/`diagnostics.npz`
into a `mdp/`/`pomdp/` subdirectory of this entry's folder
(`Benchmark.submit_entry`'s `subdir` argument) - `PQNEntry.hparams` is
the same per-env-override mechanism either way, `observation_mode` is
just another (fixed-at-construction, not per-env) field on the entry,
same reasoning as `navix/benchmarks/search.py`'s module docstring for
why this isn't a `Benchmark`/`AlgorithmEntry` feature: what's
configurable here is inherently entry-specific (not every algorithm has
a pluggable encoder to swap - every rejax network always flattens
internally, no CNN option at all, so this two-observation-mode setup is
specific to navix's own agents, not a general `Benchmark` capability).

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
from navix.agents import PQN, PQNHparams
from navix.agents.agent import masked_mean
from navix.agents.models import QConvEncoder, QMLPEncoder, QNetwork
from navix.benchmarks import AlgorithmEntry, Navix100K, TrainingCurve
from navix.benchmarks.search import search_hparams
from navix.environments.environment import Environment
from navix.environments.registry import make
from navix.es import LogUniform

HERE = Path(__file__).resolve().parent

OBSERVATION_MODES = ("mdp", "pomdp")

HPARAMS_DISTR = {
    "lr": LogUniform(low=1e-5, high=1e-2),
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
Gaussian noise). Shared across both observation modes - it's the same
PQNHparams either way, only the network/env wiring differs."""

SEARCH_SEEDS = (0, 1, 2, 3)
SEARCH_POP_SIZE = {"mdp": 8, "pomdp": 4}
SEARCH_NUM_GENERATIONS = {"mdp": 10, "pomdp": 5}
SEARCH_BUDGET_FRACTION = {"mdp": 0.1, "pomdp": 0.05}
"""Search trains at this fraction of the real scoring budget per
candidate - enough to differentiate good from bad hyperparameters
without paying full training cost for every one of pop_size *
num_generations candidates. Lower for `pomdp` than `mdp`: `QConvEncoder`
plus PQNHparams' own `num_minibatches=32`/`num_epochs=8` (empirically
justified, not arbitrary - see `navix.agents.pqn`'s module docstring on
why PQN needs more SGD passes per rollout than CleanRL's CartPole-tuned
reference; verified against rejax's own even-higher `num_minibatches=
128` tuned config) makes one PQN pomdp candidate roughly 32x more
per-update compute than one PPO candidate under the same search shape -
confirmed in practice: a `pomdp`-mode search at `mdp`'s pop_size/
generations/budget ran 2+ hours on one environment's first generation
without finishing, while `navix-ppo`'s entire pomdp run (all 6
environments, search + scoring) completed in ~35 minutes. This isn't a
PQNHparams problem to fix - those values are what make PQN actually
solve these environments - so the search itself pays less per
candidate instead: smaller population, fewer generations, shorter
per-candidate training, for `pomdp` only. `mdp` mode already completes
in reasonable time at the original settings, so it keeps them."""


def flatten_obs(env: Environment) -> Environment:
    """`QMLPEncoder` needs a pre-flattened observation - only used for
    `mdp` mode. `pomdp` mode's `QConvEncoder` wants the raw `(H, W, 3)`
    image `observations.rgb_first_person` already returns."""
    flatten_obs_fn = lambda x: jnp.ravel(env.observation_fn(x))
    flatten_shape = (int(np.prod(env.observation_space.shape)),)
    return env.replace(
        observation_fn=flatten_obs_fn,
        observation_space=env.observation_space.replace(shape=flatten_shape),
    )


def train_with_hparams(
    hparams: Dict[str, float], env_id: str, budget: int, rng: jax.Array, observation_mode: str
) -> TrainingCurve:
    """Builds `env_id`'s env/network for `observation_mode`, trains PQN
    on it for `budget` frames with `hparams` overriding `PQNHparams`'
    defaults, and reduces the result to a `TrainingCurve`. The shared
    trainable both `search_hparams` (at a reduced budget) and
    `PQNEntry.train` (at the real budget) call - only `budget` and
    `hparams` differ between the two calls.

    Args:
        hparams (Dict[str, float]): `PQNHparams` field overrides (see
            `HPARAMS_DISTR` above) - empty uses `PQNHparams`' own
            defaults.
        env_id (str): The environment to train on.
        budget (int): Training budget in frames.
        rng (jax.Array): PRNG key for this single training run (the
            caller vmaps over seeds, this function never does).
        observation_mode (str): `"mdp"` or `"pomdp"` (see
            `OBSERVATION_MODES`).

    Returns:
        TrainingCurve: `episodic_returns`/`lengths` (masked-mean over
        completed episodes), plus every `loss/*`/`agent/*` entry PQN's
        own training loop already computes per update as
        `diagnostics`."""
    encoder_cls: Type[nn.Module]
    if observation_mode == "mdp":
        env = flatten_obs(make(env_id, observation_fn=observations.symbolic))
        encoder_cls = QMLPEncoder
    elif observation_mode == "pomdp":
        env = make(env_id, observation_fn=observations.rgb_first_person)
        encoder_cls = QConvEncoder
    else:
        raise ValueError(f"Unknown observation_mode {observation_mode!r}, expected one of {OBSERVATION_MODES}.")

    hp = PQNHparams(budget=budget).replace(**hparams)
    # QNetwork's encoder flattens/convolves whatever shape env.observation_fn
    # returns internally - no FlattenObsWrapper needed for mdp mode, same
    # ergonomics as Dreamer's world model.
    network = QNetwork(action_dim=len(env.action_set), encoder=encoder_cls(hidden_size=hp.hidden_size))
    agent = PQN(hparams=hp, network=network, env=env)
    _, logs = agent.train(rng)
    mask = jnp.asarray(logs["done_mask"], dtype=jnp.bool_)
    # PQN.update already reduces loss/q_loss (and agent/epsilon, PQN's
    # exploration schedule) to one scalar per training update (see
    # PQN.update in navix/agents/pqn.py) - already the exact per-update-
    # curve shape TrainingCurve.diagnostics wants, no further reduction
    # needed. Surfacing these is what makes Benchmark.plot_diagnostics/
    # `diagnostics.npz` show more than just episodic_returns/length for
    # this entry.
    diagnostics = {key: value for key, value in logs.items() if key.startswith("loss/") or key.startswith("agent/")}
    return TrainingCurve(
        episodic_returns=masked_mean(logs["returns"], mask, axis=(-2, -1)),
        lengths=masked_mean(logs["lengths"], mask, axis=(-2, -1)),
        diagnostics=diagnostics,
    )


@dataclass
class PQNEntry(AlgorithmEntry):
    """navix's own PQN, wired into `Benchmark`'s `AlgorithmEntry`
    protocol - see this module's docstring for the two observation
    modes and per-env hparam search this entry scores under."""

    hparams: Dict[str, Dict[str, float]] = field(default_factory=dict)
    """env_id -> per-field hyperparameter overrides (see `run.py`'s
    module docstring) - looked up per env_id in `train`, empty
    (PQNHparams' own defaults) for any env_id not present."""
    observation_mode: str = "mdp"
    """`"mdp"` or `"pomdp"` - fixed for the whole entry (unlike
    `hparams`, doesn't vary per env_id)."""

    def train(self, env_id: str, budget: int, rng: jax.Array) -> TrainingCurve:
        """`AlgorithmEntry.train`, delegating to `train_with_hparams`
        with this entry's own `hparams`/`observation_mode`."""
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

    for observation_mode in modes_to_run:
        print(f"\n{'=' * 20} observation_mode={observation_mode} {'=' * 20}")
        entry = PQNEntry(
            name=config["name"],
            author=config["author"],
            paper_url=config["paper_url"],
            navix_commit_url=config["navix_commit_url"],
            algorithm_commit_url=config["algorithm_commit_url"],
            observation_mode=observation_mode,
        )

        search_budget = max(1, int(benchmark.budget * SEARCH_BUDGET_FRACTION[observation_mode]))
        pop_size = SEARCH_POP_SIZE[observation_mode]
        num_generations = SEARCH_NUM_GENERATIONS[observation_mode]

        tuned_hparams = {}
        for env_id in benchmark.env_ids:
            print(f"Searching hyperparameters for {env_id} (budget={search_budget}, pop_size={pop_size}, num_generations={num_generations})...")
            best_hparams, best_fitness = search_hparams(
                trainable=lambda hp, rng, env_id=env_id: train_with_hparams(
                    hp, env_id, search_budget, rng, observation_mode
                ),
                hparams_distr=HPARAMS_DISTR,
                seeds=SEARCH_SEEDS,
                pop_size=pop_size,
                num_generations=num_generations,
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
