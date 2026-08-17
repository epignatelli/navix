# Copyright 2023 The Navix Authors.

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""A `Benchmark` is a preset experimental setup - a fixed set of
environments, a fixed training budget, a fixed scoring rule - that scores
an algorithm against it, rather than the single-environment,
single-config runs `Experiment` (`navix/experiment.py`) already supports.
`Benchmark.run` is an orchestration layer *over* `Experiment` (one
`Experiment` per environment, budget-overridden), not a replacement for
it.

This is the "Benchmark" issue #130 (the navix leaderboard proposal)
already scoped conceptually - "a scenario-specific function that takes a
trained agent and returns the metric(s) used to rank it" - implemented
here for the simplest of #130's four protocols, from-scratch training,
as two starting presets (`NAVIX_1M`, `NAVIX_100K`) that differ only in
`budget`. The other protocols (zero-shot, curriculum, open-ended) are
still open per #130 and not attempted here.

Every algorithm a `Benchmark` runs is wrapped in an `AlgorithmEntry`,
carrying the provenance metadata #130's "Structure (decided)" section
requires of a leaderboard row (name, suite/author, commit SHA) plus the
implementation author and a link to the paper - so a `BenchmarkResult`
comes out already shaped for a leaderboard row, rather than needing that
metadata assembled separately later. `iter/wall_time` isn't one of these
fields: it's a *result* (already in `logs['iter/wall_time']` via
`Experiment.run`), not something the entry declares up front - and
neither is the ".sh reproduction script" from #130's spec, which is
whatever `benchmarks/*.py` script constructs the entry and calls
`Benchmark.run`, not a field an object carries.

Per #130's "we never vendor an external algorithm's code" decision,
`AlgorithmEntry.agent_factory` is how a `Benchmark` stays algorithm-
agnostic without navix owning the algorithm's implementation: for navix's
own agents (PPO/Dreamer/PQN) it builds a `navix.agents` instance; for an
external algorithm (e.g. rejax, called as a dependency) it would instead
wrap that library's own training entrypoint behind the same `Environment
-> Agent`-shaped interface `Experiment` expects, with no source code
copied into navix.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Tuple

from jax import Array
import jax.numpy as jnp

from .agents.agent import Agent
from .environments.environment import Environment
from .environments.registry import make, registry
from .experiment import Experiment
from .plotting import derive_scalar_metrics


@dataclass
class AlgorithmEntry:
    """One algorithm to score against a `Benchmark`, plus the provenance
    metadata every navix leaderboard entry requires (issue #130)."""

    name: str
    """Algorithm name, e.g. "PPO"."""
    author: str
    """Author of *this implementation* (person/handle) - distinct from
    the paper's authors, since a paper can have multiple independent
    implementations (navix's own, rejax's, a submitter's own)."""
    paper_url: str
    """Link to the paper the algorithm is from."""
    commit_sha: str
    """Commit that produced this implementation, so a result stays
    reproducible even as navix/the algorithm's own repo moves on."""
    agent_factory: Callable[[Environment], Agent]
    """Builds a fresh, env-shaped `Agent` for the given `Environment` -
    called once per environment in the benchmark, the same way
    `benchmarks/ppo.py`/`dreamer.py` build a fresh agent per env inside
    their own loops today. Needed because agents need per-env-shaped
    networks (action_dim/obs_dim), not because state is shared across
    environments."""
    suite: str = "navix"
    """Which suite produced this implementation - "navix" for agents
    shipped in `navix.agents`, or the name of the external
    package/repo/submitter otherwise (e.g. "rejax")."""


@dataclass
class BenchmarkResult:
    entry: AlgorithmEntry
    """Echoes back the entry that was run, so a result is self-describing
    without needing to be paired back up with its `AlgorithmEntry`."""
    scores: Dict[str, Array]
    """Per-environment score (`Benchmark.score_fn`'s output), keyed by
    `env_id`."""
    score: Array
    """Aggregate score: `mean(scores.values())`."""
    logs: Dict[str, Dict]
    """Per-environment raw `logs`, keyed by `env_id` - the same pytree
    `Experiment.run` returns, for diagnostics/plotting via
    `navix.plotting` beyond just the scalar score."""


def default_score_fn(logs: Dict[str, Array]) -> Array:
    """Mean `perf/success_rate` (see `navix.plotting.MANDATORY_METRICS`)
    over the last fifth of training - matching the "last20%" convergence
    check this project's own performance investigations already use by
    hand, now made the default so every `Benchmark` result uses the same
    convention. `perf/success_rate` is already bounded `[0, 1]` and
    comparable across environments regardless of their raw reward scale,
    which is what makes it usable as a cross-environment score without
    any extra normalisation - the open question #130 flagged."""
    metrics = derive_scalar_metrics(logs)
    success_rate = metrics["perf/success_rate"]  # (..., num_updates)
    num_updates = success_rate.shape[-1]
    tail = max(1, num_updates // 5)
    return jnp.mean(success_rate[..., -tail:])


@dataclass
class Benchmark:
    name: str
    budget: int
    """Overrides `agent.hparams.budget` for every environment this
    benchmark runs - the one knob `NAVIX_1M`/`NAVIX_100K` differ on."""
    env_ids: Tuple[str, ...] = field(default_factory=lambda: tuple(registry().keys()))
    """Defaults to every registered environment (`nx.registry()`),
    matching what `benchmarks/ppo.py`/`dreamer.py` already do - resolved
    lazily (not at import time) so it reflects the full registry
    regardless of import order."""
    seeds: Tuple[int, ...] = (0,)
    score_fn: Callable[[Dict[str, Array]], Array] = default_score_fn

    def run(self, entry: AlgorithmEntry, log_to_wandb: bool = False) -> BenchmarkResult:
        """Runs `entry` against every environment in `self.env_ids`, at
        `self.budget`, and scores each with `self.score_fn`."""
        scores: Dict[str, Array] = {}
        logs_by_env: Dict[str, Dict] = {}

        for env_id in self.env_ids:
            env = make(env_id)
            agent = entry.agent_factory(env)
            agent = agent.replace(hparams=agent.hparams.replace(budget=self.budget))

            experiment = Experiment(
                name=self.name,
                agent=agent,
                env=env,
                env_id=env_id,
                seeds=self.seeds,
            )
            _, logs = experiment.run(log_to_wandb=log_to_wandb)

            scores[env_id] = self.score_fn(logs)
            logs_by_env[env_id] = logs

        score = jnp.mean(jnp.stack(list(scores.values())))
        return BenchmarkResult(entry=entry, scores=scores, score=score, logs=logs_by_env)


NAVIX_1M = Benchmark(name="navix-1m", budget=1_000_000)
"""From-scratch training, every registered environment, 1M-frame budget
per environment - the standard budget `PPOHparams`/`PQNHparams` already
default to."""

NAVIX_100K = Benchmark(name="navix-100k", budget=100_000)
"""Same as `NAVIX_1M`, at a 100K-frame budget - a cheaper preset for
quick iteration/CI-scale checks, not intended as a low-sample-efficiency
benchmark in the Atari-100k sense (navix's environments are cheap enough
that "100K" here is about wall-clock cost, not measuring sample
efficiency specifically)."""
