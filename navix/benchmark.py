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
environments, a fixed training budget - that scores an algorithm
against it, rather than the single-environment, single-config runs
`Experiment` (`navix/experiment.py`) already supports. `Benchmark.run`
is an orchestration layer *over* `Experiment` (one `Experiment` per
environment, budget-overridden), not a replacement for it.

This is the "Benchmark" issue #130 (the navix leaderboard proposal)
already scoped conceptually - "a scenario-specific function that takes a
trained agent and returns the metric(s) used to rank it" - implemented
here for the simplest of #130's four protocols, from-scratch training,
as two presets (`Navix1M`, `Navix100K`) that differ only in `budget`.

`Benchmark` itself is a base class, not something instantiated directly:
`name`/`budget` are fixed per preset (each preset sets them as class
attributes), while `entry`/`env_ids`/`seeds` are set per run - so a
preset is used as `Navix1M(entry).run()`, binding the algorithm to score
before running it.

Every algorithm a `Benchmark` runs is wrapped in an `AlgorithmEntry`,
carrying the provenance metadata #130's "Structure (decided)" section
requires of a leaderboard row - name, a GitHub-handle-validated author,
and full commit URLs (not bare SHAs, so they're directly traceable and
self-describing about which repo they belong to) for both the navix
commit and the algorithm implementation's own commit the result was
produced against - plus a link to the paper.

Per #130's "we never vendor an external algorithm's code" decision,
`AlgorithmEntry.agent_factory` is how a `Benchmark` stays algorithm-
agnostic without navix owning the algorithm's implementation.

What comes back is a `BenchmarkResult`: the same `Metrics` shape twice
over - once as `overall` (meaned across every environment the benchmark
covers) and once per environment in `per_environment` - rather than an
aggregate-only result with per-environment detail buried in raw logs
you'd have to re-derive metrics from yourself. `Metrics` itself is
`returns`/`episode_length` (from training), `flops`/`memory_bytes`/
`compile_time_seconds` (from `Agent.cost_analysis` - see its own
docstring for exactly what these measure and why), and `fps`/
`wall_time` (registered because they're useful for comparing runs
executed on the same hardware, even though - unlike every other field
here - they aren't hardware-independent, so cross-hardware comparisons
of just these two are not meaningful)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, ClassVar, Dict, Tuple
from urllib.parse import urlparse

import jax
from jax import Array
import jax.numpy as jnp

from .agents.agent import Agent, CostAnalysis
from .environments.environment import Environment
from .environments.registry import make, registry
from .experiment import Experiment
from .plotting import derive_scalar_metrics

# GitHub username rules: alphanumeric or single hyphens, no leading/
# trailing/doubled hyphen, max 39 characters.
_GITHUB_HANDLE_RE = re.compile(r"^[a-zA-Z\d](?:[a-zA-Z\d]|-(?=[a-zA-Z\d])){0,38}$")
# A git commit SHA (abbreviated or full) is lowercase hex, 7-40 chars -
# `git rev-parse HEAD` always produces the full 40, but 7 is accepted
# too since that's the common abbreviated form elsewhere. Matched
# against a commit URL's final path segment, e.g.
# https://github.com/epignatelli/navix/commit/<sha>.
_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")


def _is_commit_url(url: str) -> bool:
    """A bare SHA doesn't say which repo it's a commit of - only useful
    for navix's own commits, where the repo is implied. An external
    algorithm's `algorithm_commit_url` has no such implied repo, so both
    commit fields are full URLs: self-describing (repo + commit) and
    directly clickable for reproduction/tracing, not just a hex string
    needing external context to resolve."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return False
    last_segment = parsed.path.rstrip("/").rsplit("/", 1)[-1]
    return bool(_SHA_RE.match(last_segment))


@dataclass
class AlgorithmEntry:
    """One algorithm to score against a `Benchmark`, plus the provenance
    metadata every navix leaderboard entry requires (issue #130)."""

    name: str
    """Algorithm name, e.g. "PPO"."""
    author: str
    """Author of *this implementation* (GitHub handle) - distinct from
    the paper's authors, since a paper can have multiple independent
    implementations (navix's own, rejax's, a submitter's own). Validated
    against GitHub's own username rules in `__post_init__`."""
    paper_url: str
    """Link to the paper the algorithm is from."""
    navix_commit_url: str
    """Link to the navix commit this result was produced against (e.g.
    `https://github.com/epignatelli/navix/commit/<sha>`) - env/reward/
    termination semantics drift over time, so a result is only
    meaningful pinned to a specific navix commit, not just "navix" in
    general (issue #130's `navix.sha` field, as a URL rather than a bare
    SHA so it's directly traceable). Validated as a well-formed
    commit URL in `__post_init__`."""
    algorithm_commit_url: str
    """Link to the commit of *the algorithm implementation's own repo*
    that produced this result (issue #130's `agent.sha` field, as a URL)
    - for a navix-shipped agent (PPO/Dreamer/PQN), the implementation
    lives in the navix repo itself, so this points at the same commit as
    `navix_commit_url`; for an external algorithm (e.g. one driving
    rejax as a dependency), this points at that other repo's own commit
    - which a bare SHA alone couldn't identify, since it doesn't say
    which repo it belongs to. Validated as a well-formed commit URL in
    `__post_init__`."""
    agent_factory: Callable[[Environment], Agent]
    """Builds a fresh, env-shaped `Agent` for the given `Environment` -
    called once per environment in the benchmark, the same way
    `benchmarks/<preset>/<entry>/run.py` build a fresh agent per env
    inside their own loops today. Needed because agents need per-env-
    shaped networks (action_dim/obs_dim), not because state is shared
    across environments."""

    def __post_init__(self) -> None:
        if not _GITHUB_HANDLE_RE.match(self.author):
            raise ValueError(
                f"AlgorithmEntry.author {self.author!r} is not a valid GitHub "
                "handle (alphanumeric/single hyphens, no leading/trailing/"
                "doubled hyphen, max 39 characters)."
            )
        for field_name in ("navix_commit_url", "algorithm_commit_url"):
            value = getattr(self, field_name)
            if not _is_commit_url(value):
                raise ValueError(
                    f"AlgorithmEntry.{field_name} {value!r} is not a valid commit "
                    "URL (expected an http(s) URL ending in a 7-40 character "
                    "lowercase hex commit SHA)."
                )


def _last_fifth_mean(logs: Dict, key: str) -> Array:
    """The mean of `logs[key]` over the last fifth of training (the
    "last20%" convergence convention this project's own performance
    investigations already use by hand) - `logs[key]` has shape
    `(..., num_updates)`; `iter/fps`/`iter/wall_time` are constant across
    a run's own updates, so this just returns that same constant for
    those two."""
    value = logs[key]
    num_updates = value.shape[-1]
    tail = max(1, num_updates // 5)
    return jnp.mean(value[..., -tail:])


@dataclass
class Metrics:
    """The fixed, algorithm-agnostic metric set `BenchmarkResult` uses
    for both `overall` and each entry of `per_environment` - the same
    shape either way, so "how did it do overall" and "how did it do on
    environment X" are directly comparable, not two different kinds of
    object."""

    returns: Array
    episode_length: Array
    flops: Array
    """From `Agent.cost_analysis` - see that method's docstring for
    exactly what's measured and its caveats."""
    memory_bytes: Array
    compile_time_seconds: Array
    fps: Array
    """Hardware-dependent, unlike every field above - only meaningful
    compared across results produced on the same hardware."""
    wall_time: Array
    """Hardware-dependent, unlike every field above - only meaningful
    compared across results produced on the same hardware."""

    @classmethod
    def from_logs_and_cost(cls, logs: Dict, cost: CostAnalysis) -> Metrics:
        return cls(
            returns=_last_fifth_mean(derive_scalar_metrics(logs), "perf/returns"),
            episode_length=_last_fifth_mean(
                derive_scalar_metrics(logs), "perf/episode_length"
            ),
            flops=jnp.asarray(cost.flops),
            memory_bytes=jnp.asarray(cost.memory_bytes),
            compile_time_seconds=jnp.asarray(cost.compile_time_seconds),
            fps=_last_fifth_mean(logs, "iter/fps"),
            wall_time=_last_fifth_mean(logs, "iter/wall_time"),
        )

    @classmethod
    def mean(cls, per_env: Dict[str, Metrics]) -> Metrics:
        """Reduces one `Metrics` per environment down to a single
        `Metrics` - every field meaned across environments, the
        `overall` a `BenchmarkResult` reports."""
        values = list(per_env.values())
        return cls(
            **{
                field_name: jnp.mean(jnp.stack([getattr(m, field_name) for m in values]))
                for field_name in Metrics.__dataclass_fields__
            }
        )


@dataclass
class BenchmarkResult:
    """The outcome of running one `AlgorithmEntry` against a
    `Benchmark`."""

    entry: AlgorithmEntry
    """Echoes back the entry that was run, so a result is self-describing
    without needing to be paired back up with its `AlgorithmEntry`."""
    overall: Metrics
    """`Metrics` meaned across every environment in the benchmark."""
    per_environment: Dict[str, Metrics]
    """`Metrics` for each `env_id`, individually - not just the raw
    `logs` those metrics were derived from."""
    logs: Dict[str, Dict]
    """Per-environment raw `logs`, keyed by `env_id` - the same pytree
    `Experiment.run` returns for that environment. `overall`/
    `per_environment` are already reduced from this; keep `logs` around
    for diagnostics/plotting via `navix.plotting`, or to compute some
    other reduction of the same underlying per-update data."""

    @classmethod
    def from_logs(
        cls,
        entry: AlgorithmEntry,
        logs_by_env: Dict[str, Dict],
        cost_by_env: Dict[str, CostAnalysis],
    ) -> BenchmarkResult:
        per_environment = {
            env_id: Metrics.from_logs_and_cost(logs_by_env[env_id], cost_by_env[env_id])
            for env_id in logs_by_env
        }
        return cls(
            entry=entry,
            overall=Metrics.mean(per_environment),
            per_environment=per_environment,
            logs=logs_by_env,
        )


@dataclass
class Benchmark:
    """Base class for a benchmark preset. Not instantiated directly -
    subclasses (`Navix1M`, `Navix100K`, below) fix `name`/`budget` as
    class attributes, so a preset is used as `Navix1M(entry).run()`:
    bind the algorithm to score, then run."""

    entry: AlgorithmEntry
    env_ids: Tuple[str, ...] = field(default_factory=lambda: tuple(registry().keys()))
    """Defaults to every registered environment (`nx.registry()`),
    matching what `benchmarks/<preset>/*.py` already do - resolved
    lazily (not at import time) so it reflects the full registry
    regardless of import order."""
    seeds: Tuple[int, ...] = (0,)

    name: ClassVar[str] = ""
    budget: ClassVar[int] = 0
    """Overrides `agent.hparams.budget` for every environment this
    benchmark runs - the one thing the presets below differ on."""

    def _build_agent(self, env_id: str) -> Tuple[Environment, Agent]:
        env = make(env_id)
        agent = self.entry.agent_factory(env)
        agent = agent.replace(hparams=agent.hparams.replace(budget=self.budget))
        return env, agent

    def _train_on(self, env_id: str, log_to_wandb: bool) -> Dict:
        """Builds and trains one environment's agent from `self.entry`,
        returning its `logs`. Pulled out of `run` as its own method so a
        future protocol that needs to run environments as a *sequence*
        rather than independently - e.g. curriculum learning (train on
        env A, then keep training the same params on env B) or open-
        ended learning (an unbounded/evolving task stream) - only needs
        to override this one method (how a single environment gets
        trained, e.g. warm-starting from a previous env's result) or
        `run` itself (how environments are sequenced), not reimplement
        scoring/aggregation too. `env_ids`/`BenchmarkResult` are already
        shaped for this: both are plain per-`env_id` dicts, agnostic to
        whether the environments were trained independently or in
        sequence."""
        env, agent = self._build_agent(env_id)
        experiment = Experiment(
            name=self.name,
            agent=agent,
            env=env,
            env_id=env_id,
            seeds=self.seeds,
        )
        _, logs = experiment.run(log_to_wandb=log_to_wandb)
        return logs

    def run(self, log_to_wandb: bool = False) -> BenchmarkResult:
        """Runs `self.entry` against every environment in
        `self.env_ids`, at `self.budget`, and reduces the result to
        `BenchmarkResult`'s fixed, algorithm-agnostic metric set."""
        logs_by_env: Dict[str, Dict] = {}
        cost_by_env: Dict[str, CostAnalysis] = {}
        for env_id in self.env_ids:
            logs_by_env[env_id] = self._train_on(env_id, log_to_wandb)
            # A fresh agent instance, not the trained one _train_on
            # produced - cost_analysis measures the training program's
            # shape/compute, not anything specific to trained weights.
            _, agent = self._build_agent(env_id)
            cost_by_env[env_id] = agent.cost_analysis(jax.random.PRNGKey(self.seeds[0]))
        return BenchmarkResult.from_logs(self.entry, logs_by_env, cost_by_env)


class Navix1M(Benchmark):
    """From-scratch training, every registered environment, 1M-frame
    budget per environment - the standard budget `PPOHparams`/
    `PQNHparams` already default to."""

    name = "NAVIX-1m"
    budget = 1_000_000


class Navix100K(Benchmark):
    """Same as `Navix1M`, at a 100K-frame budget - a cheaper preset for
    quick iteration/CI-scale checks, not intended as a low-sample-
    efficiency benchmark in the Atari-100k sense (navix's environments
    are cheap enough that "100K" here is about wall-clock cost, not
    measuring sample efficiency specifically)."""

    name = "NAVIX-100k"
    budget = 100_000
