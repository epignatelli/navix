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

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, ClassVar, Dict, Tuple
from urllib.parse import urlparse

from jax import Array
import jax.numpy as jnp

from ..agents.agent import Agent
from ..environments.environment import Environment
from ..environments.registry import make, registry
from ..experiment import Experiment
from ..plotting import derive_scalar_metrics

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


def _aggregate(logs_by_env: Dict[str, Dict], key: str) -> Array:
    """Reduces one `navix.plotting.MANDATORY_METRICS` key to a single
    scalar across an entire `Benchmark` run: per environment, the mean
    over the last fifth of training (the "last20%" convergence
    convention this project's own performance investigations already use
    by hand); then meaned across environments. Works uniformly for both
    the derived `perf/*` keys and the `iter/fps`/`iter/wall_time` keys
    already in `logs` verbatim - the latter are constant across a run's
    own updates, so "last-fifth mean" just returns that same constant."""
    per_env = []
    for logs in logs_by_env.values():
        metrics = derive_scalar_metrics(logs)
        value = metrics[key]  # (..., num_updates)
        num_updates = value.shape[-1]
        tail = max(1, num_updates // 5)
        per_env.append(jnp.mean(value[..., -tail:]))
    return jnp.mean(jnp.stack(per_env))


@dataclass
class BenchmarkResult:
    """The outcome of running one `AlgorithmEntry` against a `Benchmark`.
    Shaped the same for every algorithm (only the *values* differ) -
    exactly `navix.plotting.MANDATORY_METRICS`, aggregated across every
    environment the benchmark covers, which issue #130's leaderboard
    spec already settled on as the standard, algorithm-agnostic
    performance/cost columns. Not something `Benchmark` subclasses are
    expected to customise: unlike `env_ids`/`budget`, this is the one
    piece meant to stay fixed so results are comparable across
    benchmarks, not just within one."""

    entry: AlgorithmEntry
    """Echoes back the entry that was run, so a result is self-describing
    without needing to be paired back up with its `AlgorithmEntry`."""
    returns: Array
    success_rate: Array
    episode_length: Array
    fps: Array
    wall_time: Array
    logs: Dict[str, Dict]
    """Per-environment raw `logs`, keyed by `env_id` - the same pytree
    `Experiment.run` returns for that environment. The five fields above
    are already reduced from this; keep `logs` around for diagnostics/
    plotting via `navix.plotting`, or to compute some other reduction of
    the same underlying per-env, per-update data."""

    @classmethod
    def from_logs(cls, entry: AlgorithmEntry, logs_by_env: Dict[str, Dict]) -> BenchmarkResult:
        return cls(
            entry=entry,
            returns=_aggregate(logs_by_env, "perf/returns"),
            success_rate=_aggregate(logs_by_env, "perf/success_rate"),
            episode_length=_aggregate(logs_by_env, "perf/episode_length"),
            fps=_aggregate(logs_by_env, "iter/fps"),
            wall_time=_aggregate(logs_by_env, "iter/wall_time"),
            logs=logs_by_env,
        )


@dataclass
class Benchmark:
    """Base class for a benchmark preset. Not instantiated directly -
    subclasses (`Navix1M`, `Navix100K`, `Navix1K`, below) fix
    `name`/`budget` as class attributes, so a preset is used as
    `Navix1M(entry).run()`: bind the algorithm to score, then run."""

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
        env = make(env_id)
        agent = self.entry.agent_factory(env)
        agent = agent.replace(hparams=agent.hparams.replace(budget=self.budget))

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
        for env_id in self.env_ids:
            logs_by_env[env_id] = self._train_on(env_id, log_to_wandb)
        return BenchmarkResult.from_logs(self.entry, logs_by_env)


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


class Navix1K(Benchmark):
    """Same as `Navix1M`/`Navix100K`, at a small budget - too small for
    any algorithm to actually converge, so this isn't a performance
    preset. Useful as a fast, cheap smoke test that an `AlgorithmEntry`'s
    `agent_factory` runs end-to-end (compiles, trains, scores) against
    every registered environment without waiting for a real budget.

    4096, not 1000 (despite the "1K" name, kept for the size ordering
    with `Navix100K`/`Navix1M`): `num_updates = budget // (num_steps *
    num_envs)` must be >= 1 for training to run at all, and every navix
    agent shipped so far (`PPOHparams`/`DreamerHparams`/`PQNHparams`)
    defaults to `num_envs=16, num_steps=128` = 2048 frames/update. A
    budget below that silently "succeeds" with zero training updates -
    every logged array shaped `(0, ...)` - which defeats the point of a
    smoke test entirely. 4096 clears that floor with margin for a
    future agent needing more."""

    name = "NAVIX-1k"
    budget = 4_096
