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

"""A `Benchmark` is a preset experimental setup (fixed environments,
fixed training budget) that scores an algorithm - an orchestration
layer over `Experiment` (navix/experiment.py), one `Experiment` per
environment with `budget` overridden.

Implements the "Benchmark" concept from issue #130 (the navix
leaderboard proposal), for the from-scratch-training protocol only, as
two presets (`Navix1M`, `Navix100K`) differing in `budget`.

`Benchmark` is a base class: `name`/`budget` are class attributes fixed
per preset, so a preset is used as `Navix1M(entry).run()`.

Each algorithm is wrapped in an `AlgorithmEntry` - the provenance
metadata #130 requires of a leaderboard row (name, GitHub-validated
author, full commit URLs for both navix and the algorithm's own repo,
paper link). Per #130's "never vendor an external algorithm's code"
rule, `AlgorithmEntry.agent_factory` is how `Benchmark` stays algorithm-
agnostic.

`BenchmarkResult` has a `summary: Metrics` (one reduced scalar per
field, meaned across environments) and a `history: Dict[str, Metrics]`
(one `Metrics` per env_id, to extract training curves) - this shape is
only valid for the from-scratch-per-environment protocol implemented
here; a future continual-learning or one-shot-generalisation protocol
won't have an "environment" axis to key `history` on and would need
its own result type. `Metrics` = `returns`/`episode_length`/`fps`/
`wall_time` (per-update curves in `history`, reduced in `summary`) and
`flops`/`memory_bytes`/`compile_time_seconds` (from
`Agent.cost_analysis` - a single measurement, not a curve, so these
stay scalar in both)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, ClassVar, Dict, Iterable, Tuple
from urllib.parse import urlparse

import jax
from jax import Array
import jax.numpy as jnp

from .agents.agent import Agent, CostAnalysis
from .environments.environment import Environment
from .environments.registry import make, registry
from .experiment import Experiment
from .plotting import derive_scalar_metrics

# GitHub username: alphanumeric/single hyphens, max 39 chars.
_GITHUB_HANDLE_RE = re.compile(r"^[a-zA-Z\d](?:[a-zA-Z\d]|-(?=[a-zA-Z\d])){0,38}$")
# Git commit SHA (abbreviated or full): lowercase hex, 7-40 chars.
_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")


def _is_commit_url(url: str) -> bool:
    """A bare SHA doesn't say which repo it's a commit of - full URLs
    are self-describing and directly clickable."""
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
    """This implementation's author (GitHub handle), not the paper's.
    Validated in `__post_init__`."""
    paper_url: str
    """Link to the paper the algorithm is from."""
    navix_commit_url: str
    """Link to the navix commit this result was produced against
    (issue #130's `navix.sha`, as a URL). Validated in
    `__post_init__`."""
    algorithm_commit_url: str
    """Link to the algorithm implementation's own commit (issue #130's
    `agent.sha`, as a URL) - same commit as `navix_commit_url` for a
    navix-shipped agent, a different repo's commit for an external one.
    Validated in `__post_init__`."""
    agent_factory: Callable[[Environment], Agent]
    """Builds a fresh, env-shaped `Agent` for a given `Environment` -
    called once per environment (agents need per-env-shaped
    networks)."""

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


def _last_fifth_mean(value: Array) -> Array:
    """Mean over the last fifth of training ("last20%" convergence
    convention). A scalar (no update axis, e.g. a cost field) passes
    through unchanged - there's nothing to reduce."""
    value = jnp.asarray(value)
    if value.ndim == 0:
        return value
    tail = max(1, value.shape[-1] // 5)
    return jnp.mean(value[..., -tail:])


@dataclass
class Metrics:
    """Same fields for both `BenchmarkResult.summary` (one reduced
    scalar per field) and each `history` entry (one per-update curve
    per field) - except `flops`/`memory_bytes`/`compile_time_seconds`,
    which are always scalar: `Agent.cost_analysis` is a single
    measurement, not something with a training curve."""

    returns: Array
    episode_length: Array
    flops: Array
    """From `Agent.cost_analysis`. Always scalar."""
    memory_bytes: Array
    """Always scalar."""
    compile_time_seconds: Array
    """Always scalar."""
    fps: Array
    """Hardware-dependent - only meaningful across results on the same
    hardware."""
    wall_time: Array
    """Hardware-dependent - only meaningful across results on the same
    hardware."""

    @classmethod
    def from_logs_and_cost(cls, logs: Dict, cost: CostAnalysis) -> Metrics:
        """Full per-update curves for `returns`/`episode_length`/`fps`/
        `wall_time` - use `.last_fifth_mean()` to reduce to a scalar."""
        scalars = derive_scalar_metrics(logs)
        return cls(
            returns=scalars["perf/returns"],
            episode_length=scalars["perf/episode_length"],
            flops=jnp.asarray(cost.flops),
            memory_bytes=jnp.asarray(cost.memory_bytes),
            compile_time_seconds=jnp.asarray(cost.compile_time_seconds),
            fps=jnp.asarray(logs["iter/fps"]),
            wall_time=jnp.asarray(logs["iter/wall_time"]),
        )

    def last_fifth_mean(self) -> Metrics:
        """Reduces every field's last-fifth-of-training mean - scalar
        fields (the cost ones) pass through unchanged."""
        return Metrics(
            **{
                field_name: _last_fifth_mean(getattr(self, field_name))
                for field_name in Metrics.__dataclass_fields__
            }
        )

    @classmethod
    def mean(cls, values: Iterable[Metrics]) -> Metrics:
        """One `Metrics`, every field meaned across the given values."""
        values = list(values)
        return cls(
            **{
                field_name: jnp.mean(
                    jnp.stack([jnp.asarray(getattr(m, field_name)) for m in values])
                )
                for field_name in Metrics.__dataclass_fields__
            }
        )


@dataclass
class BenchmarkResult:
    """The outcome of running one `AlgorithmEntry` against a
    `Benchmark`. Valid only for the from-scratch-per-environment
    protocol implemented by `Benchmark` - a future continual-learning
    or one-shot-generalisation protocol needs its own result type,
    since `history` assumes one independent `Metrics` per env_id."""

    entry: AlgorithmEntry
    """Echoes back the entry that was run."""
    summary: Metrics
    """Each field's last-fifth-mean, meaned across every environment."""
    history: Dict[str, Metrics]
    """Full per-update curves per `env_id` - reduce with
    `.last_fifth_mean()` for a per-env scalar, or plot directly."""

    @classmethod
    def from_logs(
        cls,
        entry: AlgorithmEntry,
        logs_by_env: Dict[str, Dict],
        cost_by_env: Dict[str, CostAnalysis],
    ) -> BenchmarkResult:
        history = {
            env_id: Metrics.from_logs_and_cost(logs_by_env[env_id], cost_by_env[env_id])
            for env_id in logs_by_env
        }
        summary = Metrics.mean([m.last_fifth_mean() for m in history.values()])
        return cls(entry=entry, summary=summary, history=history)


@dataclass
class Benchmark:
    """Base class for a benchmark preset. Subclasses (`Navix1M`,
    `Navix100K`, below) fix `name`/`budget` as class attributes; used
    as `Navix1M(entry).run()`."""

    entry: AlgorithmEntry
    env_ids: Tuple[str, ...] = field(default_factory=lambda: tuple(registry().keys()))
    """Defaults to every registered environment - resolved lazily so it
    reflects the full registry regardless of import order."""
    seeds: Tuple[int, ...] = (0,)

    name: ClassVar[str] = ""
    budget: ClassVar[int] = 0
    """Overrides `agent.hparams.budget` - the one thing the presets
    below differ on."""

    def _build_agent(self, env_id: str) -> Tuple[Environment, Agent]:
        env = make(env_id)
        agent = self.entry.agent_factory(env)
        agent = agent.replace(hparams=agent.hparams.replace(budget=self.budget))
        return env, agent

    def _train_on(self, env_id: str, log_to_wandb: bool) -> Dict:
        """Builds and trains one environment's agent, returning its
        `logs`. Separate from `run` so a future sequential protocol
        (curriculum/open-ended learning) can override just this method,
        not scoring/aggregation too."""
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
        """Runs `self.entry` against every env in `self.env_ids`, at
        `self.budget`."""
        logs_by_env: Dict[str, Dict] = {}
        cost_by_env: Dict[str, CostAnalysis] = {}
        for env_id in self.env_ids:
            logs_by_env[env_id] = self._train_on(env_id, log_to_wandb)
            # Fresh agent, not _train_on's trained one - cost_analysis
            # measures the program's shape, not trained weights.
            _, agent = self._build_agent(env_id)
            cost_by_env[env_id] = agent.cost_analysis(jax.random.PRNGKey(self.seeds[0]))
        return BenchmarkResult.from_logs(self.entry, logs_by_env, cost_by_env)


class Navix1M(Benchmark):
    """1M-frame budget per environment - `PPOHparams`/`PQNHparams`'
    own default."""

    name = "NAVIX-1m"
    budget = 1_000_000


class Navix100K(Benchmark):
    """Same as `Navix1M`, at 100K frames - a cheaper preset for quick
    checks."""

    name = "NAVIX-100k"
    budget = 100_000
