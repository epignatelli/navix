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

"""A `Benchmark` is an experimentation protocol - independent of any
one algorithm: an orchestration layer over `Experiment`
(navix/experiment.py). It does not hold the algorithm being scored;
that's an argument to `run`, so the same `Benchmark` (or preset) can
score many algorithms.

Implements the "Benchmark" concept from issue #130 (the navix
leaderboard proposal). `Benchmark` itself only owns what's truly
protocol-agnostic - building an agent for one environment
(`_build_agent`) and training it (`_train_on`). Everything about *how
results are shaped* - what counts as a per-unit result, how per-unit
results combine into an aggregate, whether training is even
independent per environment - is a template method a protocol
subclass overrides: `_score_env`, `_summarize`, and `run` itself.

This module implements exactly one protocol concretely: a flat,
order-independent list of environments, each trained from scratch with
no transfer between them (`Navix1M`/`Navix100K`, differing only in
`budget`). Other protocols issue #130 anticipates - curriculum
learning (a fixed stage sequence, graduating per stage), continual
learning (one agent through a task sequence, tracking retention/
forgetting via an R-matrix of task x task performance), open-ended
learning (an adaptive generator, scored by periodic zero-shot
performance on a fixed held-out suite) - have result shapes with
nothing in common with this one or each other, so they're expected to
subclass `Benchmark` and return their own result type rather than
conform to `BenchmarkResult`.

`Benchmark` never needs instantiating: `name`/`budget`/`env_ids`/
`seeds` are class attributes fixed per preset, and `run` is a
classmethod, so a preset is used directly as `Navix1M.run(entry)` -
`entry` is the only thing that varies across runs.

Each algorithm scored is wrapped in an `AlgorithmEntry` - the
provenance metadata #130 requires of a leaderboard row (name, GitHub-
validated author, full commit URLs for both navix and the algorithm's
own repo, paper link). Per #130's "never vendor an external
algorithm's code" rule, `AlgorithmEntry.agent_factory` is how `run`
stays algorithm-agnostic.

`BenchmarkResult` (this protocol's own result type) is self-
referential and carries three views, everything a leaderboard needs:
`summary` (the standardized, comparable-across-algorithms fields -
`returns`/`episode_length`/`fps`/`wall_time`/`flops`/`memory_bytes`/
`compile_time_seconds` - full per-update curves on a leaf, reduced
last-fifth-mean scalars on the aggregate), `history` (one leaf per
`env_id`, for drill-down - empty on a leaf), and `detail` (the
complete raw per-update log for one environment, algorithm-specific
diagnostics included, not just the standardized subset - empty on the
aggregate, since detail is inherently per-environment)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Callable, ClassVar, Dict, Optional, Tuple
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


# The array-valued fields that get reduced (last_fifth_mean) and
# aggregated (summarize) - excludes entry/history, which aren't arrays.
_METRIC_FIELDS: Tuple[str, ...] = (
    "returns",
    "episode_length",
    "flops",
    "memory_bytes",
    "compile_time_seconds",
    "fps",
    "wall_time",
)


@dataclass
class BenchmarkResult:
    """Result shape for `Benchmark`'s flat, order-independent env-list
    protocol (`Navix1M`/`Navix100K`) - not shared with other protocols
    (see module docstring). Self-referential: this IS both one
    environment's leaf result (full curves, `detail` populated,
    `history={}`) and the aggregate `Benchmark.run` returns (`summary`
    fields reduced and meaned, `history` populated with one leaf per
    `env_id`, `detail={}`)."""

    entry: AlgorithmEntry
    """Echoes back the entry that was run."""
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
    history: Dict[str, BenchmarkResult] = field(default_factory=dict)
    """One leaf per `env_id`, for drill-down - empty on a leaf itself
    (this result IS that leaf); populated only on what `Benchmark.run`
    returns."""
    detail: Dict = field(default_factory=dict)
    """The complete raw per-update log for this environment - every
    key `Experiment.run` produced, including algorithm-specific
    diagnostics (e.g. `loss/actor_loss`) that aren't standardized or
    comparable across different algorithms, unlike `summary`'s fields.
    Populated on a leaf; empty on the aggregate, since detail is
    inherently per-environment."""

    def last_fifth_mean(self) -> BenchmarkResult:
        """Reduces this result's own `summary` fields to their last-
        fifth-of-training mean - `history`/`detail` are left
        untouched."""
        return replace(
            self,
            **{
                field_name: _last_fifth_mean(getattr(self, field_name))
                for field_name in _METRIC_FIELDS
            },
        )


class Benchmark:
    """A benchmark preset - a fixed protocol, not tied to any one
    algorithm and not needing an instance: `name`/`budget` are class
    attributes (subclasses `Navix1M`/`Navix100K`, below, fix them),
    and `run` is a classmethod, so a preset is used directly as
    `Navix1M.run(entry)`. `env_ids`/`seeds` follow the same pattern -
    override for one call via `run(entry, env_ids=...)`, or for a
    reusable custom preset via subclassing, the same way `name`/
    `budget` are overridden.

    Only `_build_agent`/`_train_on` are truly protocol-agnostic - every
    protocol trains *some* agent on *some* environment the same way.
    `_score_env`/`_summarize`/`run` are template methods this class
    implements for the flat, order-independent env-list protocol; a
    different protocol (curriculum/continual/open-ended learning - see
    module docstring) overrides them and returns its own result type
    instead of `BenchmarkResult`."""

    name: ClassVar[str] = ""
    budget: ClassVar[int] = 0
    """Overrides `agent.hparams.budget` - the one thing `Navix1M`/
    `Navix100K` differ on."""
    env_ids: ClassVar[Optional[Tuple[str, ...]]] = None
    """`None` resolves lazily, at `run` time, to every registered
    environment - not resolved at class-definition time, so it
    reflects the full registry regardless of import order."""
    seeds: ClassVar[Tuple[int, ...]] = (0,)

    @classmethod
    def _build_agent(cls, entry: AlgorithmEntry, env_id: str) -> Tuple[Environment, Agent]:
        env = make(env_id)
        agent = entry.agent_factory(env)
        agent = agent.replace(hparams=agent.hparams.replace(budget=cls.budget))
        return env, agent

    @classmethod
    def _train_on(
        cls, entry: AlgorithmEntry, env_id: str, seeds: Tuple[int, ...], log_to_wandb: bool
    ) -> Tuple[Agent, Dict]:
        """Builds and trains one environment's agent, returning the
        pre-training `Agent` alongside its `logs` - `Agent` is an
        immutable pytree, so `experiment.run` below never mutates it;
        the caller can reuse it for `cost_analysis` (which only depends
        on shapes, not trained weight values) without building a second
        one."""
        env, agent = cls._build_agent(entry, env_id)
        experiment = Experiment(
            name=cls.name,
            agent=agent,
            env=env,
            env_id=env_id,
            seeds=seeds,
        )
        _, logs = experiment.run(log_to_wandb=log_to_wandb)
        return agent, logs

    @classmethod
    def _score_env(cls, entry: AlgorithmEntry, logs: Dict, cost: CostAnalysis) -> BenchmarkResult:
        """Builds one environment's leaf result: full comparable-metric
        curves (`summary`), this agent's `cost_analysis`, and `detail`
        - the complete raw `logs` dict, unfiltered. Override to change
        what "one environment's result" captures for a different
        protocol."""
        scalars = derive_scalar_metrics(logs)
        return BenchmarkResult(
            entry=entry,
            returns=scalars["perf/returns"],
            episode_length=scalars["perf/episode_length"],
            flops=jnp.asarray(cost.flops),
            memory_bytes=jnp.asarray(cost.memory_bytes),
            compile_time_seconds=jnp.asarray(cost.compile_time_seconds),
            fps=jnp.asarray(logs["iter/fps"]),
            wall_time=jnp.asarray(logs["iter/wall_time"]),
            detail=logs,
        )

    @classmethod
    def _summarize(
        cls, entry: AlgorithmEntry, per_env: Dict[str, BenchmarkResult]
    ) -> BenchmarkResult:
        """Combines per-env leaf results into the aggregate `run`
        returns: each `summary` field is its last-fifth-mean, meaned
        across environments; `history` keeps the untouched leaves
        (curves + detail) for drill-down. Override to change how a
        protocol combines its own per-unit results - e.g. a continual-
        learning subclass would build an R-matrix here instead of a
        flat mean."""
        reduced = [result.last_fifth_mean() for result in per_env.values()]
        return BenchmarkResult(
            entry=entry,
            **{
                field_name: jnp.mean(
                    jnp.stack([jnp.asarray(getattr(m, field_name)) for m in reduced])
                )
                for field_name in _METRIC_FIELDS
            },
            history=per_env,
        )

    @classmethod
    def run(
        cls,
        entry: AlgorithmEntry,
        log_to_wandb: bool = False,
        env_ids: Optional[Tuple[str, ...]] = None,
        seeds: Optional[Tuple[int, ...]] = None,
    ) -> BenchmarkResult:
        """Runs `entry` against `env_ids` (default: every registered
        environment) independently, at `cls.budget` - no ordering, no
        transfer assumed between environments. A sequential protocol
        (curriculum/continual learning) that needs one environment's
        outcome to affect the next overrides `run` itself, not just
        `_score_env`/`_summarize`."""
        env_ids = env_ids if env_ids is not None else cls.env_ids or tuple(registry().keys())
        seeds = seeds if seeds is not None else cls.seeds
        per_env: Dict[str, BenchmarkResult] = {}
        for env_id in env_ids:
            agent, logs = cls._train_on(entry, env_id, seeds, log_to_wandb)
            cost = agent.cost_analysis(jax.random.PRNGKey(seeds[0]))
            per_env[env_id] = cls._score_env(entry, logs, cost)
        return cls._summarize(entry, per_env)


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
