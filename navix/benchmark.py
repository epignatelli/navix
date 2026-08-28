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
leaderboard proposal). `Benchmark` is abstract: `build_agent`/
`train_on` build *some* agent for *some* environment and train it via
`Experiment`, the same way for every protocol, so they're implemented
here. `run` (what gets trained, and how raw `BenchmarkResult`s get
produced and collected), `summary` (the single comparable score a
leaderboard's table shows for one algorithm entry), and `details` (the
content a leaderboard shows for that entry beyond its `summary` row -
training curves, a retention matrix, a generated-level log, depending
on the protocol) have no protocol-agnostic implementation, so all
three raise `NotImplementedError` - a concrete protocol (see
`FromScratchBenchmark`, below) overrides them. A caller interprets a
`Benchmark.run()` call's raw output by calling `summary`/`details` on
that same class, so the interpretation logic always travels with the
class that produced the data. A preset's display name is
`cls.__name__`; `budget` belongs to whichever concrete protocol
defines one (a curriculum might have one per stage, open-ended
learning might have none at all), not to `Benchmark` itself.

`BenchmarkResult` is the atomic, comparable unit of measurement every
protocol shares; `run` decides how many to produce and how to collect
them. Its fields are exactly what's structurally guaranteed for any
navix agent: `returns`/`episode_length` (from `Agent.log_to_wandb`'s
`done_mask`/`returns`/`lengths` handling), `fps`/`wall_time` (always
logged per update), and `flops`/`memory_bytes`/`compile_time_seconds`
(from `Agent.cost_analysis`, implemented generically using only
`train`, so every `Agent` has it). `info: Dict` is a free-form field
for whatever a protocol or caller wants to attach beyond that - empty
by default.

`BenchmarkResult` is a JAX pytree (`flax.struct.PyTreeNode`):
`returns`/`episode_length`/`flops`/`memory_bytes`/
`compile_time_seconds`/`fps`/`wall_time` are its leaves, while `info`
is marked `pytree_node=False` - static auxiliary data, invisible to
`jax.tree.map` and friends, so a tree reduction never touches it. A
single, generic `last_percent_mean` function (below) reduces any
`BenchmarkResult`'s curves to scalars via `jax.tree.map`, since the
pytree registration already tracks which fields are leaves.

`FromScratchBenchmark` implements a flat, order-independent list of
environments, each trained from scratch with no transfer between them
(`Navix1M`/`Navix100K`, below, fix `budget` on top of it). Its `run`
returns `Dict[str, BenchmarkResult]`, one per env_id, full per-update
curves; its `summary` reduces and means across environments into one
aggregate `BenchmarkResult`, via `last_percent_mean` (the "last20%"
convergence convention, by default) applied through `jax.tree.map`;
its `details` is the identity, since `run`'s own output - full curves
keyed by env_id - is already the click-through content a leaderboard
would want to show. Other protocols issue #130 anticipates -
curriculum learning (a fixed stage sequence, graduating per stage),
continual learning (one agent through a task sequence, tracking
retention/forgetting via an R-matrix of task x task performance),
open-ended learning (an adaptive generator, scored by periodic zero-
shot performance on a fixed held-out suite) - subclass `Benchmark`
directly, collect their `BenchmarkResult`s by stage, by task pair, or
by checkpoint, and derive `summary`/`details` with logic specific to
each.

Each algorithm scored is wrapped in an `AlgorithmEntry` - the
provenance metadata #130 requires of a leaderboard row (name, GitHub-
validated author, full commit URLs for both navix and the algorithm's
own repo, paper link). Per #130's "never vendor an external
algorithm's code" rule, `AlgorithmEntry.agent_factory` is how `run`
stays algorithm-agnostic. `entry` isn't part of `BenchmarkResult` -
the caller already has it (they passed it to `run`); pairing the two
for display happens at the call site, not inside this module."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, ClassVar, Dict, Optional, Tuple
from urllib.parse import urlparse

import jax
from jax import Array
import jax.numpy as jnp
from flax import struct

from .agents.agent import Agent, CostAnalysis
from .environments.environment import Environment
from .environments.registry import make, registry
from .experiment import Experiment
from .plotting import derive_scalar_metrics

# GitHub username: alphanumeric/single hyphens, max 39 chars.
_GITHUB_HANDLE_RE = re.compile(r"^[a-zA-Z\d](?:[a-zA-Z\d]|-(?=[a-zA-Z\d])){0,38}$")
# Git commit SHA (abbreviated or full): lowercase hex, 7-40 chars.
_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")


def is_commit_url(url: str) -> bool:
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
            if not is_commit_url(value):
                raise ValueError(
                    f"AlgorithmEntry.{field_name} {value!r} is not a valid commit "
                    "URL (expected an http(s) URL ending in a 7-40 character "
                    "lowercase hex commit SHA)."
                )


class BenchmarkResult(struct.PyTreeNode):
    """The atomic unit of measurement every `Benchmark` protocol
    produces (see module docstring). `returns`/`episode_length`/`fps`/
    `wall_time` hold either full per-update curves (a raw, per-unit
    result) or reduced scalars (a `summary` aggregate) - the type
    doesn't distinguish; `flops`/`memory_bytes`/`compile_time_seconds`
    (from `Agent.cost_analysis`) are always scalar.

    A JAX pytree: the seven fields above are its leaves, `info` is
    marked `pytree_node=False` (static, not a leaf) - so `jax.tree.map`
    only ever touches the array fields. One consequence: combining
    several `BenchmarkResult`s via a *multi*-input `jax.tree.map` (as
    `FromScratchBenchmark.summary` does) requires their `info` to be
    identical, since static data is part of a pytree's structure, not
    something a reduction can average away."""

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
    info: Dict = struct.field(pytree_node=False, default_factory=dict)
    """Free-form - whatever a protocol or caller wants to attach beyond
    the mandatory fields above. Empty by default; nothing populates it
    automatically."""


def last_percent_mean(value: Array, percent: float = 20) -> Array:
    """Mean over the last `percent`% of training (the "last20%"
    convergence convention, by default). A scalar (no update axis,
    e.g. a cost field) passes through unchanged - there's nothing to
    reduce. Applied to a whole `BenchmarkResult` via
    `jax.tree.map(last_percent_mean, result)`."""
    value = jnp.asarray(value)
    if value.ndim == 0:
        return value
    tail = max(1, int(value.shape[-1] * percent / 100))
    return jnp.mean(value[..., -tail:])


class Benchmark:
    """A benchmark protocol - not tied to any one algorithm, and not
    needing an instance: every method is a classmethod, so a preset is
    used directly as `Navix1M.run(entry)`. Its display name is
    `cls.__name__`.

    `build_agent`/`train_on` are shared, concrete mechanics - the "some
    agent, some environment" primitive every protocol needs. A concrete
    protocol with its own hparam overrides (e.g. a training budget)
    layers them on by overriding `build_agent` (see
    `FromScratchBenchmark`, below). `run`/`summary`/`details` have no
    protocol-agnostic default and must be overridden (also see the
    module docstring for the other protocols issue #130 anticipates)."""

    @classmethod
    def build_agent(cls, entry: AlgorithmEntry, env_id: str) -> Tuple[Environment, Agent]:
        """Builds an agent for `env_id` via `entry.agent_factory`. A
        protocol with its own hparam overrides layers them on top of
        `super().build_agent(...)`."""
        env = make(env_id)
        agent = entry.agent_factory(env)
        return env, agent

    @classmethod
    def train_on(cls, agent: Agent, env: Environment, seeds: Tuple[int, ...]) -> Tuple[Agent, Dict]:
        """Trains `agent` on `env`, returning the pre-training `Agent`
        alongside its `logs` - `Agent` is an immutable pytree, so
        `experiment.run` below never mutates it; the caller can reuse
        it for `cost_analysis` (which only depends on shapes, not
        trained weight values) without building a second one. Never
        streams to wandb - a benchmark run is always the fast, local
        path (see `Agent`'s own docstring on the two logging
        strategies); a caller wanting wandb integration does so itself
        afterward, using the full `logs` this returns."""
        experiment = Experiment(name=cls.__name__, agent=agent, env=env, seeds=seeds)
        _, logs = experiment.run(log_to_wandb=False)
        return agent, logs

    @classmethod
    def run(cls, entry: AlgorithmEntry):
        """Trains `entry` under this protocol, returning however many
        `BenchmarkResult`s it produces, collected however makes sense
        for this protocol - a `Dict[str, BenchmarkResult]` keyed by
        env_id for `FromScratchBenchmark`, something else entirely for
        a sequential protocol. No protocol-agnostic default exists;
        must be overridden."""
        raise NotImplementedError

    @classmethod
    def summary(cls, raw):
        """Reduces whatever `run` returned into one comparable
        `BenchmarkResult`. How you reduce a protocol's own collection
        into a single score is inherently protocol-specific (a flat
        mean for independent environments; Average Accuracy/Forgetting
        from an R-matrix for continual learning; ...), so there's no
        default here either."""
        raise NotImplementedError

    @classmethod
    def details(cls, raw):
        """Shapes whatever `run` returned into what a leaderboard shows
        for one algorithm entry beyond its `summary` row - the content
        behind a click-through, e.g. training curves per environment,
        a retention matrix, or a generated-level log, depending on the
        protocol. No protocol-agnostic default exists here either."""
        raise NotImplementedError


class FromScratchBenchmark(Benchmark):
    """Trains `entry` from scratch, independently, across a flat list
    of environments - no ordering, no transfer assumed between them.
    `Navix1M`/`Navix100K` (below) fix `budget` on top of this."""

    budget: ClassVar[int] = 0
    """Overrides `agent.hparams.budget`, layered on in `build_agent`."""
    env_ids: ClassVar[Optional[Tuple[str, ...]]] = None
    """`None` resolves lazily, at `run` time, to every registered
    environment - not resolved at class-definition time, so it
    reflects the full registry regardless of import order."""
    seeds: ClassVar[Tuple[int, ...]] = (0,)

    @classmethod
    def build_agent(cls, entry: AlgorithmEntry, env_id: str) -> Tuple[Environment, Agent]:
        env, agent = super().build_agent(entry, env_id)
        agent = agent.replace(hparams=agent.hparams.replace(budget=cls.budget))
        return env, agent

    @classmethod
    def score_env(cls, logs: Dict, cost: CostAnalysis) -> BenchmarkResult:
        """Builds one environment's raw `BenchmarkResult`: full
        comparable-metric curves (`returns`/`episode_length`/`fps`/
        `wall_time`) plus this agent's `cost_analysis` (always
        scalar)."""
        scalars = derive_scalar_metrics(logs)
        return BenchmarkResult(
            returns=scalars["perf/returns"],
            episode_length=scalars["perf/episode_length"],
            flops=jnp.asarray(cost.flops),
            memory_bytes=jnp.asarray(cost.memory_bytes),
            compile_time_seconds=jnp.asarray(cost.compile_time_seconds),
            fps=jnp.asarray(logs["iter/fps"]),
            wall_time=jnp.asarray(logs["iter/wall_time"]),
        )

    @classmethod
    def run(
        cls,
        entry: AlgorithmEntry,
        env_ids: Optional[Tuple[str, ...]] = None,
        seeds: Optional[Tuple[int, ...]] = None,
    ) -> Dict[str, BenchmarkResult]:
        """Runs `entry` against `env_ids` (default: every registered
        environment) independently, at `cls.budget`. Returns one raw
        `BenchmarkResult` per env_id - that's already the per-
        environment breakdown for this protocol; call `summary` on
        this same class to reduce it to a single comparable score."""
        env_ids = env_ids if env_ids is not None else cls.env_ids or tuple(registry().keys())
        seeds = seeds if seeds is not None else cls.seeds
        raw: Dict[str, BenchmarkResult] = {}
        for env_id in env_ids:
            env, agent = cls.build_agent(entry, env_id)
            agent, logs = cls.train_on(agent, env, seeds)
            cost = agent.cost_analysis(jax.random.PRNGKey(seeds[0]))
            raw[env_id] = cls.score_env(logs, cost)
        return raw

    @classmethod
    def summary(cls, raw: Dict[str, BenchmarkResult]) -> BenchmarkResult:
        """Each field's last-percent-mean, meaned across every
        environment in `raw`."""
        reduced = [jax.tree.map(last_percent_mean, result) for result in raw.values()]
        return jax.tree.map(lambda *values: jnp.mean(jnp.stack(values)), *reduced)

    @classmethod
    def details(cls, raw: Dict[str, BenchmarkResult]) -> Dict[str, BenchmarkResult]:
        """The per-environment training curves - already what `raw`
        holds for this protocol."""
        return raw


class Navix1M(FromScratchBenchmark):
    """1M-frame budget per environment - `PPOHparams`/`PQNHparams`'
    own default."""

    budget = 1_000_000


class Navix100K(FromScratchBenchmark):
    """Same as `Navix1M`, at 100K frames - a cheaper preset for quick
    checks."""

    budget = 100_000
