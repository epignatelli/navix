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

"""`FromScratchBenchmark` and its `Navix1M`/`Navix100K` presets -
trains an entry from scratch, independently, across a flat list of
environments. See `navix.benchmarks` (this package's `__init__.py`)
for the full design."""
from __future__ import annotations

from typing import Any, ClassVar, Dict, Tuple

import jax
import jax.numpy as jnp
from flax import struct

from .benchmark import AlgorithmEntry, Benchmark, BenchmarkResult
from ..environments.registry import registry


DEFAULT_ENV_IDS: Tuple[str, ...] = (
    "Navix-Empty-8x8-v0",
    "Navix-Dynamic-Obstacles-5x5-v0",
    "Navix-FourRooms-v0",
    "Navix-KeyCorridorS4R3-v0",
    "Navix-DoorKey-8x8-v0",
    "Navix-SimpleCrossingS9N2-v0",
)
"""`FromScratchBenchmark.env_ids`'s default - chosen for the RL
capability each env isolates, not just family coverage: `Empty-8x8`
(convergence rate - solvable with no confounding structure, so speed
differences are legible), `Dynamic-Obstacles-5x5` (variance - the
stochasticity is in the environment itself, moving obstacles differ
every episode), `FourRooms` (exploration - Sutton & Precup's original
domain, no reward signal until the goal room is found), `KeyCorridorS4R3`
(credit assignment, deep chain - key -> door -> goal with no partial
credit), `DoorKey-8x8` (credit assignment, shallow chain - the same
idea at a smaller difficulty gradient), `SimpleCrossingS9N2` (static-
obstacle pathing). Not every registered environment - falsy (e.g.
explicitly set to `None`) resolves lazily, at `run`/`details` time, to
every registered environment instead."""


class FromScratchBenchmark(Benchmark):
    """Trains `entry` from scratch, independently, across a flat list
    of environments - no ordering, no transfer assumed between them.

    `env_ids`/`seeds`/`budget` are all fixed per preset - overridden
    by subclassing, not by a `run` argument, so every run of a given
    class scores the same environments with the same seeds;
    `Navix1M`/`Navix100K` (below) fix `budget` on top of this.

    Attributes:
        budget (int): Passed to `entry.train`/`entry.cost_analysis` as
            their `budget` argument.
        env_ids (Tuple[str, ...]): Environments to train on. Defaults
            to `DEFAULT_ENV_IDS` - a small, curated set spanning
            several environment families, not every registered
            environment. Falsy (e.g. explicitly set to `None`)
            resolves lazily, at `run`/`details` time, to every
            registered environment instead.
    """

    budget: int = struct.field(pytree_node=False, default=0)
    env_ids: Tuple[str, ...] = struct.field(pytree_node=False, default_factory=lambda: DEFAULT_ENV_IDS)

    NON_NUMERIC_DETAILS: ClassVar[Tuple[str, ...]] = ("env_ids",)
    """`self.details(...)`'s keys that aren't `jnp.mean`-able - row
    labels, not metrics."""

    def run(self, entry: AlgorithmEntry) -> BenchmarkResult:
        """Runs `entry` against `self.env_ids` independently, at
        `self.budget`, using `self.seeds`.

        Args:
            entry (AlgorithmEntry): The algorithm to score.

        Returns:
            BenchmarkResult: One result per env_id, stacked along a
            new leading axis - row `i` is `self.env_ids[i]` (or the
            `i`-th registered env, if `self.env_ids` is unset); see
            `details`.
        """
        env_ids = self.env_ids or tuple(registry().keys())
        per_env = [self.run_env(entry, env_id, self.budget) for env_id in env_ids]
        return jax.tree.map(lambda *values: jnp.stack(values), *per_env)

    def summary(self, results: BenchmarkResult) -> Dict[str, jax.Array]:
        """The leaderboard table row for this protocol.

        Args:
            results (BenchmarkResult): This protocol's `run` output.

        Returns:
            Dict[str, jax.Array]: Each numeric column of
            `self.details(results)` meaned across its env axis -
            `episodic_returns`' last-percent-mean (bias), variance,
            convergence rate, and finite fraction (see `self.details`),
            plus `fps`/`flops`/`memory_bytes`/`compile_time_seconds`/
            `wall_time`'s bias. `length`/
            `env_ids` (see `NON_NUMERIC_DETAILS`) aren't included -
            still on `self.details(results)`. Non-finite values (e.g.
            `returns_convergence_rate`'s `overall / target` is `0/0` or
            `x/0` when an environment's episodic_returns never leaves
            zero - a real algorithm never solving that environment, not
            a bug) are excluded from the mean rather than propagated -
            one degenerate environment/seed shouldn't blank out every
            other one's otherwise-valid signal. `self.details(results)`
            keeps the raw, un-filtered per-environment values (a NaN
            there is itself informative), only this aggregate step
            filters them.
        """
        details = self.details(results)
        skip = self.NON_NUMERIC_DETAILS + ("length",)

        def finite_mean(value: jax.Array) -> jax.Array:
            return jnp.nanmean(jnp.where(jnp.isfinite(value), value, jnp.nan))

        return {key: finite_mean(value) for key, value in details.items() if key not in skip}

    def details(self, results: BenchmarkResult) -> Dict[str, Any]:
        """Per-environment breakdown of this run's metrics.

        Args:
            results (BenchmarkResult): This protocol's `run` output.

        Returns:
            Dict[str, Any]: The same last-percent reduction `summary`
            aggregates further, but stopped one step earlier - every
            column keeps its leading env axis. Includes `env_ids`
            (which row is which - a `Tuple[str, ...]`, not an `Array`),
            `length` (not in `summary`, but a useful per-env
            diagnostic), and `returns_finite_fraction` (fraction of
            `self.seeds` whose `returns_convergence_rate` was finite -
            i.e. made *some* real progress in the final 20% of training;
            a reliability signal `episodic_returns`' bias alone can't
            distinguish "consistently mediocre" from "mostly zero, one
            seed got lucky").
        """
        bias = results.curve.last_percent_mean()
        variance = results.curve.last_percent_variance().episodic_returns
        convergence_rate = results.curve.convergence_rate().episodic_returns
        # convergence_rate is overall/target - only finite when target
        # (the last-percent-mean) is nonzero, i.e. the seed made *some*
        # real progress in the final 20% of training. The fraction of
        # seeds where that's true is itself a real reliability signal
        # summary()'s finite-value-excluding mean can't surface on its
        # own - e.g. "0.15 mean return, 4/16 seeds finite" and "0.15
        # mean return, 16/16 seeds finite" look identical in every other
        # column, but very different in practice (one is a strong,
        # consistent policy; the other is one lucky seed dragging up a
        # near-total-failure average).
        returns_finite_fraction = jnp.mean(jnp.isfinite(convergence_rate), axis=-1)
        env_ids = self.env_ids or tuple(registry().keys())
        return {
            "env_ids": env_ids,
            "episodic_returns": bias.episodic_returns,
            "returns_variance": variance,
            "returns_convergence_rate": convergence_rate,
            "returns_finite_fraction": returns_finite_fraction,
            "length": bias.lengths,
            "fps": results.fps,
            "flops": results.cost.flops,
            "memory_bytes": results.cost.memory_bytes,
            "compile_time_seconds": results.cost.compile_time_seconds,
            "wall_time": results.wall_time,
        }


class Navix1M(FromScratchBenchmark):
    """1M-frame budget per environment - `PPOHparams`/`PQNHparams`'
    own default."""

    budget: int = struct.field(pytree_node=False, default=1_000_000)


class Navix100K(FromScratchBenchmark):
    """Same as `Navix1M`, at 100K frames - a cheaper preset for quick
    checks."""

    budget: int = struct.field(pytree_node=False, default=100_000)
