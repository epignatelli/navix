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

from typing import Any, ClassVar, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import struct

from .benchmark import AlgorithmEntry, Benchmark, BenchmarkResult
from ..environments.registry import registry


DEFAULT_ENV_IDS: Tuple[str, ...] = (
    "Navix-Empty-5x5-v0",
    "Navix-Dynamic-Obstacles-5x5-v0",
    "Navix-DoorKey-8x8-v0",
    "Navix-Crossings-S9N2-v0",
)


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
        env_ids (Optional[Tuple[str, ...]]): Environments to train on.
            `None` resolves lazily, at `run` time, to every registered
            environment.
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
            and convergence rate, plus `fps`/`flops`/`memory_bytes`/
            `compile_time_seconds`/`wall_time`'s bias. `length`/
            `env_ids` (see `NON_NUMERIC_DETAILS`) aren't included -
            still on `self.details(results)`.
        """
        details = self.details(results)
        skip = self.NON_NUMERIC_DETAILS + ("length",)
        return {key: jnp.mean(value) for key, value in details.items() if key not in skip}

    def details(self, results: BenchmarkResult) -> Dict[str, Any]:
        """Per-environment breakdown of this run's metrics.

        Args:
            results (BenchmarkResult): This protocol's `run` output.

        Returns:
            Dict[str, Any]: The same last-percent reduction `summary`
            aggregates further, but stopped one step earlier - every
            column keeps its leading env axis. Includes `env_ids`
            (which row is which - a `Tuple[str, ...]`, not an `Array`)
            and `length` (not in `summary`, but a useful per-env
            diagnostic).
        """
        bias = results.curve.last_percent_mean()
        variance = results.curve.last_percent_variance().episodic_returns
        convergence_rate = results.curve.convergence_rate().episodic_returns
        env_ids = self.env_ids or tuple(registry().keys())
        return {
            "env_ids": env_ids,
            "episodic_returns": bias.episodic_returns,
            "returns_variance": variance,
            "returns_convergence_rate": convergence_rate,
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
