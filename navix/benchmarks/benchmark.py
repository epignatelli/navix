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

"""`TrainingCurve` (what `AlgorithmEntry.train` returns - purely
what's computable inside a `jax.jit` trace), `AlgorithmEntry` (the
provenance and hardware metadata wrapper for one algorithm being
scored, issue #130) plus its `CostAnalysis`, `BenchmarkResult` (one
entry's scored run - a `TrainingCurve` plus everything only an
external, un-jitted wrapper can measure: wall-clock time and cost),
and `Benchmark` (the abstract protocol base every preset, e.g.
`navix.benchmarks.scratch.FromScratchBenchmark`, subclasses). See
`navix.benchmarks` (this package's `__init__.py`) for the full
design."""
from __future__ import annotations

import dataclasses
import inspect
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

import chex
import jax
import jaxlib
from jax import Array
import jax.numpy as jnp
import numpy as np
from flax import struct

from .hardware import cpu_type, cuda_version, cudnn_version, gpu_type, ram_bytes
from ..environments.registry import registry

# Budget used only to shape-check train()'s output in AlgorithmEntry's
# __post_init__ - not a real training run, so any small positive value
# works.
_VALIDATION_BUDGET = 128

# GitHub username: alphanumeric/single hyphens, max 39 chars.
_GITHUB_HANDLE_RE = re.compile(r"^[a-zA-Z\d](?:[a-zA-Z\d]|-(?=[a-zA-Z\d])){0,38}$")
# Git commit SHA (abbreviated or full): lowercase hex, 7-40 chars.
_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")


class TrainingCurve(struct.PyTreeNode):
    """One `AlgorithmEntry.train` call's measurements - purely what's
    computable from inside a `jax.jit` trace, from the real (state,
    action, reward, done) interaction stream. No wall-clock timing, no
    cost - `time.time()` inside a jitted function only ever fires at
    trace time, not per call, so neither can be measured here; see
    `BenchmarkResult` for those.

    Every field is a per-update curve - shape `(num_updates,)` for a
    single training curve (checked by `AlgorithmEntry.
    validate_train_contract`), so `last_percent_mean`/
    `last_percent_variance`/`convergence_rate` reduce all of them the
    same way, uniformly, no exceptions.

    Attributes:
        episodic_returns (Array): Episodic return, masked-mean over
            completed episodes only.
        lengths (Array): Episode length, masked-mean over completed
            episodes only.
        diagnostics (Dict[str, Array]): Free-form per-update diagnostic
            curves (e.g. `{"loss": ..., "lr": ...}`) - whatever helps
            debug the algorithm. Each value shape `(num_updates,)`.
            Empty by default.
    """

    episodic_returns: Array
    lengths: Array
    diagnostics: Dict = struct.field(default_factory=dict)

    def last_percent_mean(self, percent: float = 20) -> TrainingCurve:
        """Reduces every field to its mean over the last `percent`% of
        its trailing axis.

        Args:
            percent (float): Percentage of the trailing axis to
                average over.

        Returns:
            TrainingCurve: A copy with every field reduced along its
            trailing axis.
        """

        def reduce(value: Array) -> Array:
            tail = max(1, int(value.shape[-1] * percent / 100))
            return jnp.mean(value[..., -tail:], axis=-1)

        return self.replace(
            episodic_returns=reduce(self.episodic_returns),
            lengths=reduce(self.lengths),
            diagnostics={key: reduce(value) for key, value in self.diagnostics.items()},
        )

    def last_percent_variance(self, percent: float = 20) -> TrainingCurve:
        """Reduces every field to its variance over the last
        `percent`% of its trailing axis - how much an already-converged
        curve still fluctuates update-to-update, not variance across
        seeds. The training-curve analogue of the per-update variance
        in Rowland, Dabney & Munos, "Adaptive Trade-Offs in Off-Policy
        Learning" (https://arxiv.org/abs/1910.07478, Definition 1.4).

        Args:
            percent (float): Percentage of the trailing axis to
                compute the variance over.

        Returns:
            TrainingCurve: A copy with every field reduced along its
            trailing axis.
        """

        def reduce(value: Array) -> Array:
            tail = max(1, int(value.shape[-1] * percent / 100))
            return jnp.var(value[..., -tail:], axis=-1)

        return self.replace(
            episodic_returns=reduce(self.episodic_returns),
            lengths=reduce(self.lengths),
            diagnostics={key: reduce(value) for key, value in self.diagnostics.items()},
        )

    def convergence_rate(self, percent: float = 20) -> TrainingCurve:
        """Reduces every field to a normalized area under its curve:
        the ratio of the curve's overall mean to its
        `last_percent_mean`. Near 1 means the curve was close to its
        own asymptote for most of training (fast convergence); near 0
        means most of training was spent far from it (slow).

        Args:
            percent (float): Percentage of the trailing axis that
                defines the asymptote (see `last_percent_mean`).

        Returns:
            TrainingCurve: A copy with every field reduced along its
            trailing axis.
        """

        def reduce(value: Array) -> Array:
            tail = max(1, int(value.shape[-1] * percent / 100))
            target = jnp.mean(value[..., -tail:], axis=-1)
            overall = jnp.mean(value, axis=-1)
            return overall / target

        return self.replace(
            episodic_returns=reduce(self.episodic_returns),
            lengths=reduce(self.lengths),
            diagnostics={key: reduce(value) for key, value in self.diagnostics.items()},
        )


def is_commit_url(url: str) -> bool:
    """Checks whether `url` is a full URL ending in a commit SHA.

    Args:
        url (str): The URL to validate.

    Returns:
        bool: True if `url` has an http(s) scheme and its last path
        segment is a 7-40 character lowercase hex SHA.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return False
    last_segment = parsed.path.rstrip("/").rsplit("/", 1)[-1]
    return bool(_SHA_RE.match(last_segment))


class CostAnalysis(struct.PyTreeNode):
    """The cost of one `AlgorithmEntry.train` call, as measured by
    `AlgorithmEntry.cost_analysis`.

    Attributes:
        flops (float): FLOPs, from `compiled.cost_analysis()`.
        memory_bytes (float): Peak memory proxy (argument + temp +
            output size), from `compiled.memory_analysis()`.
        compile_time_seconds (float): Wall-clock time to compile.
            Hardware/XLA-version-sensitive.
    """

    flops: float
    memory_bytes: float
    compile_time_seconds: float


@dataclass
class AlgorithmEntry:
    """One algorithm to score against a `Benchmark`.

    To submit an algorithm: subclass `AlgorithmEntry`, override
    `train` to build whatever model `env_id` needs and train it and
    return a `TrainingCurve` - the only requirement - then construct an
    instance with the provenance fields below (see
    `benchmarks/README.md`). The hardware fields (`gpu_type` through
    `jaxlib_version`) are auto-detected in `__post_init__`, not
    constructor arguments. `__post_init__` also checks that `train`
    returns a `TrainingCurve` with the right shape, so a malformed
    `train` fails at construction time, not partway through a real
    benchmark run.

    Attributes:
        name (str): Algorithm name, e.g. "PPO".
        author (str): This implementation's author (GitHub handle),
            not the paper's. Validated in `__post_init__`.
        paper_url (str): Link to the paper the algorithm is from.
        navix_commit_url (str): Link to the navix commit this result
            was produced against (issue #130's `navix.sha`, as a
            URL). Validated in `__post_init__`.
        algorithm_commit_url (str): Link to the algorithm
            implementation's own commit (issue #130's `agent.sha`, as
            a URL) - same commit as `navix_commit_url` for a
            navix-shipped agent, a different repo's commit for an
            external one. Validated in `__post_init__`.
        gpu_type (Optional[str]): The GPU model JAX runs on, or `None`
            if JAX isn't running on a GPU.
        cpu_type (str): The CPU's model name.
        ram_bytes (int): Total system RAM, in bytes.
        cuda_version (Optional[str]): The CUDA version jaxlib runs on,
            or `None` if JAX isn't running on a GPU.
        cudnn_version (Optional[str]): The cuDNN version jaxlib runs
            on, or `None` if JAX isn't running on a GPU.
        jax_version (str): `jax.__version__`.
        jaxlib_version (str): `jaxlib.__version__`.
    """

    name: str
    author: str
    paper_url: str
    navix_commit_url: str
    algorithm_commit_url: str
    gpu_type: Optional[str] = field(init=False, default=None)
    cpu_type: str = field(init=False, default="")
    ram_bytes: int = field(init=False, default=0)
    cuda_version: Optional[str] = field(init=False, default=None)
    cudnn_version: Optional[str] = field(init=False, default=None)
    jax_version: str = field(init=False, default="")
    jaxlib_version: str = field(init=False, default="")

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
        self.gpu_type = gpu_type()
        self.cpu_type = cpu_type()
        self.ram_bytes = ram_bytes()
        self.cuda_version = cuda_version()
        self.cudnn_version = cudnn_version()
        self.jax_version = jax.__version__
        self.jaxlib_version = jaxlib.__version__  # type: ignore[attr-defined]
        self.validate_train_contract()

    def validate_train_contract(self) -> None:
        """Checks `self.train` returns a `TrainingCurve` with the right
        shape, without running any real training - `jax.eval_shape`
        traces `self.train` for its output structure only, against one
        representative registered environment. A single (un-vmapped)
        call, so `episodic_returns`/`lengths`/`diagnostics`' values must
        all be rank 1 (one point per update).

        Raises:
            TypeError: If `self.train`'s output isn't a `TrainingCurve`.
            AssertionError: If `episodic_returns`/`lengths`/
                `diagnostics`' values aren't rank 1 (`chex.assert_rank`).
        """
        env_id = sorted(registry().keys())[0]
        rng = jax.random.PRNGKey(0)
        result_shape = jax.eval_shape(lambda rng: self.train(env_id, _VALIDATION_BUDGET, rng), rng)
        if not isinstance(result_shape, TrainingCurve):
            raise TypeError(
                f"{type(self).__name__}.train(...) must return a TrainingCurve, "
                f"got {type(result_shape).__name__}."
            )
        chex.assert_rank(
            [result_shape.episodic_returns, result_shape.lengths] + list(result_shape.diagnostics.values()),
            1,
        )

    def train(self, env_id: str, budget: int, rng: jax.Array) -> TrainingCurve:
        """Builds a fresh, env-shaped model for `env_id` and trains it
        at `budget`. The one method every submission must override -
        building the model is inherently algorithm-specific, so
        there's no protocol-agnostic default.

        Runs inside a `jax.jit`/`jax.vmap` trace (see `Benchmark.
        run_env`), so only genuinely jittable work belongs here -
        wall-clock timing and cost are measured separately, from
        outside any trace (see `BenchmarkResult`, `cost_analysis`).

        Args:
            env_id (str): The environment to train on.
            budget (int): Training budget - whatever the running
                `Benchmark` protocol passed to `Benchmark.run_env`. Use
                it to build your hparams (e.g.
                `PPOHparams(budget=budget)`) if your algorithm's
                training length should respect it.
            rng (jax.Array): The PRNG key to train with.

        Returns:
            TrainingCurve: `episodic_returns`/`lengths` filled in;
            `diagnostics` optionally, with whatever per-update values
            help debug this algorithm.

        Raises:
            NotImplementedError: Always, on `AlgorithmEntry` itself -
                must be overridden.
        """
        raise NotImplementedError

    def cost_analysis(self, env_id: str, budget: int) -> CostAnalysis:
        """Compiles `self.train(env_id, budget, ...)` and reads its
        FLOPs/memory/compile-time. Always seed 0, since cost is
        shape-driven, not value-driven.

        Necessarily includes whatever env interaction `train` does
        internally. In practice this lands close to "one update's
        cost", not the whole run's: every navix-shipped agent's
        `train` is init + `jax.lax.scan(self.update, ...,
        length=num_updates)`, and XLA's `cost_analysis()` on a
        compiled scan reports one iteration's cost, not `length`
        copies. An agent whose `train` isn't scan-shaped will report a
        different figure here.

        Args:
            env_id (str): The environment to build the model for.
            budget (int): Training budget, passed to `self.train`.

        Returns:
            CostAnalysis: FLOPs, peak memory, and compile time for one
            `self.train(env_id, budget, ...)` call.
        """
        rng = jax.random.PRNGKey(0)
        start = time.time()
        compiled = jax.jit(lambda rng: self.train(env_id, budget, rng)).lower(rng).compile()
        compile_time_seconds = time.time() - start

        analysis = compiled.cost_analysis()
        # Some jaxlib versions return a dict directly, others a list of
        # per-computation dicts (usually length 1) - normalise both.
        if isinstance(analysis, list):
            analysis = analysis[0] if analysis else {}
        if not isinstance(analysis, dict):
            raise TypeError(f"Unexpected compiled.cost_analysis() return type {type(analysis)}")
        flops = analysis.get("flops", float("nan"))
        mem = compiled.memory_analysis()
        if mem is None:
            raise ValueError("compiled.memory_analysis() returned None")
        memory_bytes = mem.argument_size_in_bytes + mem.temp_size_in_bytes + mem.output_size_in_bytes
        return CostAnalysis(
            flops=flops,
            memory_bytes=memory_bytes,
            compile_time_seconds=compile_time_seconds,
        )


class BenchmarkResult(struct.PyTreeNode):
    """One `AlgorithmEntry`'s scored run under a `Benchmark` protocol -
    a `TrainingCurve` plus everything only an external, un-jitted
    wrapper can measure. Built by `Benchmark.run_env` in one shot, from
    three independently-measured pieces.

    Attributes:
        curve (TrainingCurve): `AlgorithmEntry.train`'s output.
        wall_time (Array): Real wall-clock time to execute the
            already-compiled `AlgorithmEntry.train` (all of
            `Benchmark.seeds`, vmapped together) - timed and
            `jax.block_until_ready`'d from outside any `jax.jit`
            trace, unlike anything `train` could measure about
            itself. Excludes compile time (see
            `cost.compile_time_seconds`). Scalar.
        fps (Array): Training throughput: `budget / wall_time`. Scalar.
            Comparable only across results measured on the same
            hardware.
        cost (CostAnalysis): From `AlgorithmEntry.cost_analysis`.
    """

    curve: TrainingCurve
    wall_time: Array
    fps: Array
    cost: CostAnalysis


class Benchmark(struct.PyTreeNode):
    """A benchmark protocol - not tied to any one algorithm.

    Used as an instance, e.g. `Navix1M().run(entry)`; its display name
    is `type(self).__name__`. `run`/`summary`/`details` have no
    protocol-agnostic default and must be overridden by a concrete
    protocol (e.g. `FromScratchBenchmark`). `submit_entry` is concrete
    here - it writes out whatever `run`/`summary`/`details` already
    produced, the same way for every protocol.
    """

    seeds: Tuple[int, ...] = struct.field(pytree_node=False, default_factory=lambda: tuple(range(16)))

    def __post_init__(self) -> None:
        if len(self.seeds) <= 1:
            raise ValueError(f"Benchmark.seeds must have more than one seed, got {self.seeds!r}.")

    def run_env(self, entry: AlgorithmEntry, env_id: str, budget: int) -> BenchmarkResult:
        """Trains `entry` on `env_id` at `budget`, vmapped over
        `self.seeds`, timing the run (compile time excluded) and
        reading its cost.

        Args:
            entry (AlgorithmEntry): The algorithm to train.
            env_id (str): The environment to train on.
            budget (int): Training budget, passed to `entry.train`/
                `entry.cost_analysis`.

        Returns:
            BenchmarkResult: `curve` is `entry.train`'s output, one
            per seed stacked along a new leading axis. `wall_time`/
            `fps`/`cost` are each a single scalar for the whole
            vmapped call - seeds train together in one fused
            computation, so there's no meaningful per-seed timing
            breakdown.
        """
        rng = jnp.asarray([jax.random.PRNGKey(seed) for seed in self.seeds])
        compiled = jax.jit(jax.vmap(lambda rng: entry.train(env_id, budget, rng))).lower(rng).compile()
        start = time.time()
        curve = jax.block_until_ready(compiled(rng))
        wall_time = jnp.asarray(time.time() - start)
        fps = jnp.asarray(budget / wall_time)
        cost = entry.cost_analysis(env_id, budget)
        return BenchmarkResult(curve=curve, wall_time=wall_time, fps=fps, cost=cost)

    def run(self, entry: AlgorithmEntry) -> BenchmarkResult:
        """Trains `entry` under this protocol.

        Args:
            entry (AlgorithmEntry): The algorithm to score.

        Returns:
            BenchmarkResult: However many things this protocol
            measures per run, stacked along a leading axis - which
            axis, and what it represents (e.g. one row per environment
            for `FromScratchBenchmark`), is protocol-specific.

        Raises:
            NotImplementedError: Always, on `Benchmark` itself - must
                be overridden by a concrete protocol.
        """
        raise NotImplementedError

    def summary(self, results: BenchmarkResult) -> Dict[str, Array]:
        """Reduces a `BenchmarkResult` from `run` into a leaderboard's
        table row for one algorithm entry.

        Args:
            results (BenchmarkResult): This protocol's `run` output.

        Returns:
            Dict[str, Array]: The named columns a leaderboard's table
            row shows for this entry - which columns exist is
            protocol-specific.

        Raises:
            NotImplementedError: Always, on `Benchmark` itself - must
                be overridden by a concrete protocol.
        """
        raise NotImplementedError

    def details(self, results: BenchmarkResult) -> Dict[str, Any]:
        """Reduces a `BenchmarkResult` from `run` into per-row
        diagnostics about this benchmark run - the same kind of
        columns `summary` aggregates, but one row per whatever this
        protocol's leading axis represents (e.g. one row per
        environment for `FromScratchBenchmark`), instead of a single
        further-aggregated row. Diagnostics about the benchmark run
        itself, not a leaderboard's per-algorithm click-through page -
        what that page shows depends on the algorithm, not the
        benchmark protocol.

        A concrete override should include whatever labels row
        identity (e.g. `env_ids`) as one of its own returned entries,
        if this protocol has a meaningful one - `submit_entry` writes
        out exactly what this returns and nothing more. Values aren't
        required to be `Array` (e.g. `env_ids` is a `Tuple[str, ...]`,
        not every metric need be jax-typed) - unlike `summary`, this
        isn't reduced to a uniform type.

        Args:
            results (BenchmarkResult): This protocol's `run` output.

        Returns:
            Dict[str, Any]: One row per unit of this protocol's
            leading axis - which columns exist, and what that axis is,
            is protocol-specific.

        Raises:
            NotImplementedError: Always, on `Benchmark` itself - must
                be overridden by a concrete protocol.
        """
        raise NotImplementedError

    def submit_entry(self, entry: AlgorithmEntry, results: BenchmarkResult, max_points: int = 50) -> None:
        """Writes one `self.run(entry)`'s output into the directory of
        whichever script called this - the same convention every
        submission's `run.py` already follows for
        `config.yml`/`requirements.txt`, so a submission's results end
        up right alongside them.

        Writes three files:

        - `summary.json`: `entry`'s provenance/hardware fields plus
          `self.summary(results)` - the leaderboard's table row.
        - `details.json`: `self.details(results)` - per-row
          diagnostics about this run.
        - `diagnostics.npz`: `results` itself. `curve.episodic_returns`/
          `curve.lengths`/`curve.diagnostics`' values are resampled to
          exactly `max_points` evenly-spaced points along their
          trailing axis, so every submission's curve fields end up the
          same fixed shape regardless of how many updates actually
          ran; `wall_time`/`fps`/`cost.*` are already scalars and are
          written as-is. `curve.diagnostics`' entries are flattened to
          top-level `diagnostics_<key>` npz entries (npz has no native
          nesting).

        Args:
            entry (AlgorithmEntry): The algorithm that produced
                `results`.
            results (BenchmarkResult): This protocol's `run` output.
            max_points (int): Number of points `curve.episodic_returns`/
                `curve.lengths`/`curve.diagnostics`' values are
                resampled to in `diagnostics.npz`.
        """
        caller = inspect.stack()[1]
        path = Path(caller.filename).resolve().parent
        entry_payload = dataclasses.asdict(entry)

        def to_jsonable(tree):
            return jax.tree.map(lambda x: np.asarray(x).tolist(), tree)

        summary = jax.device_get(self.summary(results))
        summary_payload = {"entry": entry_payload, "summary": to_jsonable(summary)}
        (path / "summary.json").write_text(json.dumps(summary_payload, indent=2))

        details = jax.device_get(self.details(results))
        details_payload = {"entry": entry_payload, "details": to_jsonable(details)}
        (path / "details.json").write_text(json.dumps(details_payload, indent=2))

        result = jax.device_get(results)

        def resample(value: Array) -> np.ndarray:
            array = np.asarray(value)
            index = np.linspace(0, array.shape[-1] - 1, max_points).astype(int)
            return array[..., index]

        curves = {
            "episodic_returns": resample(result.curve.episodic_returns),
            "length": resample(result.curve.lengths),
            "wall_time": np.asarray(result.wall_time),
            "fps": np.asarray(result.fps),
            "flops": np.asarray(result.cost.flops),
            "memory_bytes": np.asarray(result.cost.memory_bytes),
            "compile_time_seconds": np.asarray(result.cost.compile_time_seconds),
        }
        for key, value in result.curve.diagnostics.items():
            curves[f"diagnostics_{key}"] = resample(value)
        np.savez_compressed(path / "diagnostics.npz", **curves)
