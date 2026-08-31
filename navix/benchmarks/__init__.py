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

"""The navix leaderboard's `Benchmark` protocol (issue #130): scores
an `AlgorithmEntry` against a preset, e.g. `Navix1M().run(entry)`.

This package is split into `benchmark.py` (`TrainingCurve`, what
`AlgorithmEntry.train` returns - the only requirement; `__post_init__`
checks this, so a malformed `train` fails at construction time, not
partway through a real benchmark run; `CostAnalysis`; `BenchmarkResult`
- a `TrainingCurve` plus the wall-clock timing and cost only an
external, un-jitted wrapper can measure; and the abstract `Benchmark`
base every protocol implements), `hardware.py` (the hardware-detection
functions `AlgorithmEntry` auto-populates itself from), `scratch.py`
(`FromScratchBenchmark` and its `Navix1M`/`Navix100K` presets - the one
concrete protocol so far), `search.py` (`search_hparams` - an optional,
`Benchmark`-independent Evolution-Strategies search an entry's own
`run.py` can use to tune its hyperparameters before scoring; not part
of `Benchmark`/`AlgorithmEntry` itself, since what's searchable is
inherently entry-specific - `benchmark.py` also has `Benchmark.
plot_summary`/`plot_details`/`plot_diagnostics` for locally inspecting
a scored run's `summary`/`details`/`diagnostics.npz` without the
online leaderboard, next to the `Benchmark` methods they render), and
`plotting.py` (a local no-wandb dashboard for `logs` - see `Agent`'s
docstring and issue #60, unrelated to `Benchmark` scoring). A future
protocol (curriculum learning, continual learning, open-ended
learning) gets its own file alongside `scratch.py`, subclassing
`Benchmark` directly.

Typical usage:

    entry = MyEntry(name=..., author=..., paper_url=...,
                     navix_commit_url=..., algorithm_commit_url=...)
    # entry's construction already checked MyEntry.train returns a
    # TrainingCurve
    benchmark = Navix1M()
    results = benchmark.run(entry)
    summary = benchmark.summary(results)
    benchmark.submit_entry(entry, results)

See `benchmarks/README.md` for the full submission workflow."""
from .benchmark import (
    AlgorithmEntry,
    Benchmark,
    BenchmarkResult,
    CostAnalysis,
    TrainingCurve,
    is_commit_url,
    plot_benchmark_summary,
    plot_benchmark_details,
    plot_benchmark_diagnostics,
)
from .hardware import cpu_type, cuda_version, cudnn_version, gpu_type, ram_bytes
from .scratch import FromScratchBenchmark, Navix1M, Navix100K
from .search import search_hparams
from . import plotting
from .plotting import MANDATORY_METRICS, plot_metric, plot_metrics, plot_dashboard

__all__ = [
    "AlgorithmEntry",
    "BenchmarkResult",
    "CostAnalysis",
    "cpu_type",
    "cuda_version",
    "cudnn_version",
    "gpu_type",
    "is_commit_url",
    "ram_bytes",
    "Benchmark",
    "TrainingCurve",
    "FromScratchBenchmark",
    "Navix1M",
    "Navix100K",
    "search_hparams",
    "plotting",
    "MANDATORY_METRICS",
    "plot_metric",
    "plot_metrics",
    "plot_dashboard",
    "plot_benchmark_summary",
    "plot_benchmark_details",
    "plot_benchmark_diagnostics",
]
