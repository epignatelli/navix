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

"""Plotting utilities for the `logs` pytree returned by `Experiment.run()`
and `Experiment.run_hparam_search()`, so training can be inspected without
wandb (see `Agent`'s docstring and issue #60: `Experiment.run(log_to_wandb=
False)` and reading `logs` directly is the fast path). Also covers a
`Benchmark`-scored run's `summary`/`details`/`diagnostics` (see
`plot_benchmark_summary`/`plot_benchmark_details`/`plot_benchmark_diagnostics`
below), for inspecting a submission locally before/without the online
leaderboard.

`MANDATORY_METRICS` is a fixed, deliberately-chosen set of plots, rather
than auto-detecting whatever keys happen to be in `logs`. Each entry is
derivable purely from the (state, action, reward, done) interaction
stream plus wall-clock time - the one interface every RL algorithm
shares, regardless of its internals - so the set stays meaningful for any
future agent, not just the ones navix ships: `perf/returns`,
`perf/success_rate`, `perf/episode_length`, `iter/fps`, `iter/wall_time`.
Identical across every navix agent so results are visually comparable
across algorithms - this is the set the navix leaderboard (#130) is
expected to standardise on.

This module deliberately does not know about "diagnostic" (algorithm-
specific) metrics - an algorithm submitted to a leaderboard won't
necessarily have any navix-specific code to declare which of its own
logged keys are diagnostic. That categorisation belongs to whatever
consumes this module (e.g. a leaderboard's own file mapping algorithm ->
diagnostic keys), not to navix itself. `plot_metrics`/`plot_dashboard`
both accept an arbitrary metrics dict for exactly this reason.

`navix.agents.agent.derive_episodic_metrics` (formerly defined here)
builds the `perf/*` keys `plot_metric`/`plot_dashboard` expect in
`logs` - it moved out because it's load-bearing for `Experiment.
run_hparam_search`'s fitness computation too, not just plotting, so a
plotting-only module was the wrong home for it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np


MANDATORY_METRICS: Dict[str, str] = {
    "perf/returns": "Episodic Return",
    "perf/success_rate": "Success Rate",
    "perf/episode_length": "Episode Length",
    "iter/fps": "Training Throughput (steps/s)",
    "iter/wall_time": "Wall-clock Training Time (s)",
}
"""The plots every navix agent's `logs` should support, so results are
directly comparable across algorithms. Kept intentionally small: only
metrics that (a) exist regardless of which algorithm produced `logs`, and
(b) are actually necessary to tell whether training worked at all."""


def plot_metric(
    logs: Dict[str, jnp.ndarray],
    key: str,
    title: Optional[str] = None,
    x_key: str = "iter/frames",
    xlabel: str = "Frames",
    ax=None,
):
    """Plots a single metric against `x_key`, aggregated with a mean line and
    a min-max shaded band over any leading batch dimensions (e.g. seeds).

    Args:
        logs (Dict[str, Array]): The `logs` pytree, as returned by
            `navix.agents.agent.derive_episodic_metrics` (for `perf/*`
            keys) or directly from `Experiment.run()` (for raw keys
            like `loss/*`, `iter/*`).
        key (str): The key in `logs` to plot.
        title (str, optional): The plot title. Defaults to `key`.
        x_key (str): The key in `logs` to use as the x-axis.
        xlabel (str): The x-axis label.
        ax (matplotlib.axes.Axes, optional): An existing axes to draw
            into. If `None`, a new standalone figure and axes are created.

    Returns:
        matplotlib.figure.Figure: The figure `ax` belongs to."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    x = jnp.asarray(logs[x_key])
    y = jnp.asarray(logs[key])
    # collapse any leading batch dims (e.g. seeds) into one axis, keeping
    # the last axis as the training-progress axis
    x = x.reshape(-1, x.shape[-1])
    y = y.reshape(-1, y.shape[-1])

    ax.plot(x[0], jnp.mean(y, axis=0), color="C0")
    if y.shape[0] > 1:
        ax.fill_between(
            x[0], jnp.min(y, axis=0), jnp.max(y, axis=0), color="C0", alpha=0.2
        )
    ax.set_title(title or key)
    ax.set_xlabel(xlabel)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_metrics(
    logs: Dict[str, jnp.ndarray],
    metrics: Dict[str, str],
    x_key: str = "iter/frames",
    xlabel: str = "Frames",
) -> Dict[str, "plt.Figure"]:
    """Plots each metric in `metrics` as its own standalone figure.

    Args:
        logs (Dict[str, Array]): The `logs` pytree (see `plot_metric`).
        metrics (Dict[str, str]): A mapping of `logs` key to plot title,
            e.g. `MANDATORY_METRICS`, or a leaderboard-side mapping of
            algorithm -> diagnostic keys.
        x_key (str): The key in `logs` to use as the x-axis.
        xlabel (str): The x-axis label.

    Returns:
        Dict[str, Figure]: One figure per metric, keyed by the same key
        as `metrics`. Keys missing from `logs` are silently skipped."""
    return {
        key: plot_metric(logs, key, title=title, x_key=x_key, xlabel=xlabel)
        for key, title in metrics.items()
        if key in logs
    }


def plot_dashboard(
    logs: Dict[str, jnp.ndarray],
    metrics: Optional[Dict[str, str]] = None,
    x_key: str = "iter/frames",
    xlabel: str = "Frames",
):
    """Plots `metrics` (`MANDATORY_METRICS` by default) as a single combined
    figure, one panel per metric.

    This is intentionally agnostic about "mandatory vs. diagnostic" -
    navix doesn't know, and shouldn't need to know, what a given algorithm
    considers diagnostic (an algorithm submitted to a leaderboard won't
    necessarily have any navix-specific code to declare that in). That
    categorisation belongs to whatever consumes this module - e.g. a
    leaderboard's own file mapping algorithm -> diagnostic keys - which
    can merge its own metrics dict with `MANDATORY_METRICS` and pass the
    result here.

    Args:
        logs (Dict[str, Array]): The `logs` pytree (see `plot_metric`).
        metrics (Dict[str, str], optional): A mapping of `logs` key to
            plot title. Defaults to `MANDATORY_METRICS`. Keys missing
            from `logs` are silently skipped.
        x_key (str): The key in `logs` to use as the x-axis.
        xlabel (str): The x-axis label.

    Returns:
        matplotlib.figure.Figure: The combined dashboard figure."""
    import matplotlib.pyplot as plt

    present = [(k, t) for k, t in (metrics or MANDATORY_METRICS).items() if k in logs]
    n_cols = max(len(present), 1)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4), squeeze=False)

    for ax in axes.flat:
        ax.axis("off")

    for i, (key, title) in enumerate(present):
        axes[0, i].axis("on")
        plot_metric(logs, key, title=title, x_key=x_key, xlabel=xlabel, ax=axes[0, i])

    fig.tight_layout()
    return fig


# -------------------------
# Benchmark.summary()/details()/diagnostics.npz - local, offline
# inspection of a submission, independent of the online leaderboard's
# own charts (see Benchmark.plot_summary/plot_details/plot_diagnostics,
# which call into these with a real BenchmarkResult's data).
# -------------------------


def _format_scalar(value: Any) -> str:
    array = np.asarray(value)
    if array.ndim != 0:
        return str(value)
    return f"{float(array):.4g}" if np.issubdtype(array.dtype, np.floating) else str(array.item())


def plot_benchmark_summary(summary: Dict[str, Any], title: str = "Summary") -> "plt.Figure":
    """A `Benchmark.summary()` dict as a metric/value table.

    A bar chart would be misleading here: `summary`'s metrics live on
    wildly different scales in the same dict (episodic returns in
    `[0, 1]` next to `flops` in the hundreds of millions), so a table
    keeps every value legible without implying they're comparable.

    Args:
        summary (Dict[str, Any]): `Benchmark.summary(results)`'s
            output (or the `"summary"` entry of a `summary.json`
            already loaded from disk).
        title (str): The figure title.

    Returns:
        matplotlib.figure.Figure: The table figure."""
    import matplotlib.pyplot as plt

    rows = [(key, _format_scalar(value)) for key, value in summary.items()]
    fig, ax = plt.subplots(figsize=(6, 0.4 * max(len(rows), 1) + 1))
    ax.axis("off")
    table = ax.table(cellText=rows or [["", ""]], colLabels=["metric", "value"], loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def _row_labels(details: Dict[str, Any]) -> Tuple[Optional[str], List[str]]:
    """The key in `details` that labels each row (e.g. `env_ids`) - the
    first key whose value is a non-empty sequence of strings. Falls
    back to positional labels if `details` has none (every value is
    numeric)."""
    for key, value in details.items():
        if isinstance(value, (list, tuple)) and value and all(isinstance(v, str) for v in value):
            return key, list(value)
    n = len(next(iter(details.values()), []))
    return None, [str(i) for i in range(n)]


def _is_numeric_sequence(value: Any) -> bool:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return False
    return array.ndim >= 1


def plot_benchmark_details(details: Dict[str, Any]) -> "plt.Figure":
    """A `Benchmark.details()` dict as one bar chart per numeric
    metric, one bar per row (e.g. one bar per environment for
    `FromScratchBenchmark`) - mean plus a std-dev error bar over
    whatever trailing axis `details` keeps raw (e.g.
    `FromScratchBenchmark` keeps every seed's own value, unlike
    `summary`'s already-averaged numbers - see
    `FromScratchBenchmark.details`'s docstring).

    Args:
        details (Dict[str, Any]): `Benchmark.details(results)`'s
            output (or the `"details"` entry of a `details.json`
            already loaded from disk).

    Returns:
        matplotlib.figure.Figure: One panel per numeric metric."""
    import matplotlib.pyplot as plt

    label_key, labels = _row_labels(details)
    numeric = {key: value for key, value in details.items() if key != label_key and _is_numeric_sequence(value)}
    n_cols = max(len(numeric), 1)
    fig, axes = plt.subplots(1, n_cols, figsize=(max(4.0, len(labels) * 0.6) * n_cols, 4), squeeze=False)

    for i, (key, value) in enumerate(numeric.items()):
        ax = axes[0, i]
        array = np.asarray(value, dtype=float)
        if array.ndim > 1:
            array = array.reshape(array.shape[0], -1)
            means, stds = np.mean(array, axis=-1), np.std(array, axis=-1)
        else:
            means, stds = array, None
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=stds, color="C0", capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(key)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return fig


_NON_CURVE_DIAGNOSTICS_KEYS = frozenset({"wall_time", "fps", "flops", "memory_bytes", "compile_time_seconds"})


def plot_benchmark_diagnostics(diagnostics: Dict[str, Any], xlabel: str = "Training progress (%)") -> "plt.Figure":
    """A `BenchmarkResult.curve`-shaped diagnostics dict (`episodic_
    returns`, `length`, any `diagnostics_<key>` free-form curves - the
    same keys `Benchmark.submit_entry` writes into `diagnostics.npz`,
    so this also plots one already loaded back with `np.load`) as one
    curve panel per key: mean line plus a min-max band over any
    leading batch dimension (e.g. seeds) - same convention as
    `plot_metric`.

    A `TrainingCurve` doesn't carry absolute frame counts (unlike the
    raw `logs` pytree `plot_metric` plots), so the x-axis is training
    progress as a 0-100% fraction of however many points the curve
    has, not a frame count.

    Args:
        diagnostics (Dict[str, Any]): Curve arrays keyed by name, plus
            optionally the scalar cost fields `Benchmark.submit_entry`
            also writes (`wall_time`, `fps`, `flops`, `memory_bytes`,
            `compile_time_seconds`) - these are skipped, they aren't
            curves.
        xlabel (str): The x-axis label.

    Returns:
        matplotlib.figure.Figure: One panel per curve."""
    import matplotlib.pyplot as plt

    curve_keys = [
        key
        for key, value in diagnostics.items()
        if key not in _NON_CURVE_DIAGNOSTICS_KEYS and np.asarray(value).ndim >= 1
    ]
    n_cols = max(len(curve_keys), 1)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4), squeeze=False)

    for i, key in enumerate(curve_keys):
        ax = axes[0, i]
        y = np.asarray(diagnostics[key], dtype=float)
        y = y.reshape(-1, y.shape[-1])
        x = np.linspace(0, 100, y.shape[-1])
        ax.plot(x, np.mean(y, axis=0), color="C0")
        if y.shape[0] > 1:
            ax.fill_between(x, np.min(y, axis=0), np.max(y, axis=0), color="C0", alpha=0.2)
        ax.set_title(key)
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig
