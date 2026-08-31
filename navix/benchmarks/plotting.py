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
False)` and reading `logs` directly is the fast path). A `Benchmark`-
scored run's own `summary`/`details`/`diagnostics` plots
(`Benchmark.plot_summary`/`plot_details`/`plot_diagnostics`) live in
`navix/benchmarks/benchmark.py` instead, next to the `Benchmark`
methods they render - not here, since `Benchmark` is the one place
that already knows those shapes.

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

from typing import Dict, Optional

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
