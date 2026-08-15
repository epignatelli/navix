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
wandb (see `Agent.log`'s docstring and issue #60: disabling wandb logging
and reading `logs` directly is the fast path).

Two fixed, deliberately-chosen sets of plots, rather than auto-detecting
whatever keys happen to be in `logs`:

- `MANDATORY_METRICS`: identical across every navix agent (`perf/returns`,
  `perf/success_rate`, `perf/episode_length`, `iter/fps`), so results are
  visually comparable across algorithms - this is the set the navix
  leaderboard (#130) is expected to standardise on.
- Diagnostic metrics are algorithm-specific ("inner machinery" - e.g. a
  PPO's clip fraction, an off-policy agent's buffer size) and are supplied
  by the caller (see `PPO.DIAGNOSTIC_METRICS` for an example), not fixed
  here.
"""

from __future__ import annotations

from typing import Dict, Optional

import jax.numpy as jnp

from .agents.agent import masked_mean


MANDATORY_METRICS: Dict[str, str] = {
    "perf/returns": "Episodic Return",
    "perf/success_rate": "Success Rate",
    "perf/episode_length": "Episode Length",
    "iter/fps": "Training Throughput (steps/s)",
}
"""The plots every navix agent's `logs` should support, so results are
directly comparable across algorithms. Kept intentionally small: only
metrics that (a) exist regardless of which algorithm produced `logs`, and
(b) are actually necessary to tell whether training worked at all."""


def derive_scalar_metrics(logs: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
    """Computes the `perf/*` entries of `MANDATORY_METRICS` from the raw
    per-step buffers (`done_mask`, `lengths`, `returns`) that `Agent.train`
    returns, for every seed and update at once.

    `Agent.log` computes the same values, but one training update at a
    time (for live wandb logging); this is the batched equivalent, for
    plotting an entire already-finished `logs` history in one call.

    Args:
        logs (Dict[str, Array]): The `logs` pytree returned by
            `Experiment.run()`, `Experiment.run_hparam_search()`, or a bare
            `Agent.train()` call. Shapes are `(..., num_steps, num_envs)`
            for `done_mask`/`lengths`/`returns` - any number of leading
            batch dimensions (e.g. seeds, hparam sets) is supported.

    Returns:
        Dict[str, Array]: `logs`, plus `perf/episode_length`,
        `perf/returns` and `perf/success_rate` (shape: `logs`' leading
        batch dimensions, with `num_steps` and `num_envs` reduced away),
        if `done_mask` was present. `logs` itself is not mutated."""
    if "done_mask" not in logs:
        return logs

    metrics = dict(logs)
    mask = jnp.asarray(logs["done_mask"], dtype=jnp.bool_)
    if "lengths" in logs:
        metrics["perf/episode_length"] = masked_mean(logs["lengths"], mask, axis=(-2, -1))
    if "returns" in logs:
        returns = logs["returns"]
        metrics["perf/returns"] = masked_mean(returns, mask, axis=(-2, -1))
        metrics["perf/success_rate"] = masked_mean(returns == 1.0, mask, axis=(-2, -1))
    return metrics


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
            `derive_scalar_metrics` (for `perf/*` keys) or directly from
            `Experiment.run()` (for raw keys like `loss/*`, `iter/*`).
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
            e.g. `MANDATORY_METRICS` or an agent's `DIAGNOSTIC_METRICS`.
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
    diagnostic_metrics: Optional[Dict[str, str]] = None,
    x_key: str = "iter/frames",
    xlabel: str = "Frames",
):
    """Plots `MANDATORY_METRICS` and, if given, `diagnostic_metrics`, as a
    single combined figure - one row for the mandatory metrics (comparable
    across algorithms), and, if any diagnostic metrics are present, a
    second row for them (algorithm-specific).

    Args:
        logs (Dict[str, Array]): The `logs` pytree (see `plot_metric`).
        diagnostic_metrics (Dict[str, str], optional): Algorithm-specific
            metrics to plot alongside the mandatory ones, e.g.
            `PPO.DIAGNOSTIC_METRICS`. Keys missing from `logs` are
            silently skipped.
        x_key (str): The key in `logs` to use as the x-axis.
        xlabel (str): The x-axis label.

    Returns:
        matplotlib.figure.Figure: The combined dashboard figure."""
    import matplotlib.pyplot as plt

    mandatory = [(k, t) for k, t in MANDATORY_METRICS.items() if k in logs]
    diagnostic = [
        (k, t) for k, t in (diagnostic_metrics or {}).items() if k in logs
    ]

    # size the grid by whichever row has more entries, so no diagnostic
    # metric is ever silently dropped for having "too many" columns
    n_cols = max(len(mandatory), len(diagnostic), 1)
    n_rows = 1 + (1 if diagnostic else 0)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)

    for ax in axes.flat:
        ax.axis("off")

    for i, (key, title) in enumerate(mandatory):
        axes[0, i].axis("on")
        plot_metric(logs, key, title=title, x_key=x_key, xlabel=xlabel, ax=axes[0, i])

    for i, (key, title) in enumerate(diagnostic):
        axes[1, i].axis("on")
        plot_metric(logs, key, title=title, x_key=x_key, xlabel=xlabel, ax=axes[1, i])

    fig.tight_layout()
    return fig
