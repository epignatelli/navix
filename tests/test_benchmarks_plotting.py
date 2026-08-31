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

import matplotlib

matplotlib.use("Agg")  # headless backend, no display needed for tests
import matplotlib.pyplot as plt

import numpy as np
import jax.numpy as jnp

from navix.agents.agent import derive_episodic_metrics
from navix.benchmarks.plotting import (
    MANDATORY_METRICS,
    plot_metric,
    plot_metrics,
    plot_dashboard,
)


def _make_logs(n_seeds=2, n_updates=3, n_steps=4, n_envs=5, seed=0):
    rng = np.random.default_rng(seed)
    shape = (n_seeds, n_updates, n_steps, n_envs)
    mask = rng.random(shape) > 0.5
    mask[:, :, 0, 0] = True  # avoid an all-False mask (division by zero)
    lengths = rng.integers(1, 50, size=shape).astype(np.float32)
    returns = rng.choice([0.0, 1.0], size=shape)
    frames = np.tile(np.arange(n_updates) * 100, (n_seeds, 1))
    return {
        "train/frames": jnp.asarray(frames),
        "iter/fps": jnp.asarray(rng.random((n_seeds, n_updates)) * 1000),
        "iter/wall_time": jnp.asarray(rng.random((n_seeds, n_updates)) * 100),
        "train/done_mask": jnp.asarray(mask),
        "train/lengths": jnp.asarray(lengths),
        "train/returns": jnp.asarray(returns),
        "loss/total_loss": jnp.asarray(rng.random((n_seeds, n_updates))),
    }, mask, lengths, returns


def test_plot_metric_returns_figure_with_data():
    logs, *_ = _make_logs()
    logs = derive_episodic_metrics(logs)
    fig = plot_metric(logs, "episode/returns", x_key="train/frames")
    assert isinstance(fig, plt.Figure)
    ax = fig.axes[0]
    assert len(ax.lines) == 1
    plt.close(fig)


def test_plot_metrics_skips_missing_keys():
    logs, *_ = _make_logs()
    logs = derive_episodic_metrics(logs)
    figs = plot_metrics(logs, MANDATORY_METRICS, x_key="train/frames")
    # episode/length, episode/returns, episode/success_rate, iter/fps,
    # iter/wall_time all present
    assert set(figs.keys()) == set(MANDATORY_METRICS.keys())
    for fig in figs.values():
        plt.close(fig)


def test_plot_dashboard_defaults_to_mandatory_metrics():
    logs, *_ = _make_logs()
    logs = derive_episodic_metrics(logs)
    fig = plot_dashboard(logs, x_key="train/frames")
    on_axes = [ax for ax in fig.axes if ax.axison]
    assert len(on_axes) == len(MANDATORY_METRICS)
    plt.close(fig)


def test_plot_dashboard_accepts_an_arbitrary_metrics_dict():
    # navix.benchmarks.plotting doesn't know about "diagnostic" metrics - a caller
    # (e.g. a leaderboard's own algorithm -> diagnostic-keys mapping) can
    # pass whatever combined dict it wants
    logs, *_ = _make_logs()
    logs = derive_episodic_metrics(logs)
    custom_metrics = {**MANDATORY_METRICS, "loss/total_loss": "Total Loss"}

    fig = plot_dashboard(logs, metrics=custom_metrics, x_key="train/frames")
    on_axes = [ax for ax in fig.axes if ax.axison]
    assert len(on_axes) == len(custom_metrics)
    plt.close(fig)
