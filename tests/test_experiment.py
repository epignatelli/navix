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

import pytest
import wandb
import jax.numpy as jnp

from navix.agents.agent import Agent, HParams
from navix.experiment import Experiment


class _FakeAgent(Agent):
    """A minimal Agent whose `train` is a no-op over the environment, so
    `Experiment.run`'s logging loop can be exercised without a real
    training loop or a real wandb backend."""

    def train(self, rng):
        n_updates = 3
        logs = {
            "iter/updates": jnp.arange(n_updates),
            "iter/frames": jnp.arange(n_updates) * 100,
        }
        return None, logs


class _FakeRun:
    def __init__(self, seed, logged, finished):
        self.seed = seed
        self._logged = logged
        self._finished = finished

    def log(self, logs, step=None):
        self._logged.append(self.seed)

    def finish(self):
        self._finished.append(self.seed)


def test_experiment_run_logs_every_seed_via_its_own_run(monkeypatch):
    # Experiment.run logs each seed sequentially - wandb.init() -> log ->
    # finish, one seed at a time, using each seed's own wandb.init()-
    # returned Run object rather than the thread-unsafe module-level
    # wandb.log (see navix/experiment.py's module docstring/comment for
    # why: thread- and process-based concurrency were both tried and
    # reverted - threads corrupted runs, processes were slower than
    # sequential up to ~4-8 seeds and crashed at 16). This checks every
    # seed still gets logged and finished exactly once.
    logged, finished = [], []

    def fake_init(project, config, group):
        return _FakeRun(config["seed"], logged, finished)

    monkeypatch.setattr(wandb, "init", fake_init)

    seeds = (0, 1, 2, 3)
    experiment = Experiment(
        name="test",
        agent=_FakeAgent(hparams=HParams(log_frequency=1), env=None),  # type: ignore[arg-type]
        env=None,  # type: ignore[arg-type]
        seeds=seeds,
    )
    experiment.run(log_to_wandb=True)

    assert finished == list(seeds)
    # 3 updates per seed (see _FakeAgent.train), so each seed logs 3 times
    assert logged == [s for s in seeds for _ in range(3)]


def test_experiment_run_do_log_is_deprecated_but_still_works(monkeypatch):
    # the pre-rename `do_log` kwarg must keep working (mapped onto
    # log_to_wandb) so existing callers aren't broken by the rename, just
    # warned.
    logged, finished = [], []

    def fake_init(project, config, group):
        return _FakeRun(config["seed"], logged, finished)

    monkeypatch.setattr(wandb, "init", fake_init)

    seeds = (0, 1)
    experiment = Experiment(
        name="test",
        agent=_FakeAgent(hparams=HParams(log_frequency=1), env=None),  # type: ignore[arg-type]
        env=None,  # type: ignore[arg-type]
        seeds=seeds,
    )

    with pytest.warns(DeprecationWarning, match="log_to_wandb"):
        experiment.run(do_log=True)

    assert finished == list(seeds)

    logged.clear()
    finished.clear()
    with pytest.warns(DeprecationWarning, match="log_to_wandb"):
        experiment.run(do_log=False)

    assert finished == []


def test_experiment_run_with_no_seeds_does_not_crash(monkeypatch):
    # an empty `seeds` must stay a no-op - the logging loop just never
    # runs, wandb.init is never called.
    def fake_init(project, config, group):
        raise AssertionError("wandb.init should not be called for zero seeds")

    monkeypatch.setattr(wandb, "init", fake_init)

    experiment = Experiment(
        name="test",
        agent=_FakeAgent(hparams=HParams(log_frequency=1), env=None),  # type: ignore[arg-type]
        env=None,  # type: ignore[arg-type]
        seeds=(),
    )
    experiment.run(log_to_wandb=True)


if __name__ == "__main__":
    class _Monkeypatch:
        def setattr(self, obj, name, value):
            setattr(obj, name, value)

    test_experiment_run_logs_every_seed_via_its_own_run(_Monkeypatch())
    test_experiment_run_do_log_is_deprecated_but_still_works(_Monkeypatch())
    test_experiment_run_with_no_seeds_does_not_crash(_Monkeypatch())
