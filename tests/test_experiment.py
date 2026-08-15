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

# Note on what's (not) unit-tested here: Experiment.run/run_hparam_search
# log each seed/hparam-set from its own spawned OS process (see
# navix.experiment._log_run_to_wandb's docstring for why). A spawned
# process re-imports `wandb` from scratch, so `monkeypatch.setattr(wandb,
# "init", ...)` in this (parent) process has no effect on it - unlike the
# old thread-based version of this file, there's no way to fake wandb
# inside the child from here. The multiprocessing plumbing itself (does
# ProcessPoolExecutor(mp_context=spawn) actually reach every seed) is
# exercised for real by examples/hparam_search.py in CI's "Run examples"
# step instead. What's tested here is what's safely testable without
# spawning a real process or touching a real wandb backend: the worker
# function's own logic (_log_run_to_wandb, called directly, in-process),
# the numpy-conversion helper, and the parts of Experiment.run that don't
# require any of that (the do_log deprecation path, the empty-seeds path).

import pytest
import wandb
import numpy as np
import jax.numpy as jnp

from navix.agents.agent import Agent, HParams
from navix.experiment import Experiment, _log_run_to_wandb, _to_numpy


class _FakeAgent(Agent):
    """A minimal Agent whose `train` is a no-op over the environment, so
    `Experiment.run`'s non-logging code paths can be exercised without a
    real training loop."""

    def train(self, rng):
        n_updates = 3
        logs = {
            "iter/updates": jnp.arange(n_updates),
            "iter/frames": jnp.arange(n_updates) * 100,
        }
        return None, logs


class _FakeRun:
    def __init__(self):
        self.logged = []
        self.finished = False

    def log(self, logs, step=None):
        self.logged.append(step)

    def finish(self):
        self.finished = True


def test_to_numpy_converts_jax_arrays_only():
    jax_array = jnp.arange(3)
    assert isinstance(_to_numpy(jax_array), np.ndarray)
    # non-array values (the pytree_node=False fields on HParams, plain
    # Python scalars, ...) must pass through unchanged, not get coerced
    assert _to_numpy(True) is True
    assert _to_numpy(3) == 3
    assert _to_numpy("x") == "x"


def test_log_run_to_wandb_calls_init_log_finish(monkeypatch):
    fake_run = _FakeRun()
    init_calls = []

    def fake_init(project, config, group):
        init_calls.append((project, config, group))
        return fake_run

    monkeypatch.setattr(wandb, "init", fake_init)

    log_np = {
        "iter/updates": np.arange(4),
        "iter/frames": np.arange(4) * 100,
    }
    _log_run_to_wandb(HParams(log_frequency=1), "proj", {"seed": 0}, "grp", log_np)

    assert init_calls == [("proj", {"seed": 0}, "grp")]
    assert len(fake_run.logged) == 4
    assert fake_run.finished


def test_experiment_run_do_log_is_deprecated_but_still_works():
    # the pre-rename `do_log` kwarg must keep working (mapped onto
    # log_to_wandb) so existing callers aren't broken by the rename, just
    # warned. hparams.debug=True short-circuits the actual logging block
    # (`if not self.agent.hparams.debug and log_to_wandb:`) regardless of
    # do_log's value, so this doesn't need to spawn a process or touch
    # wandb at all - just checks the warning fires and run() still
    # completes.
    experiment = Experiment(
        name="test",
        agent=_FakeAgent(hparams=HParams(log_frequency=1, debug=True)),
        env=None,  # type: ignore[arg-type]
        seeds=(0, 1),
    )

    with pytest.warns(DeprecationWarning, match="log_to_wandb"):
        experiment.run(do_log=True)


def test_experiment_run_with_no_seeds_does_not_crash():
    # an empty `seeds` must stay a no-op (no process ever spawned, since
    # the loop over self.seeds never runs), not crash.
    experiment = Experiment(
        name="test",
        agent=_FakeAgent(hparams=HParams(log_frequency=1)),
        env=None,  # type: ignore[arg-type]
        seeds=(),
    )
    experiment.run(log_to_wandb=True)


if __name__ == "__main__":
    class _Monkeypatch:
        def setattr(self, obj, name, value):
            setattr(obj, name, value)

    test_to_numpy_converts_jax_arrays_only()
    test_log_run_to_wandb_calls_init_log_finish(_Monkeypatch())
    test_experiment_run_do_log_is_deprecated_but_still_works()
    test_experiment_run_with_no_seeds_does_not_crash()
