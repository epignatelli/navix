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

from unittest.mock import patch

import pytest
import numpy as np
import jax.numpy as jnp

from navix.agents.agent import Agent, HParams


def test_log_to_wandb_on_train_end_respects_log_frequency():
    # https://github.com/epignatelli/navix/issues/60
    # log_to_wandb_on_train_end used to index into every field of `logs`
    # (a device-to-host transfer per field) for every single recorded
    # step, even though log_to_wandb() would immediately discard most of
    # them via the log_frequency check - wandb.log() was correctly only
    # called for the kept steps, but the expensive tree indexing happened
    # regardless. This asserts the *set of steps actually logged* is
    # unchanged by hoisting that check earlier: still exactly the steps
    # where iter/updates % log_frequency == 0.
    n_steps = 10
    log_frequency = 3
    logs = {
        "iter/updates": jnp.arange(n_steps),
        "iter/frames": jnp.arange(n_steps) * 100,
    }
    agent = Agent(hparams=HParams(log_frequency=log_frequency))

    with patch("navix.agents.agent.wandb.log") as mock_log:
        agent.log_to_wandb_on_train_end(logs)

    logged_steps = [int(call.kwargs["step"]) for call in mock_log.call_args_list]
    expected_steps = [s for s in range(n_steps) if s % log_frequency == 0]
    assert logged_steps == expected_steps, (
        f"Expected wandb.log to be called for steps {expected_steps}, "
        f"got {logged_steps}"
    )


def test_log_on_train_end_is_deprecated_but_still_works():
    # the pre-rename name must keep working (delegating to
    # log_to_wandb_on_train_end) so existing callers aren't broken by the
    # rename, just warned.
    n_steps = 4
    logs = {
        "iter/updates": jnp.arange(n_steps),
        "iter/frames": jnp.arange(n_steps) * 100,
    }
    agent = Agent(hparams=HParams(log_frequency=1))

    with patch("navix.agents.agent.wandb.log") as mock_log:
        with pytest.warns(DeprecationWarning, match="log_to_wandb_on_train_end"):
            agent.log_on_train_end(logs)

    assert mock_log.call_count == n_steps


def test_log_to_wandb_masked_mean_matches_boolean_indexing():
    # https://github.com/epignatelli/navix/issues/60
    # `lengths[mask]` / `returns[mask]` (boolean-indexing a variable number
    # of completed episodes per step) produce a *dynamically*-shaped
    # output, forcing JAX to recompile a fresh XLA program for every
    # distinct episode count it hasn't seen before - profiling a real PPO
    # run showed this was ~60% of total logging wall-clock time, and
    # explained why wandb.log() calls got slower the more *different*
    # episode-completion patterns a run produced, not just their number.
    # Replaced with a masked sum/count, which keeps a static (T, N) ->
    # scalar shape regardless of how many entries are masked, so XLA
    # compiles it once. This checks the new arithmetic still matches
    # plain numpy boolean-indexing exactly.
    rng = np.random.default_rng(0)
    mask = rng.random((5, 4)) > 0.5
    lengths = rng.integers(1, 50, size=(5, 4)).astype(np.float32)
    returns = rng.choice([0.0, 1.0], size=(5, 4))
    # ensure at least one True and one entry equal to 1.0 under the mask,
    # so both branches are non-degenerate
    mask[0, 0] = True
    returns[0, 0] = 1.0

    expected_length = np.mean(lengths[mask])
    expected_returns = np.mean(returns[mask])
    expected_success_rate = np.mean(returns[mask] == 1.0)

    logs = {
        "iter/updates": jnp.asarray(0),
        "iter/frames": jnp.asarray(0),
        "done_mask": jnp.asarray(mask),
        "lengths": jnp.asarray(lengths),
        "returns": jnp.asarray(returns),
    }
    agent = Agent(hparams=HParams())

    with patch("navix.agents.agent.wandb.log") as mock_log:
        agent.log_to_wandb(dict(logs))

    logged = mock_log.call_args.args[0]
    assert np.allclose(logged["perf/episode_length"], expected_length)
    assert np.allclose(logged["perf/returns"], expected_returns)
    assert np.allclose(logged["perf/success_rate"], expected_success_rate)


def test_log_is_deprecated_but_still_works():
    # the pre-rename name must keep working (delegating to log_to_wandb)
    # so existing callers aren't broken by the rename, just warned.
    logs = {
        "iter/updates": jnp.asarray(0),
        "iter/frames": jnp.asarray(0),
    }
    agent = Agent(hparams=HParams())

    with patch("navix.agents.agent.wandb.log") as mock_log:
        with pytest.warns(DeprecationWarning, match="log_to_wandb"):
            agent.log(dict(logs))

    assert mock_log.call_count == 1


if __name__ == "__main__":
    test_log_to_wandb_on_train_end_respects_log_frequency()
    test_log_on_train_end_is_deprecated_but_still_works()
    test_log_to_wandb_masked_mean_matches_boolean_indexing()
    test_log_is_deprecated_but_still_works()
