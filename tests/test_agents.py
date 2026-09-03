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
import jax
import jax.numpy as jnp

import navix as nx
from navix.agents.agent import Agent, HParams, derive_episodic_metrics, stack_frame_history

# None of the tests below exercise anything env-specific (they only call
# log_to_wandb/log_to_wandb_on_train_end, which never touch self.env) -
# any real, registered env satisfies Agent's required `env` field.
_DUMMY_ENV = nx.make("Navix-Empty-5x5-v0")


def _make_logs(n_seeds=2, n_updates=3, n_steps=4, n_envs=5, seed=0):
    rng = np.random.default_rng(seed)
    shape = (n_seeds, n_updates, n_steps, n_envs)
    mask = rng.random(shape) > 0.5
    mask[:, :, 0, 0] = True  # avoid an all-False mask (division by zero)
    lengths = rng.integers(1, 50, size=shape).astype(np.float32)
    returns = rng.choice([0.0, 1.0], size=shape)
    return {
        "agent/train/done_mask": jnp.asarray(mask),
        "agent/train/lengths": jnp.asarray(lengths),
        "agent/train/returns": jnp.asarray(returns),
    }, mask, lengths, returns


def test_derive_episodic_metrics_matches_manual_masked_mean():
    logs, mask, lengths, returns = _make_logs()
    n_seeds, n_updates = mask.shape[:2]

    derived = derive_episodic_metrics(logs)

    for s in range(n_seeds):
        for u in range(n_updates):
            m = mask[s, u]
            expected_length = np.mean(lengths[s, u][m])
            expected_returns = np.mean(returns[s, u][m])
            expected_success = np.mean(returns[s, u][m] == 1.0)
            assert np.allclose(derived["agent/episode/length"][s, u], expected_length)
            assert np.allclose(derived["agent/episode/returns"][s, u], expected_returns)
            assert np.allclose(derived["agent/episode/success_rate"][s, u], expected_success)

    # original logs dict must not be mutated
    assert "agent/episode/returns" not in logs


def test_derive_episodic_metrics_raises_on_missing_keys():
    logs = {"agent/train/done_mask": jnp.ones((2,), dtype=bool)}
    with pytest.raises(KeyError, match="lengths.*returns"):
        derive_episodic_metrics(logs)


def test_log_to_wandb_on_train_end_respects_log_frequency():
    # https://github.com/epignatelli/navix/issues/60
    # log_to_wandb_on_train_end used to index into every field of `logs`
    # (a device-to-host transfer per field) for every single recorded
    # step, even though log_to_wandb() would immediately discard most of
    # them via the log_frequency check - wandb.log() was correctly only
    # called for the kept steps, but the expensive tree indexing happened
    # regardless. This asserts the *set of steps actually logged* is
    # unchanged by hoisting that check earlier: still exactly the steps
    # where agent/train/updates % log_frequency == 0.
    n_steps = 10
    log_frequency = 3
    logs = {
        "agent/train/updates": jnp.arange(n_steps),
        "agent/train/frames": jnp.arange(n_steps) * 100,
    }
    agent = Agent(hparams=HParams(log_frequency=log_frequency), env=_DUMMY_ENV)

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
        "agent/train/updates": jnp.arange(n_steps),
        "agent/train/frames": jnp.arange(n_steps) * 100,
    }
    agent = Agent(hparams=HParams(log_frequency=1), env=_DUMMY_ENV)

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
        "agent/train/updates": jnp.asarray(0),
        "agent/train/frames": jnp.asarray(0),
        "agent/train/done_mask": jnp.asarray(mask),
        "agent/train/lengths": jnp.asarray(lengths),
        "agent/train/returns": jnp.asarray(returns),
    }
    agent = Agent(hparams=HParams(), env=_DUMMY_ENV)

    with patch("navix.agents.agent.wandb.log") as mock_log:
        agent.log_to_wandb(dict(logs))

    logged = mock_log.call_args.args[0]
    assert np.allclose(logged["agent/episode/length"], expected_length)
    assert np.allclose(logged["agent/episode/returns"], expected_returns)
    assert np.allclose(logged["agent/episode/success_rate"], expected_success_rate)


def test_log_is_deprecated_but_still_works():
    # the pre-rename name must keep working (delegating to log_to_wandb)
    # so existing callers aren't broken by the rename, just warned.
    logs = {
        "agent/train/updates": jnp.asarray(0),
        "agent/train/frames": jnp.asarray(0),
    }
    agent = Agent(hparams=HParams(), env=_DUMMY_ENV)

    with patch("navix.agents.agent.wandb.log") as mock_log:
        with pytest.warns(DeprecationWarning, match="log_to_wandb"):
            agent.log(dict(logs))

    assert mock_log.call_count == 1


def test_stack_frame_history_no_episode_boundary():
    # obs = [0, 1, 2, 3, 4, 5], context=3, no done anywhere - each
    # window is simply the last 3 frames, left-padded by repeating
    # obs[0] until enough real history has accumulated.
    obs = jnp.arange(6, dtype=jnp.float32)[:, None]  # (6, 1)
    done = jnp.zeros((6,), dtype=jnp.bool_)

    windows = stack_frame_history(obs, done, context=3)

    assert windows.shape == (6, 3, 1)
    expected = jnp.asarray(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
        ],
        dtype=jnp.float32,
    )[:, :, None]
    assert jnp.array_equal(windows, expected)


def test_stack_frame_history_does_not_leak_across_episode_boundary():
    # two episodes back to back: [0, 1, 2 (done)], [3, 4, 5]. context=3.
    # windows[3] (the first frame of episode 2) must be [3, 3, 3], never
    # reaching back to read 2/1/0 from the episode that just ended.
    obs = jnp.arange(6, dtype=jnp.float32)[:, None]
    done = jnp.asarray([False, False, True, False, False, False])

    windows = stack_frame_history(obs, done, context=3)

    assert jnp.array_equal(windows[3, :, 0], jnp.asarray([3.0, 3.0, 3.0]))
    assert jnp.array_equal(windows[4, :, 0], jnp.asarray([3.0, 3.0, 4.0]))
    assert jnp.array_equal(windows[5, :, 0], jnp.asarray([3.0, 4.0, 5.0]))
    # episode 1's own windows are unaffected by what comes after it.
    assert jnp.array_equal(windows[2, :, 0], jnp.asarray([0.0, 1.0, 2.0]))


def test_stack_frame_history_jit_vmap_compatible():
    key = jax.random.PRNGKey(0)
    num_envs, T, context = 4, 10, 3
    obs = jax.random.uniform(key, (num_envs, T, 2, 2))
    done = jax.random.bernoulli(key, 0.2, (num_envs, T))

    fn = jax.jit(jax.vmap(stack_frame_history, in_axes=(0, 0, None)), static_argnums=(2,))
    windows = fn(obs, done, context)
    assert windows.shape == (num_envs, T, context, 2, 2)


if __name__ == "__main__":
    test_log_to_wandb_on_train_end_respects_log_frequency()
    test_log_on_train_end_is_deprecated_but_still_works()
    test_log_to_wandb_masked_mean_matches_boolean_indexing()
    test_log_is_deprecated_but_still_works()
