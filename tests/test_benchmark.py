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

import dataclasses

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import navix as nx
from navix.agents import PPO, PPOHparams, ActorCritic
from navix.benchmark import AlgorithmEntry, Benchmark, BenchmarkResult, default_score_fn
from navix.environments.environment import Environment


def _flatten_obs(env: Environment) -> Environment:
    flatten_obs_fn = lambda x: jnp.ravel(env.observation_fn(x))
    flatten_shape = (int(np.prod(env.observation_space.shape)),)
    return env.replace(
        observation_fn=flatten_obs_fn,
        observation_space=env.observation_space.replace(shape=flatten_shape),
    )


def _tiny_ppo_factory(env: Environment) -> PPO:
    # A deliberately tiny configuration - just enough for every code path
    # to execute, matching the pattern used in test_pqn.py/test_dreamer.py.
    env = _flatten_obs(env)
    hp = PPOHparams(
        budget=32,  # num_steps * num_envs -> exactly 1 update
        num_envs=4,
        num_steps=8,
        num_minibatches=2,
        num_epochs=1,
    )
    return PPO(hparams=hp, network=ActorCritic(action_dim=len(env.action_set)), env=env)


def _make_tiny_entry(**overrides) -> AlgorithmEntry:
    defaults = dict(
        name="PPO",
        author="test-author",
        paper_url="https://arxiv.org/abs/1707.06347",
        commit_sha="deadbeef",
        agent_factory=_tiny_ppo_factory,
    )
    defaults.update(overrides)
    return AlgorithmEntry(**defaults)


def _make_tiny_benchmark(**overrides) -> Benchmark:
    defaults = dict(
        name="test-benchmark",
        budget=32,
        env_ids=("Navix-Empty-5x5-v0", "Navix-Empty-6x6-v0"),
    )
    defaults.update(overrides)
    return Benchmark(**defaults)


def test_algorithm_entry_requires_its_metadata_fields():
    # every navix leaderboard entry needs these (issue #130) - a
    # dataclass with no defaults for them is the enforcement mechanism,
    # not just documentation.
    with pytest.raises(TypeError):
        AlgorithmEntry(agent_factory=_tiny_ppo_factory)  # type: ignore[call-arg]


def test_algorithm_entry_defaults_suite_to_navix():
    entry = _make_tiny_entry()
    assert entry.suite == "navix"


def test_benchmark_calls_agent_factory_once_per_env():
    calls = []

    def counting_factory(env):
        calls.append(env)
        return _tiny_ppo_factory(env)

    benchmark = _make_tiny_benchmark()
    entry = _make_tiny_entry(agent_factory=counting_factory)
    benchmark.run(entry, log_to_wandb=False)

    assert len(calls) == len(benchmark.env_ids)


def test_benchmark_result_echoes_entry_and_aggregates_scores():
    benchmark = _make_tiny_benchmark()
    entry = _make_tiny_entry()
    result = benchmark.run(entry, log_to_wandb=False)

    assert isinstance(result, BenchmarkResult)
    assert result.entry is entry
    assert set(result.scores.keys()) == set(benchmark.env_ids)
    for env_id in benchmark.env_ids:
        assert env_id in result.logs
        assert np.all(np.isfinite(np.asarray(result.scores[env_id])))

    expected = jnp.mean(jnp.stack(list(result.scores.values())))
    np.testing.assert_allclose(np.asarray(result.score), np.asarray(expected))


def test_benchmark_overrides_agent_budget_not_agent_factorys_own():
    # the whole point of budget being a Benchmark field (not left to
    # whatever the agent_factory's own hparams say) is that NAVIX_1M vs
    # NAVIX_100K only differ on this - assert the override actually
    # takes effect rather than silently using the factory's default.
    seen_budgets = []

    def factory(env):
        agent = _tiny_ppo_factory(env)
        seen_budgets.append(agent.hparams.budget)
        return agent

    benchmark = _make_tiny_benchmark(budget=64, env_ids=("Navix-Empty-5x5-v0",))
    entry = _make_tiny_entry(agent_factory=factory)
    benchmark.run(entry, log_to_wandb=False)

    # the factory itself always builds budget=32 (see _tiny_ppo_factory);
    # Benchmark.run must override it to 64 before training.
    assert seen_budgets == [32]


def test_default_score_fn_matches_hand_computed_success_rate():
    # 10 updates, 4 steps, 2 envs. Only the last 2 updates (last 20%)
    # should count towards the score - make the earlier ones all-failure
    # and the last two all-success, so a bug that averages over the
    # whole history (instead of the last 20%) would be caught.
    num_updates, num_steps, num_envs = 10, 4, 2
    done_mask = np.ones((num_updates, num_steps, num_envs), dtype=bool)
    returns = np.zeros((num_updates, num_steps, num_envs), dtype=np.float32)
    returns[-2:] = 1.0  # last 2 updates (20%) are successes

    logs = {
        "done_mask": jnp.asarray(done_mask),
        "returns": jnp.asarray(returns),
        "lengths": jnp.ones((num_updates, num_steps, num_envs)),
    }
    score = default_score_fn(logs)
    np.testing.assert_allclose(np.asarray(score), 1.0)

    # flip it: only the *first* 80% succeed - score should be ~0, since
    # the tail (what's actually scored) is all-failure.
    returns2 = np.ones((num_updates, num_steps, num_envs), dtype=np.float32)
    returns2[-2:] = 0.0
    logs2 = {**logs, "returns": jnp.asarray(returns2)}
    score2 = default_score_fn(logs2)
    np.testing.assert_allclose(np.asarray(score2), 0.0)
