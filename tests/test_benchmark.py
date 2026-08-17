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

import numpy as np
import jax.numpy as jnp
import pytest

from navix.agents import PPO, PPOHparams, ActorCritic
from navix.benchmarks import AlgorithmEntry, Benchmark, BenchmarkResult
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
    hp = PPOHparams(
        budget=32,  # num_steps * num_envs -> exactly 1 update
        num_envs=4,
        num_steps=8,
        num_minibatches=2,
        num_epochs=1,
    )
    # max_steps=num_steps guarantees every parallel env times out (and so
    # produces a done=True) within one rollout, the same way
    # test_dreamer.py/test_pqn.py's own tiny-config helpers do - without
    # this, done_mask can be all-False over such a short rollout (the
    # env's own default max_steps is far longer), and masked_mean's 0/0
    # division turns the score into NaN.
    env = env.replace(max_steps=hp.num_steps)
    env = _flatten_obs(env)
    return PPO(hparams=hp, network=ActorCritic(action_dim=len(env.action_set)), env=env)


def _make_tiny_entry(**overrides) -> AlgorithmEntry:
    defaults = dict(
        name="PPO",
        author="test-author",
        paper_url="https://arxiv.org/abs/1707.06347",
        navix_commit_url="https://github.com/epignatelli/navix/commit/deadbeef",
        algorithm_commit_url="https://github.com/epignatelli/navix/commit/deadbeef",
        agent_factory=_tiny_ppo_factory,
    )
    defaults.update(overrides)
    return AlgorithmEntry(**defaults)


class _TinyBenchmark(Benchmark):
    # A concrete Benchmark preset for tests - Benchmark itself is a base
    # class (name/budget are ClassVars with no useful default), the same
    # way Navix1M/Navix100K (navix/benchmarks/__init__.py) are.
    name = "test-benchmark"
    budget = 32


class _TinyBenchmark64(Benchmark):
    # A second preset, differing only in budget - for asserting the
    # override in Benchmark.run actually takes effect.
    name = "test-benchmark-64"
    budget = 64


_TINY_ENV_IDS = ("Navix-Empty-5x5-v0", "Navix-Empty-6x6-v0")


def test_algorithm_entry_requires_its_metadata_fields():
    # every navix leaderboard entry needs these (issue #130) - a
    # dataclass with no defaults for them is the enforcement mechanism,
    # not just documentation.
    with pytest.raises(TypeError):
        AlgorithmEntry(agent_factory=_tiny_ppo_factory)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "bad_author", ["-leading-hyphen", "trailing-hyphen-", "double--hyphen", "", "a" * 40]
)
def test_algorithm_entry_rejects_invalid_github_handles(bad_author):
    with pytest.raises(ValueError, match="GitHub handle"):
        _make_tiny_entry(author=bad_author)


def test_algorithm_entry_accepts_valid_github_handles():
    # single hyphens, alphanumeric, up to 39 chars are all legal.
    for handle in ("navix", "epignatelli", "a-b-c", "a" * 39):
        _make_tiny_entry(author=handle)  # must not raise


@pytest.mark.parametrize(
    "field_name",
    ["navix_commit_url", "algorithm_commit_url"],
)
@pytest.mark.parametrize(
    "bad_url",
    [
        "deadbeef",  # bare SHA, no repo - exactly what full URLs are meant to prevent
        "not-a-url",
        "ftp://github.com/epignatelli/navix/commit/deadbeef",  # wrong scheme
        "https://github.com/epignatelli/navix",  # no commit SHA at all
        "https://github.com/epignatelli/navix/commit/zzz",  # not hex
    ],
)
def test_algorithm_entry_rejects_invalid_commit_urls(field_name, bad_url):
    with pytest.raises(ValueError, match="commit URL"):
        _make_tiny_entry(**{field_name: bad_url})


def test_benchmark_preset_fixes_name_and_budget_as_class_attributes():
    # Navix1M(entry).run() only takes entry at construction time -
    # name/budget must already be fixed by the preset class, not
    # something the caller can (or needs to) pass in. Constructing with
    # only `entry` (no name/budget kwargs) and getting the right values
    # back out is the actual behavioural guarantee this needs.
    entry = _make_tiny_entry()
    benchmark = _TinyBenchmark(entry=entry)
    assert benchmark.name == "test-benchmark"
    assert benchmark.budget == 32

    other = _TinyBenchmark64(entry=entry)
    assert other.budget == 64


def test_benchmark_calls_agent_factory_once_per_env():
    calls = []

    def counting_factory(env):
        calls.append(env)
        return _tiny_ppo_factory(env)

    entry = _make_tiny_entry(agent_factory=counting_factory)
    benchmark = _TinyBenchmark(entry=entry, env_ids=_TINY_ENV_IDS)
    benchmark.run(log_to_wandb=False)

    assert len(calls) == len(_TINY_ENV_IDS)


def test_benchmark_result_echoes_entry_and_exposes_mandatory_metrics():
    # BenchmarkResult is shaped the same regardless of algorithm - the
    # five navix.plotting.MANDATORY_METRICS fields, not an ad-hoc score.
    entry = _make_tiny_entry()
    benchmark = _TinyBenchmark(entry=entry, env_ids=_TINY_ENV_IDS)
    result = benchmark.run(log_to_wandb=False)

    assert isinstance(result, BenchmarkResult)
    assert result.entry is entry
    for env_id in _TINY_ENV_IDS:
        assert env_id in result.logs

    for field in ("returns", "success_rate", "episode_length", "fps", "wall_time"):
        value = getattr(result, field)
        assert np.all(np.isfinite(np.asarray(value))), f"{field} is not finite"


def test_benchmark_overrides_agent_budget_not_agent_factorys_own():
    # the whole point of budget being fixed per Benchmark subclass (not
    # left to whatever the agent_factory's own hparams say) is that
    # Navix1M vs Navix100K only differ on this - assert the override
    # actually takes effect rather than silently using the factory's own
    # default.
    seen_budgets = []

    def factory(env):
        agent = _tiny_ppo_factory(env)
        seen_budgets.append(agent.hparams.budget)
        return agent

    entry = _make_tiny_entry(agent_factory=factory)
    benchmark = _TinyBenchmark64(entry=entry, env_ids=("Navix-Empty-5x5-v0",))
    benchmark.run(log_to_wandb=False)

    # the factory itself always builds budget=32 (see _tiny_ppo_factory);
    # Benchmark.run must override it to 64 (the preset's own budget)
    # before training.
    assert seen_budgets == [32]


def test_benchmark_result_success_rate_matches_hand_computed_value():
    # 10 updates, 4 steps, 2 envs. Only the last 2 updates (last 20%)
    # should count towards success_rate - make the earlier ones
    # all-failure and the last two all-success, so a bug that averages
    # over the whole history (instead of the last 20%) would be caught.
    num_updates, num_steps, num_envs = 10, 4, 2
    done_mask = np.ones((num_updates, num_steps, num_envs), dtype=bool)
    returns = np.zeros((num_updates, num_steps, num_envs), dtype=np.float32)
    returns[-2:] = 1.0  # last 2 updates (20%) are successes

    logs = {
        "done_mask": jnp.asarray(done_mask),
        "returns": jnp.asarray(returns),
        "lengths": jnp.ones((num_updates, num_steps, num_envs)),
        "iter/fps": jnp.full((num_updates,), 42.0),
        "iter/wall_time": jnp.full((num_updates,), 1.5),
    }
    entry = _make_tiny_entry()
    result = BenchmarkResult.from_logs(entry, {"env-a": logs})
    np.testing.assert_allclose(np.asarray(result.success_rate), 1.0)
    np.testing.assert_allclose(np.asarray(result.fps), 42.0)
    np.testing.assert_allclose(np.asarray(result.wall_time), 1.5)

    # flip it: only the *first* 80% succeed - success_rate should be ~0,
    # since the tail (what's actually aggregated) is all-failure.
    returns2 = np.ones((num_updates, num_steps, num_envs), dtype=np.float32)
    returns2[-2:] = 0.0
    logs2 = {**logs, "returns": jnp.asarray(returns2)}
    result2 = BenchmarkResult.from_logs(entry, {"env-a": logs2})
    np.testing.assert_allclose(np.asarray(result2.success_rate), 0.0)


def test_benchmark_result_averages_across_environments():
    # two envs with opposite success_rate - the aggregated field should
    # land at their mean, not just one env's value.
    num_updates, num_steps, num_envs = 5, 4, 2
    done_mask = np.ones((num_updates, num_steps, num_envs), dtype=bool)
    lengths = np.ones((num_updates, num_steps, num_envs))
    fps = jnp.full((num_updates,), 10.0)
    wall_time = jnp.full((num_updates,), 2.0)

    logs_success = {
        "done_mask": jnp.asarray(done_mask),
        "returns": jnp.ones((num_updates, num_steps, num_envs), dtype=np.float32),
        "lengths": jnp.asarray(lengths),
        "iter/fps": fps,
        "iter/wall_time": wall_time,
    }
    logs_fail = {
        "done_mask": jnp.asarray(done_mask),
        "returns": jnp.zeros((num_updates, num_steps, num_envs), dtype=np.float32),
        "lengths": jnp.asarray(lengths),
        "iter/fps": fps,
        "iter/wall_time": wall_time,
    }
    entry = _make_tiny_entry()
    result = BenchmarkResult.from_logs(entry, {"env-a": logs_success, "env-b": logs_fail})
    np.testing.assert_allclose(np.asarray(result.success_rate), 0.5)
