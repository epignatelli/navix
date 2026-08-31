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

import importlib.util
import json
from typing import Tuple

import numpy as np
import jax
import jax.numpy as jnp
import pytest
from flax import struct

from navix.agents import PPO, PPOHparams, ActorCritic
from navix.agents.agent import masked_mean
from navix.benchmarks import AlgorithmEntry, Benchmark, BenchmarkResult, CostAnalysis, FromScratchBenchmark, TrainingCurve
from navix.benchmarks.scratch import DEFAULT_ENV_IDS
from navix.environments.environment import Environment
from navix.environments.registry import make, registry


def _flatten_obs(env: Environment) -> Environment:
    flatten_obs_fn = lambda x: jnp.ravel(env.observation_fn(x))
    flatten_shape = (int(np.prod(env.observation_space.shape)),)
    return env.replace(
        observation_fn=flatten_obs_fn,
        observation_space=env.observation_space.replace(shape=flatten_shape),
    )


class _TinyPPOEntry(AlgorithmEntry):
    """A deliberately tiny PPO entry - just enough for every code path
    (train/cost_analysis/validate_train_contract) to actually execute,
    matching the pattern used in test_pqn.py/test_dreamer.py."""

    def train(self, env_id: str, budget: int, rng: jax.Array) -> TrainingCurve:
        env = _flatten_obs(make(env_id))
        hp = PPOHparams(
            budget=budget,  # num_steps * num_envs -> exactly 1 update at budget=32
            num_envs=4,
            num_steps=8,
            num_minibatches=2,
            num_epochs=1,
        )
        # max_steps=num_steps guarantees every parallel env times out (and so
        # produces a done=True) within one rollout, the same way
        # test_dreamer.py/test_pqn.py's own tiny-config helpers do - without
        # this, done_mask can be all-False over such a short rollout, and
        # masked_mean's 0/0 division turns the curve into NaN.
        env = env.replace(max_steps=hp.num_steps)
        agent = PPO(hparams=hp, network=ActorCritic(action_dim=len(env.action_set)), env=env)
        _, logs = agent.train(rng)
        mask = jnp.asarray(logs["agent/train/done_mask"], dtype=jnp.bool_)
        return TrainingCurve(
            episodic_returns=masked_mean(logs["agent/train/returns"], mask, axis=(-2, -1)),
            lengths=masked_mean(logs["agent/train/lengths"], mask, axis=(-2, -1)),
        )


class _BadTrainingCurveEntry(AlgorithmEntry):
    """train() returns something that isn't a TrainingCurve at all."""

    def train(self, env_id: str, budget: int, rng: jax.Array):
        return {"episodic_returns": jnp.zeros((4,)), "lengths": jnp.zeros((4,))}


class _WrongRankEntry(AlgorithmEntry):
    """train() returns a TrainingCurve, but its fields are rank 2, not the
    required rank 1 (one un-vmapped call must produce one point per
    update, not a batch of curves)."""

    def train(self, env_id: str, budget: int, rng: jax.Array) -> TrainingCurve:
        return TrainingCurve(
            episodic_returns=jnp.zeros((2, 4)),
            lengths=jnp.zeros((2, 4)),
        )


_TINY_KWARGS = dict(
    name="PPO",
    author="test-author",
    paper_url="https://arxiv.org/abs/1707.06347",
    navix_commit_url="https://github.com/epignatelli/navix/commit/deadbeef",
    algorithm_commit_url="https://github.com/epignatelli/navix/commit/deadbeef",
)


def _make_tiny_entry(**overrides) -> _TinyPPOEntry:
    kwargs = {**_TINY_KWARGS, **overrides}
    return _TinyPPOEntry(**kwargs)


_TINY_ENV_IDS: Tuple[str, ...] = ("Navix-Empty-5x5-v0", "Navix-Empty-6x6-v0")


class _TinyBenchmark(FromScratchBenchmark):
    budget: int = struct.field(pytree_node=False, default=32)
    env_ids: Tuple[str, ...] = struct.field(pytree_node=False, default_factory=lambda: _TINY_ENV_IDS)


# ----------------------------------------------------------------------
# TrainingCurve
# ----------------------------------------------------------------------


def test_training_curve_last_percent_mean_matches_hand_computed_value():
    # 10 updates - only the last 2 (last 20%) should count. Earlier
    # points are all 0, the last two are all 1, so a bug that averages
    # over the whole history (instead of just the tail) would be caught.
    values = jnp.concatenate([jnp.zeros((8,)), jnp.ones((2,))])
    curve = TrainingCurve(episodic_returns=values, lengths=values, diagnostics={"loss": values})

    reduced = curve.last_percent_mean()

    np.testing.assert_allclose(np.asarray(reduced.episodic_returns), 1.0)
    np.testing.assert_allclose(np.asarray(reduced.lengths), 1.0)
    # diagnostics are reduced exactly the same way as the mandatory
    # fields - no special-casing.
    np.testing.assert_allclose(np.asarray(reduced.diagnostics["loss"]), 1.0)


def test_training_curve_last_percent_variance_matches_hand_computed_value():
    # last 20% (2 points) alternate 0/1 -> variance 0.25; the other 80%
    # is constant, so a bug averaging over the whole history would give
    # a much smaller (near-zero) variance instead.
    values = jnp.concatenate([jnp.full((8,), 5.0), jnp.array([0.0, 1.0])])
    curve = TrainingCurve(episodic_returns=values, lengths=values)

    reduced = curve.last_percent_variance()

    np.testing.assert_allclose(np.asarray(reduced.episodic_returns), 0.25)
    np.testing.assert_allclose(np.asarray(reduced.lengths), 0.25)


def test_training_curve_convergence_rate_is_one_for_a_flat_curve():
    # a curve that's already at its own asymptote for its entire length
    # has convergence_rate == 1 by construction (overall mean == tail
    # mean).
    values = jnp.full((10,), 3.0)
    curve = TrainingCurve(episodic_returns=values, lengths=values)

    reduced = curve.convergence_rate()

    np.testing.assert_allclose(np.asarray(reduced.episodic_returns), 1.0)
    np.testing.assert_allclose(np.asarray(reduced.lengths), 1.0)


def test_training_curve_diagnostics_defaults_empty():
    curve = TrainingCurve(episodic_returns=jnp.zeros((4,)), lengths=jnp.zeros((4,)))
    assert curve.diagnostics == {}


# ----------------------------------------------------------------------
# AlgorithmEntry construction / validation
# ----------------------------------------------------------------------


def test_algorithm_entry_requires_its_metadata_fields():
    with pytest.raises(TypeError):
        _TinyPPOEntry(name="PPO")  # type: ignore[call-arg]


@pytest.mark.parametrize("bad_author", ["-leading-hyphen", "trailing-hyphen-", "double--hyphen", "", "a" * 40])
def test_algorithm_entry_rejects_invalid_github_handles(bad_author):
    with pytest.raises(ValueError, match="GitHub handle"):
        _make_tiny_entry(author=bad_author)


def test_algorithm_entry_accepts_valid_github_handles():
    for handle in ("navix", "epignatelli", "a-b-c", "a" * 39):
        _make_tiny_entry(author=handle)  # must not raise


@pytest.mark.parametrize("field_name", ["navix_commit_url", "algorithm_commit_url"])
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


def test_algorithm_entry_populates_hardware_fields():
    entry = _make_tiny_entry()
    assert entry.cpu_type
    assert entry.jax_version
    assert entry.jaxlib_version


def test_algorithm_entry_rejects_train_not_returning_training_curve():
    with pytest.raises(TypeError, match="TrainingCurve"):
        _BadTrainingCurveEntry(**_TINY_KWARGS)


def test_algorithm_entry_rejects_wrong_rank_training_curve():
    with pytest.raises(AssertionError):
        _WrongRankEntry(**_TINY_KWARGS)


def test_algorithm_entry_cost_analysis_returns_finite_values():
    entry = _make_tiny_entry()
    cost = entry.cost_analysis("Navix-Empty-5x5-v0", budget=32)
    assert np.isfinite(cost.flops)
    assert np.isfinite(cost.memory_bytes)
    assert np.isfinite(cost.compile_time_seconds)


# ----------------------------------------------------------------------
# Benchmark.run_env
# ----------------------------------------------------------------------


def test_benchmark_requires_more_than_one_seed():
    with pytest.raises(ValueError, match="more than one seed"):
        _TinyBenchmark(seeds=(0,))


def test_run_env_shapes():
    entry = _make_tiny_entry()
    benchmark = _TinyBenchmark(seeds=(0, 1))

    result = benchmark.run_env(entry, "Navix-Empty-5x5-v0", budget=32)

    assert isinstance(result, BenchmarkResult)
    # one curve per seed, one point per update (budget=32, num_envs=4,
    # num_steps=8 -> exactly 1 update).
    assert result.curve.episodic_returns.shape == (2, 1)
    assert result.curve.lengths.shape == (2, 1)
    # wall_time/fps/cost are single scalars for the whole vmapped call -
    # seeds train together, so there's no per-seed breakdown.
    assert result.wall_time.shape == ()
    assert result.fps.shape == ()
    assert np.isfinite(np.asarray(result.wall_time))
    assert np.isfinite(np.asarray(result.fps))


# ----------------------------------------------------------------------
# FromScratchBenchmark
# ----------------------------------------------------------------------


def test_from_scratch_benchmark_env_ids_defaults_to_curated_set():
    assert FromScratchBenchmark().env_ids == DEFAULT_ENV_IDS


def test_default_env_ids_are_all_registered():
    # a typo here (e.g. an unregistered env id) would silently break
    # every default Navix1M()/Navix100K() run.
    known = set(registry().keys())
    for env_id in DEFAULT_ENV_IDS:
        assert env_id in known, f"{env_id!r} is not a registered environment"


def test_from_scratch_benchmark_run_stacks_one_result_per_env():
    entry = _make_tiny_entry()
    benchmark = _TinyBenchmark(seeds=(0, 1))

    raw = benchmark.run(entry)

    # leading env axis, then seed, then update.
    assert raw.curve.episodic_returns.shape == (len(_TINY_ENV_IDS), 2, 1)
    assert raw.wall_time.shape == (len(_TINY_ENV_IDS),)


def test_from_scratch_benchmark_summary_excludes_non_numeric_and_length():
    entry = _make_tiny_entry()
    benchmark = _TinyBenchmark(seeds=(0, 1))
    raw = benchmark.run(entry)

    summary = benchmark.summary(raw)
    details = benchmark.details(raw)

    assert "env_ids" not in summary
    assert "benchmark/episode/length" not in summary
    assert set(details.keys()) - set(summary.keys()) == {"env_ids", "benchmark/episode/length"}
    assert details["env_ids"] == _TINY_ENV_IDS
    for key, value in summary.items():
        # benchmark/episode/convergence_rate excluded: it's overall/target
        # (see TrainingCurve.convergence_rate), so a still-near-untrained
        # tiny entry (budget=32) can legitimately produce 0/0 = NaN for
        # every env/seed here - there's no valid signal to average, so NaN
        # is the correct output, not a bug (see the test below for the
        # actual non-finite-robustness behavior, on hand-built data where
        # only *some* values are degenerate).
        if key != "benchmark/episode/convergence_rate":
            assert np.all(np.isfinite(np.asarray(value))), f"summary[{key!r}] is not finite"
        # summary is each numeric detail column meaned across envs,
        # ignoring non-finite entries (see Benchmark.summary's docstring).
        finite = np.where(np.isfinite(np.asarray(details[key])), np.asarray(details[key]), np.nan)
        np.testing.assert_allclose(np.asarray(value), np.nanmean(finite))


def test_summary_ignores_non_finite_values_details_keeps_them():
    # env 0: a real, varying returns curve -> convergence_rate (overall
    # mean / tail mean) is finite. env 1: returns are exactly zero for
    # every step - the algorithm never solved it - so convergence_rate
    # is 0/0 = NaN for every seed. summary must still report env 0's
    # real value, not let env 1's NaN blank out the whole aggregate;
    # details must keep env 1's NaN visible, since it's real information
    # (this environment was never solved), not something to hide.
    num_seeds, num_updates = 2, 10
    env0_returns = jnp.tile(jnp.linspace(0.1, 1.0, num_updates), (num_seeds, 1))
    env1_returns = jnp.zeros((num_seeds, num_updates))
    returns = jnp.stack([env0_returns, env1_returns])  # (2 envs, 2 seeds, 10 updates)

    curve = TrainingCurve(episodic_returns=returns, lengths=jnp.ones_like(returns))
    cost = CostAnalysis(
        flops=jnp.asarray([1.0, 1.0]),
        memory_bytes=jnp.asarray([1.0, 1.0]),
        compile_time_seconds=jnp.asarray([1.0, 1.0]),
    )
    raw = BenchmarkResult(
        curve=curve, wall_time=jnp.asarray([1.0, 1.0]), fps=jnp.asarray([1.0, 1.0]), cost=cost
    )

    benchmark = _TinyBenchmark(seeds=(0, 1))
    details = benchmark.details(raw)
    summary = benchmark.summary(raw)

    conv_rate = np.asarray(details["benchmark/episode/convergence_rate"])
    assert np.all(np.isnan(conv_rate[1])), "env 1 (all-zero returns) should keep its NaN in details"
    assert np.all(np.isfinite(conv_rate[0])), "env 0 (real curve) should be finite in details"

    assert np.isfinite(np.asarray(summary["benchmark/episode/convergence_rate"])), "summary should not be NaN"
    np.testing.assert_allclose(
        np.asarray(summary["benchmark/episode/convergence_rate"]), np.mean(conv_rate[0]), rtol=1e-5
    )


def test_from_scratch_benchmark_falsy_env_ids_resolves_to_registry(monkeypatch):
    fake_registry = {env_id: None for env_id in _TINY_ENV_IDS}
    monkeypatch.setattr("navix.benchmarks.scratch.registry", lambda: fake_registry)

    benchmark = _TinyBenchmark(env_ids=(), seeds=(0, 1))
    entry = _make_tiny_entry()

    raw = benchmark.run(entry)
    details = benchmark.details(raw)

    assert details["env_ids"] == tuple(fake_registry.keys())


# ----------------------------------------------------------------------
# Benchmark.submit_entry
# ----------------------------------------------------------------------


def test_submit_entry_writes_three_files(tmp_path):
    # submit_entry locates its output directory from its caller's own
    # __file__ (the convention every run.py already follows), so the
    # call must happen from a file that actually lives in tmp_path -
    # calling it directly from this test module would write into the
    # tests/ directory instead.
    caller_path = tmp_path / "caller.py"
    caller_path.write_text("def call(benchmark, entry, raw):\n    benchmark.submit_entry(entry, raw)\n")
    spec = importlib.util.spec_from_file_location("caller", caller_path)
    caller = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(caller)  # type: ignore[union-attr]

    entry = _make_tiny_entry()
    benchmark = _TinyBenchmark(seeds=(0, 1))
    raw = benchmark.run(entry)

    caller.call(benchmark, entry, raw)

    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "details.json").exists()
    assert (tmp_path / "diagnostics.npz").exists()

    summary_payload = json.loads((tmp_path / "summary.json").read_text())
    assert summary_payload["entry"]["name"] == entry.name
    assert "benchmark/episode/returns" in summary_payload["summary"]

    details_payload = json.loads((tmp_path / "details.json").read_text())
    assert details_payload["details"]["env_ids"] == list(_TINY_ENV_IDS)

    npz = np.load(tmp_path / "diagnostics.npz")
    for key in (
        "benchmark/episode/returns",
        "benchmark/episode/length",
        "benchmark/costs/wall_time",
        "benchmark/costs/fps",
        "benchmark/costs/flops",
        "benchmark/costs/memory_bytes",
        "benchmark/costs/compile_time_seconds",
    ):
        assert key in npz.files


# ----------------------------------------------------------------------
# Benchmark.plot_summary / plot_details / plot_diagnostics
# ----------------------------------------------------------------------


def _make_two_env_raw() -> Tuple["_TinyBenchmark", BenchmarkResult]:
    # Same synthetic shape as test_summary_ignores_non_finite_values_
    # details_keeps_them: one env with a real, varying curve, one with
    # an all-zero (never solved) curve - real enough to exercise the
    # non-finite-value paths a hand-picked "everything is 1.0" fixture
    # wouldn't.
    num_seeds, num_updates = 2, 10
    env0_returns = jnp.tile(jnp.linspace(0.1, 1.0, num_updates), (num_seeds, 1))
    env1_returns = jnp.zeros((num_seeds, num_updates))
    returns = jnp.stack([env0_returns, env1_returns])  # (2 envs, 2 seeds, 10 updates)

    curve = TrainingCurve(
        episodic_returns=returns,
        lengths=jnp.ones_like(returns),
        diagnostics={"loss": jnp.abs(returns - 1.0)},
    )
    cost = CostAnalysis(
        flops=jnp.asarray([1.0, 1.0]),
        memory_bytes=jnp.asarray([1.0, 1.0]),
        compile_time_seconds=jnp.asarray([1.0, 1.0]),
    )
    raw = BenchmarkResult(curve=curve, wall_time=jnp.asarray([1.0, 1.0]), fps=jnp.asarray([1.0, 1.0]), cost=cost)
    benchmark = _TinyBenchmark(seeds=(0, 1))
    return benchmark, raw


def test_plot_summary_returns_a_table_figure():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    benchmark, raw = _make_two_env_raw()
    fig = benchmark.plot_summary(raw)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes[0].tables) == 1
    plt.close(fig)


def test_plot_details_has_one_panel_per_numeric_metric_with_env_id_labels():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    benchmark, raw = _make_two_env_raw()
    details = benchmark.details(raw)
    numeric_keys = [k for k, v in details.items() if k != "env_ids"]

    fig = benchmark.plot_details(raw)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == len(numeric_keys)
    # one bar per environment, labelled with the real env_ids
    ax = fig.axes[0]
    assert len(ax.patches) == len(details["env_ids"])
    xticklabels = [t.get_text() for t in ax.get_xticklabels()]
    assert xticklabels == list(details["env_ids"])
    plt.close(fig)


def test_plot_diagnostics_has_one_panel_per_curve_including_custom_diagnostics():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    benchmark, raw = _make_two_env_raw()
    fig = benchmark.plot_diagnostics(raw)
    assert isinstance(fig, plt.Figure)
    # benchmark/episode/returns, benchmark/episode/length, and curve.
    # diagnostics's own "loss" key, unchanged (see curve_diagnostics).
    assert len(fig.axes) == 3
    titles = {ax.get_title() for ax in fig.axes}
    assert titles == {"benchmark/episode/returns", "benchmark/episode/length", "loss"}
    plt.close(fig)
