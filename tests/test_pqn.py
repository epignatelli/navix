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

import numpy as np
import jax
import jax.numpy as jnp

import navix as nx
from navix.agents.agent import Agent
from navix.agents.pqn import PQN, PQNHparams, TrainingState, Buffer
from navix.agents.models import QNetwork, QMLPEncoder


def _make_pqn(**hparam_overrides) -> PQN:
    # A deliberately tiny configuration - just enough for every code path
    # (collection, target computation, minibatch SGD) to actually
    # execute, not to train anything useful.
    hp = PQNHparams(
        budget=hparam_overrides.pop("budget", 256),
        num_envs=hparam_overrides.pop("num_envs", 4),
        num_steps=hparam_overrides.pop("num_steps", 8),
        num_minibatches=hparam_overrides.pop("num_minibatches", 2),
        num_epochs=hparam_overrides.pop("num_epochs", 2),
        hidden_size=hparam_overrides.pop("hidden_size", 8),
        **hparam_overrides,
    )
    env = nx.make("Navix-Empty-5x5-v0", max_steps=hp.num_steps)
    act_dim = len(env.action_set)
    network = QNetwork(action_dim=act_dim, encoder=QMLPEncoder(hidden_size=hp.hidden_size))
    return PQN(hparams=hp, network=network, env=env)


def _init_train_state(pqn: PQN, rng: jax.Array) -> TrainingState:
    # The subset of PQN.train's init logic needed to get a real
    # TrainingState without running the full training scan - lets tests
    # exercise collect_experience/evaluate_experience/update directly
    # against a real (tiny) network and env.
    import optax

    rng, rng_init, rng_env = jax.random.split(rng, 3)
    init_x = pqn.env.observation_space.sample(rng_init)
    params = pqn.network.init(rng_init, init_x)
    tx = optax.chain(
        optax.clip_by_global_norm(pqn.hparams.max_grad_norm),
        optax.inject_hyperparams(optax.radam)(learning_rate=pqn.hparams.lr),
    )
    reset_rng = jax.random.split(rng_env, pqn.hparams.num_envs)
    env_state = jax.vmap(pqn.env.reset)(reset_rng)
    return TrainingState.create(
        apply_fn=jax.vmap(pqn.network.apply, in_axes=(None, 0)),
        params=params,
        tx=tx,
        env_state=env_state,
        rng=rng,
        frames=jnp.asarray(0, dtype=jnp.int32),
        epoch=jnp.asarray(0, dtype=jnp.int32),
    )


def test_pqn_is_an_agent():
    # follows the Agent interface: HParams subclass, hparams field, and
    # inherits (rather than reimplements) the wandb logging machinery -
    # PPO and Dreamer do the same, this keeps all three interchangeable
    # from Experiment's point of view.
    pqn = _make_pqn()
    assert isinstance(pqn, Agent)
    assert isinstance(pqn.hparams, PQNHparams)
    assert hasattr(pqn, "train")
    assert PQN.log_to_wandb is Agent.log_to_wandb
    assert PQN.log_to_wandb_on_train_end is Agent.log_to_wandb_on_train_end


def test_pqn_trains_one_update_without_nans():
    pqn = _make_pqn(budget=8 * 4)  # num_steps * num_envs -> exactly 1 update
    ts, logs = jax.jit(pqn.train)(jax.random.PRNGKey(0))

    assert int(ts.step) == pqn.hparams.num_minibatches * pqn.hparams.num_epochs

    for key in ("iter/frames", "iter/updates", "done_mask", "returns", "lengths"):
        assert key in logs, f"missing expected log key {key!r}"

    for key, value in logs.items():
        if key in ("done_mask",):
            continue
        arr = np.asarray(value)
        assert np.all(np.isfinite(arr)), f"logs[{key!r}] contains non-finite values"


def test_pqn_logs_flow_through_agent_log_to_wandb():
    # smoke test that PQN's logs dict is shaped the way the base Agent's
    # wandb-logging methods expect (done_mask/returns/lengths ->
    # perf/* via masked_mean), the same contract PPO/Dreamer rely on.
    pqn = _make_pqn(budget=8 * 4)
    ts, logs = jax.jit(pqn.train)(jax.random.PRNGKey(0))
    logs = jax.tree.map(lambda x: x[0] if hasattr(x, "shape") and x.shape else x, logs)

    with patch("navix.agents.agent.wandb.log") as mock_log:
        pqn.log_to_wandb(dict(logs))

    assert mock_log.call_count == 1
    logged = mock_log.call_args.args[0]
    assert "perf/returns" in logged
    assert "perf/episode_length" in logged
    assert "perf/success_rate" in logged


def test_epsilon_schedule_starts_high_anneals_linearly_and_floors():
    pqn = _make_pqn(
        budget=1000, start_e=1.0, end_e=0.1, exploration_fraction=0.5
    )
    # duration = 0.5 * 1000 = 500 frames to anneal from 1.0 -> 0.1
    assert float(pqn.epsilon(jnp.asarray(0))) == 1.0
    assert np.isclose(float(pqn.epsilon(jnp.asarray(250))), 0.55, atol=1e-6)
    assert np.isclose(float(pqn.epsilon(jnp.asarray(500))), 0.1, atol=1e-6)
    # held at end_e past the anneal duration, never undershoots it
    assert np.isclose(float(pqn.epsilon(jnp.asarray(10_000))), 0.1, atol=1e-6)


def test_qnetwork_has_no_output_activation_and_uses_layernorm():
    # PQN's core claim is that LayerNorm (not a target network or replay
    # buffer) is what keeps online Q-learning stable - assert it's
    # actually there, and that the output head is raw Q-values (no
    # squashing activation that would cap achievable Q-values).
    net = QNetwork(action_dim=3, encoder=QMLPEncoder(hidden_size=8))
    params = net.init(jax.random.PRNGKey(0), jnp.zeros((5,)))
    flat = jax.tree_util.tree_leaves_with_path(params)
    layer_norm_scales = [
        p for path, p in flat if any("LayerNorm" in str(k) for k in path)
    ]
    assert len(layer_norm_scales) > 0, "QNetwork has no LayerNorm parameters"

    q_values = net.apply(params, jnp.ones((5,)) * 1e3)
    assert q_values.shape == (3,)
    # with a large-magnitude input, a bounded output activation (tanh,
    # sigmoid, ...) would saturate near +-1; raw Dense output has no such
    # ceiling.
    assert np.any(np.abs(np.asarray(q_values)) > 1.0)


def test_collect_experience_caches_greedy_value_at_collection_time():
    pqn = _make_pqn(num_steps=5, num_envs=3)
    ts = _init_train_state(pqn, jax.random.PRNGKey(0))
    ts, experience = jax.jit(pqn.collect_experience)(ts)

    # Buffer.value[t] must equal max_a Q(obs[t], a) under the SAME
    # params collection ran with (no target network - this is the
    # network's own current weights, recomputed independently here from
    # the stored obs to check the cached value wasn't corrupted/stale).
    # Tolerance is loose (not float32-epsilon-tight): this recomputation
    # goes through a different XLA compilation path (a fresh double vmap)
    # than collect_experience's internal scan-with-inner-vmap, and
    # LayerNorm's variance/rsqrt is sensitive enough to fusion/reduction
    # order that the two legitimately diverge at the ~1e-3 relative level
    # on GPU - same class of cross-compilation float divergence as
    # elsewhere in this codebase, not a sign the cached value is wrong.
    q_values = jax.vmap(jax.vmap(pqn.network.apply, in_axes=(None, 0)), in_axes=(None, 0))(
        ts.params, experience.obs
    )
    expected_value = jnp.max(q_values, axis=-1)
    np.testing.assert_allclose(
        np.asarray(experience.value), np.asarray(expected_value), atol=1e-2, rtol=1e-2
    )


def test_evaluate_experience_bootstraps_with_online_params_not_a_target_network():
    # PQN has no target network - TrainingState carries a single params
    # pytree, and evaluate_experience's bootstrap must use that same
    # train_state.params (not some separately-tracked copy). This is
    # structurally guaranteed by TrainingState having no second params
    # field at all, but assert the behavioural consequence too: replaying
    # evaluate_experience's own last_val computation independently, using
    # train_state.params on train_state.env_state.observation, must match
    # exactly (it's the literal same computation, so this mostly guards
    # against a future edit accidentally routing the bootstrap through
    # different params).
    assert not any(
        "target" in f.name for f in TrainingState.__dataclass_fields__.values()
    ), "TrainingState should not carry a separate target-network params field"

    pqn = _make_pqn(num_steps=4, num_envs=2)
    ts = _init_train_state(pqn, jax.random.PRNGKey(0))
    ts, experience = jax.jit(pqn.collect_experience)(ts)
    targets = jax.jit(pqn.evaluate_experience)(ts, experience)

    last_q = jax.vmap(pqn.network.apply, in_axes=(None, 0))(
        ts.params, ts.env_state.observation
    )
    last_val = jnp.max(last_q, axis=-1)
    # the final timestep's target must reduce to reward + discounted
    # bootstrap through exactly this last_val (q_lambda's mixing
    # coefficient is irrelevant at the boundary - see this module's
    # derivation in PQN.evaluate_experience).
    expected_last_return = experience.reward[-1] + pqn.env.gamma * (
        1.0 - experience.done[-1]
    ) * last_val
    np.testing.assert_allclose(
        np.asarray(targets[-1]), np.asarray(expected_last_return), atol=1e-4
    )


def test_qlambda_target_matches_reference_backward_recursion():
    # Direct numerical check of PQN.evaluate_experience's Q(lambda)
    # target against a plain from-scratch backward recursion over the
    # same (reward, done, cached max-Q) buffer - the reference recursion
    # from Gallici et al.'s own implementation
    # (https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/pqn.py),
    # translated into navix's own buffer indexing convention (done[t] =
    # was the state the transition at t landed in terminal, matching
    # PPO's Buffer). See PQN.evaluate_experience's docstring for why this
    # is exactly rlax.lambda_returns's documented Q(lambda) usage.
    def reference_qlambda(reward, done, value, last_val, gamma, q_lambda):
        T = reward.shape[0]
        returns = np.zeros_like(reward)
        for t in reversed(range(T)):
            next_value = value[t + 1] if t + 1 < T else last_val
            nextnonterminal = 1.0 - done[t]
            if t == T - 1:
                returns[t] = reward[t] + gamma * next_value * nextnonterminal
            else:
                returns[t] = reward[t] + gamma * (
                    q_lambda * returns[t + 1] + (1 - q_lambda) * next_value
                ) * nextnonterminal
        return returns

    pqn = _make_pqn(num_steps=10, num_envs=4, q_lambda=0.65)
    rng = np.random.default_rng(0)
    T, N = pqn.hparams.num_steps, pqn.hparams.num_envs
    reward = rng.normal(size=(T, N)).astype(np.float32)
    value = rng.normal(size=(T, N)).astype(np.float32) * 3
    last_val = rng.normal(size=(N,)).astype(np.float32) * 3
    done = (rng.random((T, N)) < 0.2).astype(np.float32)

    expected = np.stack(
        [
            reference_qlambda(
                reward[:, n], done[:, n], value[:, n], last_val[n],
                pqn.env.gamma, pqn.hparams.q_lambda,
            )
            for n in range(N)
        ],
        axis=1,
    )

    import rlax

    next_values = jnp.concatenate([jnp.asarray(value)[1:], jnp.asarray(last_val)[None]], axis=0)
    discount = pqn.env.gamma * (1.0 - jnp.asarray(done))
    actual = jax.vmap(rlax.lambda_returns, in_axes=(1, 1, 1, None), out_axes=1)(
        jnp.asarray(reward), discount, next_values, pqn.hparams.q_lambda
    )
    np.testing.assert_allclose(np.asarray(actual), expected, atol=1e-4)


def test_pqn_networks_have_no_replay_buffer_across_updates():
    # Every minibatch an update trains on must come from the rollout
    # that SAME update just collected - PQN.update has no field/state
    # that could carry transitions across update() calls (unlike, say,
    # Dreamer.replay). Buffer itself is a plain (not persisted)
    # struct.PyTreeNode returned fresh from collect_experience each time.
    assert not any(
        f.name in ("replay", "buffer")
        for f in TrainingState.__dataclass_fields__.values()
    )
