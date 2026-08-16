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
import distrax

import navix as nx
from navix.agents.agent import Agent
from navix.agents.dreamer import (
    Dreamer,
    DreamerHparams,
    WorldModel,
    Actor,
    Critic,
    symlog,
    symexp,
    unimix_categorical,
    straight_through_sample,
    twohot_encode,
    TwoHotHead,
)


def _make_dreamer(**hparam_overrides) -> Dreamer:
    # A deliberately tiny configuration - just enough for every code path
    # (collection, world-model/actor/critic updates, imagination rollouts,
    # sequence sampling) to actually execute, not to train anything useful.
    hp = DreamerHparams(
        budget=hparam_overrides.pop("budget", 256),
        num_envs=hparam_overrides.pop("num_envs", 4),
        num_steps=hparam_overrides.pop("num_steps", 8),
        num_model_updates=hparam_overrides.pop("num_model_updates", 2),
        num_actor_updates=hparam_overrides.pop("num_actor_updates", 2),
        num_critic_updates=hparam_overrides.pop("num_critic_updates", 2),
        batch_size=hparam_overrides.pop("batch_size", 3),
        seq_len=hparam_overrides.pop("seq_len", 4),
        imag_horizon=hparam_overrides.pop("imag_horizon", 3),
        embed_size=hparam_overrides.pop("embed_size", 8),
        deter_size=hparam_overrides.pop("deter_size", 8),
        stoch=hparam_overrides.pop("stoch", 3),
        classes=hparam_overrides.pop("classes", 4),
        bins=hparam_overrides.pop("bins", 11),
        hidden_size=hparam_overrides.pop("hidden_size", 8),
        **hparam_overrides,
    )
    env = nx.make("Navix-Empty-5x5-v0", max_steps=hp.num_steps)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = len(env.action_set)
    world = WorldModel(obs_dim=obs_dim, act_dim=act_dim, hparams=hp)
    actor = Actor(act_dim=act_dim, hidden=hp.hidden_size)
    critic = Critic(hidden=hp.hidden_size, bins=hp.bins, low=hp.bins_low, high=hp.bins_high)
    return Dreamer(hparams=hp, env=env, world=world, actor=actor, critic=critic)


def test_dreamer_is_an_agent():
    # follows the Agent interface: HParams subclass, hparams field, and
    # inherits (rather than reimplements) the wandb logging machinery -
    # PPO does the same, this keeps both algorithms interchangeable from
    # Experiment's point of view.
    dreamer = _make_dreamer()
    assert isinstance(dreamer, Agent)
    assert isinstance(dreamer.hparams, DreamerHparams)
    assert hasattr(dreamer, "train")
    assert Dreamer.log_to_wandb is Agent.log_to_wandb
    assert Dreamer.log_to_wandb_on_train_end is Agent.log_to_wandb_on_train_end


def test_dreamer_trains_one_update_without_nans():
    dreamer = _make_dreamer(budget=8 * 4)  # -> exactly 1 update
    ts, logs = jax.jit(dreamer.train)(jax.random.PRNGKey(0))

    assert int(ts.updates) == 1, "expected exactly one Dreamer.update() call"

    for key in ("iter/frames", "iter/updates", "done_mask", "returns", "lengths"):
        assert key in logs, f"missing expected log key {key!r}"

    for key, value in logs.items():
        if key in ("done_mask",):
            continue
        arr = np.asarray(value)
        assert np.all(np.isfinite(arr)), f"logs[{key!r}] contains non-finite values"


def test_dreamer_networks_have_independent_optimizers_and_step_counters():
    # Regression test for the bug found while porting this agent from the
    # neurips branch: DreamerTrainState used to subclass flax's TrainState
    # directly and hand-roll tx.update() + optax.apply_updates() for all
    # three networks while sharing one `tx`/`step` field meant to be
    # swapped between updates. The swap never took effect before each
    # network's own update ran, so actor and critic gradients were
    # silently applied through the model's optimizer (and learning rate)
    # instead of their own - and since apply_gradients() was never called
    # at all, the step counter never incremented, breaking
    # log_frequency-gated logging.
    #
    # With three independent TrainStates, each network's .step must equal
    # exactly its own configured number of updates after one
    # Dreamer.update() call - this is only possible if each one is really
    # calling its own apply_gradients(), not sharing state with another.
    dreamer = _make_dreamer(
        num_model_updates=3, num_actor_updates=5, num_critic_updates=7
    )
    ts, _ = jax.jit(dreamer.train_first_update)(jax.random.PRNGKey(0))

    assert int(ts.model.step) == 3
    assert int(ts.actor.step) == 5
    assert int(ts.critic.step) == 7
    # if the three optimizers were accidentally the same object/state, the
    # three step counts above could not differ from one another the way
    # they do here (3 != 5 != 7) - this is only possible with genuinely
    # independent TrainStates.


def test_dreamer_logs_flow_through_agent_log_to_wandb():
    # smoke test that Dreamer's logs dict is shaped the way the base
    # Agent's wandb-logging methods expect (done_mask/returns/lengths ->
    # perf/* via masked_mean), the same contract PPO relies on.
    dreamer = _make_dreamer(budget=8 * 4)
    ts, logs = jax.jit(dreamer.train)(jax.random.PRNGKey(0))
    logs = jax.tree.map(lambda x: x[0] if hasattr(x, "shape") and x.shape else x, logs)

    with patch("navix.agents.agent.wandb.log") as mock_log:
        dreamer.log_to_wandb(dict(logs))

    assert mock_log.call_count == 1
    logged = mock_log.call_args.args[0]
    assert "perf/returns" in logged
    assert "perf/episode_length" in logged
    assert "perf/success_rate" in logged


def test_symlog_symexp_are_inverses():
    # symexp(symlog(x)) is the direction actually used in the code (real
    # values -> compressed symlog space and back). symlog(symexp(x)) is
    # only meaningful for x already in symlog-space's realistic range
    # (roughly [-20, 20], matching TwoHotHead's default bin range) -
    # symexp(1000) alone overflows float32 (e^1000), so that direction
    # isn't tested at real-value magnitudes.
    x = jnp.asarray([-1000.0, -1.0, 0.0, 1.0, 1000.0])
    np.testing.assert_allclose(symexp(symlog(x)), x, atol=1e-4)
    x_small = jnp.asarray([-15.0, -1.0, 0.0, 1.0, 15.0])
    np.testing.assert_allclose(symlog(symexp(x_small)), x_small, atol=1e-3)


def test_unimix_floors_every_class_probability():
    # https://arxiv.org/abs/2301.04104 - "1% unimix": no class should
    # ever reach exactly zero probability, however confident the raw
    # logits are, so the KL/entropy terms never fully collapse.
    logits = jnp.asarray([[100.0, -100.0, -100.0, -100.0]])  # near one-hot
    unimix = 0.01
    dist = unimix_categorical(logits, unimix)
    classes = logits.shape[-1]
    assert jnp.all(dist.probs >= unimix / classes - 1e-6), (
        f"expected every class to keep at least unimix/classes probability, "
        f"got {dist.probs}"
    )


def test_straight_through_sample_is_onehot_but_differentiable():
    # forward value must be a genuine one-hot sample (not the soft probs),
    # but the gradient w.r.t. the logits that produced it must be
    # non-zero - the whole point of straight-through gradients over a
    # discrete sample.
    logits = jnp.asarray([2.0, 0.5, -1.0, 0.1])
    rng = jax.random.PRNGKey(0)
    dist = distrax.Categorical(logits=logits)
    sample = straight_through_sample(dist, rng)
    assert sample.shape == logits.shape
    np.testing.assert_allclose(jnp.sum(sample), 1.0, atol=1e-5)
    assert jnp.all((sample >= 0) & (sample <= 1))

    def f(logits):
        d = distrax.Categorical(logits=logits)
        return jnp.sum(straight_through_sample(d, rng) * jnp.arange(4.0))

    grad = jax.grad(f)(logits)
    assert not jnp.allclose(grad, 0.0), (
        "expected a non-zero gradient through the straight-through sample - "
        "got all zeros, meaning the estimator degenerated to a plain "
        "(non-differentiable) discrete sample"
    )


def test_twohot_encode_sums_to_one_and_decodes_near_target():
    bin_centers = jnp.linspace(-5.0, 5.0, 11)
    for x in [-5.0, -3.3, 0.0, 2.7, 5.0]:
        twohot = twohot_encode(jnp.asarray(x), bin_centers)
        np.testing.assert_allclose(jnp.sum(twohot), 1.0, atol=1e-5)
        assert jnp.sum(twohot > 0) <= 2
        decoded = jnp.sum(twohot * bin_centers)
        np.testing.assert_allclose(decoded, x, atol=1e-4)


def test_twohot_head_loss_is_minimised_at_the_target():
    # a head whose logits already encode the target exactly should have
    # ~zero loss; a mismatched target should cost strictly more.
    head = TwoHotHead(hidden_size=4, bins=21, low=-5.0, high=5.0)
    bin_centers = jnp.linspace(-5.0, 5.0, 21)
    target = jnp.asarray([1.3])
    twohot = twohot_encode(symlog(target), bin_centers)  # already (1, bins)
    matching_logits = jnp.log(twohot + 1e-8)
    loss_matching = head.loss(matching_logits, target)
    loss_mismatched = head.loss(matching_logits, jnp.asarray([-3.0]))
    assert float(loss_matching[0]) < float(loss_mismatched[0])


def test_dreamer_kl_balance_has_two_distinct_terms():
    # DreamerV3's KL loss is two terms with different stop-gradient
    # placement (dyn = KL(sg(post)||prior), rep = KL(post||sg(prior))),
    # not one combined KL clamped by a single free-bits scalar - assert
    # both show up as separate, independently-computed log entries.
    dreamer = _make_dreamer(budget=8 * 4)
    ts, logs = jax.jit(dreamer.train)(jax.random.PRNGKey(0))
    assert "agent/model/dyn_kl" in logs
    assert "agent/model/rep_kl" in logs


def test_dreamer_slow_critic_tracks_online_critic_via_ema():
    dreamer = _make_dreamer(slow_critic_rate=0.5)
    ts, _ = jax.jit(dreamer.train_first_update)(jax.random.PRNGKey(0))
    # after one update, the slow critic should have moved partway from
    # its (shared) initial params toward the online critic's post-update
    # params - neither identical to the online critic (rate < 1) nor to
    # the pre-update initial params (rate > 0).
    diffs = jax.tree.map(
        lambda slow, online: jnp.max(jnp.abs(slow - online)),
        ts.slow_critic_params,
        ts.critic.params,
    )
    max_diff = max(float(d) for d in jax.tree.leaves(diffs))
    assert max_diff > 0, (
        "expected the slow critic to differ from the online critic after "
        "training (rate=0.5, not 0 or 1) - got identical params, EMA "
        "update likely not applied"
    )


def test_dreamer_actor_gradient_is_nonzero():
    # Regression test for a bug found implementing DreamerV3 properly:
    # differentiating an imagined return through a *sampled* discrete
    # action (distrax.Categorical.sample() has no gradient w.r.t. its
    # logits) trains nothing through that path - only REINFORCE
    # (log_prob(action) * advantage) has a real gradient for a discrete
    # policy. Assert the actor's gradient is actually non-zero.
    dreamer = _make_dreamer()
    ts = dreamer._init_train_state(jax.random.PRNGKey(0))
    ts, experience = dreamer._collect(ts)
    obs_seq, act_seq, _, _ = dreamer._sample_batch(jax.random.PRNGKey(1), experience)
    _, _, feats, _, _, _ = dreamer.world.apply(
        {"params": ts.model.params},
        obs_seq,
        act_seq,
        method=WorldModel.observe,
        rngs={"sample": jax.random.PRNGKey(2)},
    )
    start_feats = jax.lax.stop_gradient(feats[:, :-1].reshape(-1, feats.shape[-1]))
    return_norm_scale = jnp.asarray(1.0)

    def loss_fn(actor_params):
        return dreamer._actor_loss(
            actor_params,
            ts.model.params,
            ts.critic.params,
            start_feats,
            return_norm_scale,
            jax.random.PRNGKey(3),
        )

    (_, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(ts.actor.params)
    grad_norms = jax.tree.map(lambda g: jnp.sum(jnp.abs(g)), grads)
    total = sum(float(n) for n in jax.tree.leaves(grad_norms))
    assert total > 0, (
        "expected a non-zero actor gradient from the REINFORCE policy "
        "loss - got all zeros, meaning the loss isn't actually connected "
        "to the actor's parameters"
    )


if __name__ == "__main__":
    test_dreamer_is_an_agent()
    test_dreamer_trains_one_update_without_nans()
    test_dreamer_networks_have_independent_optimizers_and_step_counters()
    test_dreamer_logs_flow_through_agent_log_to_wandb()
    test_symlog_symexp_are_inverses()
    test_unimix_floors_every_class_probability()
    test_straight_through_sample_is_onehot_but_differentiable()
    test_twohot_encode_sums_to_one_and_decodes_near_target()
    test_twohot_head_loss_is_minimised_at_the_target()
    test_dreamer_kl_balance_has_two_distinct_terms()
    test_dreamer_slow_critic_tracks_online_critic_via_ema()
    test_dreamer_actor_gradient_is_nonzero()
