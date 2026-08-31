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
import rlax

import navix as nx
from navix.agents.agent import Agent
from navix.agents.dreamer import (
    Dreamer,
    DreamerHparams,
    DreamerTrainState,
    WorldModel,
    Actor,
    Critic,
    Replay,
)
from navix.agents.models import (
    unimix_categorical,
    straight_through_sample,
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
        recurrent_size=hparam_overrides.pop("recurrent_size", 8),
        num_latents=hparam_overrides.pop("num_latents", 3),
        num_classes=hparam_overrides.pop("num_classes", 4),
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

    for key in ("agent/train/frames", "agent/train/updates", "agent/train/done_mask", "agent/train/returns", "agent/train/lengths"):
        assert key in logs, f"missing expected log key {key!r}"

    for key, value in logs.items():
        if key in ("agent/train/done_mask",):
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
    # Agent's wandb-logging methods expect (agent/train/done_mask/
    # agent/train/returns/agent/train/lengths -> agent/episode/* via
    # masked_mean), the same contract PPO relies on.
    dreamer = _make_dreamer(budget=8 * 4)
    ts, logs = jax.jit(dreamer.train)(jax.random.PRNGKey(0))
    logs = jax.tree.map(lambda x: x[0] if hasattr(x, "shape") and x.shape else x, logs)

    with patch("navix.agents.agent.wandb.log") as mock_log:
        dreamer.log_to_wandb(dict(logs))

    assert mock_log.call_count == 1
    logged = mock_log.call_args.args[0]
    assert "agent/episode/returns" in logged
    assert "agent/episode/length" in logged
    assert "agent/episode/success_rate" in logged


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


def test_twohot_head_loss_is_minimised_at_the_target():
    # a head whose logits already encode the target exactly should have
    # ~zero loss; a mismatched target should cost strictly more.
    head = TwoHotHead(hidden_size=4, bins=21, low=-5.0, high=5.0)
    target = jnp.asarray([1.3])
    twohot = rlax.transform_to_2hot(
        rlax.signed_logp1(target), head.low, head.high, head.bins
    )  # already (1, bins)
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
    assert "agent/diagnostics/model/dyn_kl" in logs
    assert "agent/diagnostics/model/rep_kl" in logs


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
    ts = DreamerTrainState.create(
        jax.random.PRNGKey(0),
        dreamer.hparams,
        dreamer.env,
        dreamer.world,
        dreamer.actor,
        dreamer.critic,
    )
    ts, experience = dreamer.collect_experience(ts)
    replay = dreamer._write_replay(ts.replay, experience)
    obs_seq, act_seq, _, first_seq, _ = dreamer._sample_batch(
        jax.random.PRNGKey(1), replay
    )
    _, _, feats, _, _, _ = dreamer.world.apply(
        {"params": ts.model.params},
        obs_seq,
        act_seq,
        first_seq,
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


def test_dreamer_actor_entropy_never_reaches_exactly_zero():
    # Regression test for a training-stability bug found while
    # benchmarking whether Dreamer actually learns: with no floor on the
    # actor's action distribution, its entropy could reach exactly 0.0.
    # Once it does, collect_experience() (which samples actions from this same
    # distribution) stops exploring too - real data collection narrows
    # to whatever the collapsed policy repeats, the world model overfits
    # to that narrow trajectory, and there's no path back. Verified
    # empirically: entropy hit exactly 0.0 and success rate permanently
    # flatlined at 0% on Navix-Empty-5x5-v0 within the first ~1500
    # frames, and stayed there even given 5x more training (500k frames)
    # - ruling out "just needs more compute". actor_unimix (mirroring
    # the world model's own unimix_categorical, already used for its
    # latents) puts a structural floor under every action's probability,
    # making that exact failure mode impossible rather than merely less
    # likely - confirmed here directly on the actor's own distribution,
    # not just the world model's.
    dreamer = _make_dreamer(num_actor_updates=8)
    ts, logs = jax.jit(dreamer.train)(jax.random.PRNGKey(0))
    entropy = np.asarray(logs["agent/diagnostics/actor/entropy"])
    assert np.all(entropy > 0), (
        f"expected actor entropy to never reach exactly 0 with "
        f"actor_unimix={dreamer.hparams.actor_unimix} > 0, got a minimum "
        f"of {entropy.min()} across training"
    )


def test_dreamer_replay_buffer_retains_past_rollouts():
    # Regression test for a structural divergence from DreamerV3 found
    # while investigating why learning never consolidated: there was no
    # replay buffer at all - the world model trained only on the rollout
    # just collected, so a rare success transition (sparse reward) was
    # seen by the reward head for exactly one update and then thrown
    # away, and imagination went straight back to predicting zero reward
    # everywhere. DreamerV3 is replay-based by design; assert that past
    # rollouts really accumulate across updates.
    dreamer = _make_dreamer(budget=8 * 4 * 3)  # -> exactly 3 updates
    ts, _ = jax.jit(dreamer.train)(jax.random.PRNGKey(0))
    assert int(ts.replay.size) == 3, (
        f"expected the replay buffer to hold all 3 collected rollouts, "
        f"got size={int(ts.replay.size)}"
    )
    # capacity is never allocated beyond what the budget can fill
    assert ts.replay.obs.shape[0] == 3


def test_dreamer_imagination_weight_and_states_are_action_aligned():
    # Regression test for an off-by-one in imagined-rollout credit
    # assignment: imagine()'s scan emits only *resulting* states, so
    # actions[t] was taken FROM feats_in[t] (start_feats at t=0), not
    # from feats[t] - and the per-step loss weight must be the
    # probability the trajectory is still alive when the action is
    # *taken* (continues of earlier states only). The old code used
    # cumprod(discount*continues)/discount over the outcome states,
    # which multiplied each action's loss by 1 - P(terminal | its own
    # outcome): the action that reaches the goal - the one gradient
    # carrying the sparse reward signal - was scaled toward zero exactly
    # when the term head (correctly!) predicted the goal ends the
    # episode.
    dreamer = _make_dreamer()
    hp = dreamer.hparams
    ts = DreamerTrainState.create(
        jax.random.PRNGKey(0),
        dreamer.hparams,
        dreamer.env,
        dreamer.world,
        dreamer.actor,
        dreamer.critic,
    )
    start_feats = jax.random.normal(
        jax.random.PRNGKey(1), (5, hp.recurrent_size + hp.latents_flat)
    )
    feats_in, feats, rews, continues, actions, weight = dreamer._actor_critic_rollout(
        ts.model.params, ts.actor.params, start_feats, jax.random.PRNGKey(2)
    )
    # the state each action was taken from: the seed at t=0, then the
    # previous step's outcome
    np.testing.assert_allclose(feats_in[:, 0], start_feats, atol=1e-6)
    np.testing.assert_allclose(feats_in[:, 1:], feats[:, :-1], atol=1e-6)
    # actions at the (real) seed states always get full weight...
    np.testing.assert_allclose(weight[:, 0], 1.0, atol=1e-6)
    # ...and weight[t] accumulates only discount/continuation of states
    # *before* the action, never the action's own outcome
    np.testing.assert_allclose(
        weight[:, 1], hp.discount * continues[:, 0], atol=1e-5
    )
    np.testing.assert_allclose(
        weight[:, 2], hp.discount**2 * continues[:, 0] * continues[:, 1], atol=1e-5
    )


def test_dreamer_untrained_heads_predict_zero_and_near_uniform_policy():
    # Regression test for a training-collapse mechanism found by
    # benchmarking: TwoHotHead's bins span symlog +-20, so a *randomly*
    # initialized reward/value head emits symexp values of garbage
    # magnitude (up to +-e^20) in the very first imagination rollouts -
    # advantages built from those slammed the actor into a deterministic
    # policy (entropy pinned at the unimix floor) within ~64 gradient
    # steps, before the world model had learned anything real, and
    # exploration never recovered. The official implementation
    # zero-initializes these heads' output layers (`outscale: 0.0`) so an
    # untrained head predicts exactly 0; the actor's output layer is
    # near-zero-initialized so the initial policy is near-uniform.
    dreamer = _make_dreamer()
    hp = dreamer.hparams
    ts = DreamerTrainState.create(
        jax.random.PRNGKey(0),
        dreamer.hparams,
        dreamer.env,
        dreamer.world,
        dreamer.actor,
        dreamer.critic,
    )
    feat = jax.random.normal(
        jax.random.PRNGKey(1), (7, hp.recurrent_size + hp.latents_flat)
    )
    head = TwoHotHead(hp.hidden_size, hp.bins, hp.bins_low, hp.bins_high)
    vals = head.mean(dreamer.critic.apply({"params": ts.critic.params}, feat))
    np.testing.assert_allclose(np.asarray(vals), 0.0, atol=1e-5)
    probs = jax.nn.softmax(
        dreamer.actor.apply({"params": ts.actor.params}, feat), axis=-1
    )
    act_dim = probs.shape[-1]
    np.testing.assert_allclose(np.asarray(probs), 1.0 / act_dim, atol=0.02)


def test_dreamer_sample_batch_shifts_done_into_first_flags():
    # Regression test for an off-by-one in episode-boundary masking:
    # navix DEFERS autoreset to the next env.step() call, so done[t] == 1
    # means obs[t+1] is the genuine terminal observation (the goal
    # actually reached by act[t], carrying the episode's reward) and it's
    # obs[t+2] that is the exogenous reset. observe()'s is_first mask for
    # scan step t must therefore be done[t-1], not done[t] - masking
    # unshifted zeroed the belief state exactly at the goal-reaching
    # transition, so the reward/term heads learned to associate the
    # sparse reward with a blank-context latent that imagination (always
    # full-context) never produces: imagined rollouts saw no reward at
    # all, and the actor's advantage signal was identically ~0 while the
    # policy stayed uniform (verified over full 100k-frame runs).
    dreamer = _make_dreamer(seq_len=4)
    L, T, N = 4, 6, 1
    done = np.zeros((1, T, N), dtype=bool)
    done[0, 2, 0] = True
    term = np.zeros((1, T, N), dtype=bool)
    term[0, 3, 0] = True
    replay = Replay(
        obs=jnp.zeros((1, T, N, 3)),
        action=jnp.zeros((1, T, N), dtype=jnp.int32),
        reward=jnp.arange(T, dtype=jnp.float32).reshape(1, T, 1),
        done=jnp.asarray(done),
        termination=jnp.asarray(term),
        idx=jnp.asarray(1, dtype=jnp.int32),
        size=jnp.asarray(1, dtype=jnp.int32),
    )
    # T - (seq_len + 1) = 1, so the only valid window start is index 1 -
    # the sample is fully deterministic despite the random key.
    obs_seq, act_seq, rew_seq, first_seq, terminal_seq = dreamer._sample_batch(
        jax.random.PRNGKey(0), replay
    )
    np.testing.assert_allclose(
        np.asarray(first_seq[0]), done[0, 0:L, 0].astype(np.float32)
    )
    np.testing.assert_allclose(
        np.asarray(terminal_seq[0]), term[0, 1 : L + 1, 0].astype(np.float32)
    )
    np.testing.assert_allclose(
        np.asarray(rew_seq[0]), np.arange(1, L + 1, dtype=np.float32)
    )


if __name__ == "__main__":
    test_dreamer_is_an_agent()
    test_dreamer_trains_one_update_without_nans()
    test_dreamer_networks_have_independent_optimizers_and_step_counters()
    test_dreamer_logs_flow_through_agent_log_to_wandb()
    test_unimix_floors_every_class_probability()
    test_straight_through_sample_is_onehot_but_differentiable()
    test_twohot_head_loss_is_minimised_at_the_target()
    test_dreamer_kl_balance_has_two_distinct_terms()
    test_dreamer_slow_critic_tracks_online_critic_via_ema()
    test_dreamer_actor_gradient_is_nonzero()
    test_dreamer_actor_entropy_never_reaches_exactly_zero()
    test_dreamer_replay_buffer_retains_past_rollouts()
    test_dreamer_imagination_weight_and_states_are_action_aligned()
    test_dreamer_untrained_heads_predict_zero_and_near_uniform_policy()
    test_dreamer_sample_batch_shifts_done_into_first_flags()
