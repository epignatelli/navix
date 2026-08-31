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

"""A from-scratch DreamerV3 (Hafner et al., "Mastering Diverse Domains
through World Models", https://arxiv.org/abs/2301.04104) agent: an RSSM
world model with categorical latents, trained jointly with an actor and
critic on imagined rollouts. Implements the paper's five headline
robustness techniques, cross-checked directly against the official
implementation (github.com/danijar/dreamerv3, dreamerv3/rssm.py and
embodied/jax/agent.py) rather than assumed from the paper text alone:

  1. Symlog inputs/reconstruction (`rlax.signed_logp1`/`signed_expm1`,
     used by the `Encoder` and `TwoHotHead` in `.models`).
  2. Categorical latents (`num_latents` independent categoricals of
     `num_classes` each, "stoch"/"classes" in the official
     implementation) with straight-through gradients and 1% "unimix" -
     mixing a little uniform mass into every categorical, so no class
     ever gets a literal zero probability - for both the prior and
     posterior.
  3. KL balancing with free bits: two separate KL terms with different
     stop-gradient placement and independent free-nats floors, not one
     combined KL clamped by a single scalar.
  4. Symexp-twohot regression for reward and value (`TwoHotHead`), not
     a Gaussian/MSE head - a discrete classification loss over an
     exponentially-spaced grid of bins, which is far less sensitive to
     reward-scale outliers than a Gaussian likelihood.
  5. Return normalization: an EMA-tracked 5th-95th percentile range of
     returns rescales advantages, so the policy gradient's magnitude
     stays stable across environments with very different reward
     scales, without per-environment tuning.

Also matches the official implementation's actor loss shape: REINFORCE
(log_prob(action) * advantage + entropy bonus), not backpropagation
through sampled discrete actions - discrete distrax.Categorical.sample()
has no gradient path back to its logits, so naively differentiating an
imagined-rollout return through a *sampled* discrete action (as an
earlier draft of this agent did) trains nothing through that path at
all - and an EMA "slow" target critic that the online critic is
regularized toward, for training stability.

Deliberate simplifications, kept for navix's small grid-world
observations rather than image-scale ones: a plain nn.GRUCell for the
deterministic recurrent state (not the official "block GRU", a
parameter-efficiency optimization for much larger deter sizes); a plain
symlog+MSE decoder for observation reconstruction (not the official
implementation's image-specific CNN decoder); ELU activations and no
RMSNorm (not load-bearing for correctness, just a smaller/simpler net);
`num_latents`/`num_classes` default to 8x8 rather than the paper's 32x32, sized
for navix's small grids rather than Atari-scale observations. None of
these are the algorithmic identity of DreamerV3 - the five techniques
above are.

The world model's reusable building blocks (categorical-latent
utilities, the symexp-twohot head, and the RSSM's encoder/decoder/prior/
posterior networks) live in `.models`, alongside PPO's shared network
components; this module holds what's specific to Dreamer itself: the
`WorldModel` that wires those blocks into an RSSM, the actor/critic
heads, and the `Dreamer` agent's collection/replay/training loop.
"""

from __future__ import annotations
from typing import Callable, Dict, Tuple

import numpy as np
import distrax
import jax
import jax.numpy as jnp
from jax import Array
import optax
import rlax
import flax.linen as nn
from flax.training.train_state import TrainState
from flax import struct

from .agent import Agent, HParams
from .models import (
    unimix_categorical,
    straight_through_sample,
    categorical_kl,
    TwoHotHead,
    Encoder,
    Decoder,
    RSSM,
    PriorNet,
    PostNet,
)
from ..environments import Environment
from ..environments.environment import Timestep
from ..states import State


# -------------------------
# Hyperparameters
# -------------------------


class DreamerHparams(HParams):
    # Training schedule
    budget: int = struct.field(pytree_node=False, default=1_000_000)
    """Number of environment frames to train for."""
    num_envs: int = struct.field(pytree_node=False, default=16)
    """Number of parallel environments to run."""
    num_steps: int = struct.field(pytree_node=False, default=128)
    """Number of environment steps to collect per update."""
    num_model_updates: int = struct.field(pytree_node=False, default=128)
    """Number of world-model gradient steps per update. The default is
    deliberately high relative to the frames collected per update (a
    replay ratio in the spirit of the official implementation's
    `train_ratio`): DreamerV3 is designed to be sample-efficient by
    gradient-stepping far more often than it collects. Concretely, with
    a sparse reward the rewarded transitions can be ~0.2% of the replay
    data, and a mean-reduced twohot cross-entropy pushes the reward head
    toward the base-rate prediction (~0) until it has seen enough
    positive examples to separate them - at 32 steps/update the reward
    head was still predicting ~0.004 at real goal transitions after a
    full 100k-frame run (policy stuck at random-walk success), while at
    128 it reaches ~0.98 and the same run ends at 100% success."""
    num_actor_updates: int = struct.field(pytree_node=False, default=128)
    """Number of actor gradient steps per update (see num_model_updates
    for why the default is high)."""
    num_critic_updates: int = struct.field(pytree_node=False, default=128)
    """Number of critic gradient steps per update (see num_model_updates
    for why the default is high)."""
    batch_size: int = struct.field(pytree_node=False, default=64)
    """Number of sequences sampled per world-model gradient step."""
    replay_capacity: int = struct.field(pytree_node=False, default=500_000)
    """Maximum number of environment frames kept in the replay buffer
    (rounded down to whole collection rollouts, and never allocated
    larger than the training budget itself can fill)."""
    seq_len: int = struct.field(pytree_node=False, default=32)
    """Length of the sequences sampled for world-model training."""
    imag_horizon: int = struct.field(pytree_node=False, default=15)
    """Length of the imagined rollouts used to train the actor and critic."""
    discount: float = 0.99
    """Discount factor used in the imagined-rollout lambda-returns."""
    lam: float = 0.95
    """Lambda parameter of the imagined-rollout lambda-returns."""

    # World model: categorical latent + KL balancing (paper defaults:
    # stoch=32, classes=32/64, unimix=0.01, free_nats=1.0, dyn_scale=1.0,
    # rep_scale=0.1 - dims reduced here for navix's small grids).
    num_latents: int = struct.field(pytree_node=False, default=8)
    """Number of independent categorical latent variables ("stoch" in
    the official implementation)."""
    num_classes: int = struct.field(pytree_node=False, default=8)
    """Number of classes per categorical latent variable ("classes" in
    the official implementation)."""
    unimix: float = 0.01
    """Fraction of uniform probability mixed into every categorical."""
    free_nats: float = 1.0
    """Per-(batch,time) KL floor - below this, dyn/rep loss is zero."""
    dyn_scale: float = 1.0
    """Weight of the KL(sg(post)||prior) ("dynamics") term."""
    rep_scale: float = 0.1
    """Weight of the KL(post||sg(prior)) ("representation") term."""

    # Reward/value heads (symexp twohot).
    bins: int = struct.field(pytree_node=False, default=41)
    """Number of bins for the reward/value symexp-twohot heads (paper:
    255; reduced here since navix's rewards/returns span a far smaller
    dynamic range than Atari's)."""
    bins_low: float = -20.0
    """Lower edge of the symlog-space bin range."""
    bins_high: float = 20.0
    """Upper edge of the symlog-space bin range."""

    # Actor: REINFORCE + entropy bonus + return normalization.
    actor_entropy: float = 3e-4
    """Entropy bonus coefficient in the actor's REINFORCE loss."""
    actor_unimix: float = 0.05
    """Fraction of uniform probability mixed into the actor's action
    distribution wherever it's sampled from or scored (data collection,
    imagination, and the REINFORCE loss) - the same `unimix_categorical`
    technique already used for the world model's own categorical latents
    (see `unimix_categorical`'s docstring), applied here too. Without
    this, the actor's entropy can reach exactly zero: once it does,
    `collect_experience` (which samples actions from this same distribution) stops
    exploring too, so real data collection narrows to whatever the
    collapsed policy repeats, the world model overfits to that narrow
    trajectory, and there is no path back - a self-reinforcing collapse
    verified empirically (entropy hits exactly 0.0 and success rate
    permanently flatlines at 0%, independent of how many actor gradient
    steps are taken per update - slowing that down only delays the same
    terminal collapse, it doesn't prevent it). A structural floor on the
    minimum action probability makes that specific failure mode
    impossible rather than merely less likely. Higher than the world
    model's 0.01 default since the action space here is much smaller
    (a handful of actions vs. many latent classes), so the same
    probability mass floor matters proportionally more per action."""
    return_norm_rate: float = 0.01
    """EMA rate for the return-normalization percentile tracker."""
    return_norm_limit: float = 1.0
    """Floor on the return-normalization scale (perc95 - perc5), so a
    near-constant reward signal doesn't blow the advantage up by dividing
    by a near-zero scale."""

    # Slow (EMA target) critic.
    slow_critic_rate: float = 0.02
    """EMA rate the slow critic's params track the online critic at."""
    slow_critic_reg: float = 1.0
    """Weight of the online critic's regularization loss toward the slow
    critic's prediction, on top of its lambda-return regression loss."""

    # Opt/grad
    model_lr: float = 3e-4
    """Learning rate for the world model's optimizer."""
    actor_lr: float = 3e-4
    """Learning rate for the actor's optimizer."""
    critic_lr: float = 3e-4
    """Learning rate for the critic's optimizer."""
    max_grad_norm: float = 100.0
    """Maximum gradient norm for clipping, applied to each of the three
    optimizers independently."""

    # Model sizes
    embed_size: int = 128
    """Size of the encoder's output embedding."""
    recurrent_size: int = 200
    """Size of the RSSM's deterministic (GRU) hidden state ("deter" in
    the official implementation)."""
    hidden_size: int = 200
    """Hidden layer size used throughout the model/actor/critic MLPs."""

    @property
    def latents_flat(self) -> int:
        """Flattened size of the categorical latent (`num_latents *
        num_classes`), i.e. how much of `feat = concat([h, z_flat])` the
        latent occupies."""
        return self.num_latents * self.num_classes


# -------------------------
# World model: RSSM with categorical latents
# -------------------------


class WorldModel(nn.Module):
    obs_dim: int
    act_dim: int
    hparams: DreamerHparams

    def setup(self):
        hp = self.hparams
        self.encoder = Encoder(hp.hidden_size, hp.embed_size)
        self.rssm = RSSM(hp.recurrent_size)
        self.prior = PriorNet(hp.hidden_size, hp.num_latents, hp.num_classes)
        self.post = PostNet(hp.hidden_size, hp.num_latents, hp.num_classes)
        self.decoder = Decoder(hp.hidden_size, self.obs_dim)
        self.reward = TwoHotHead(hp.hidden_size, hp.bins, hp.bins_low, hp.bins_high)
        # Zero-init output like the reward/critic heads (see TwoHotHead):
        # an untrained term head then predicts sigmoid(0) = 0.5 rather
        # than confident random continue/terminate calls.
        self.term = nn.Sequential(
            [
                nn.Dense(hp.hidden_size),
                nn.elu,
                nn.Dense(hp.hidden_size),
                nn.elu,
                nn.Dense(1, kernel_init=nn.initializers.zeros),
            ]
        )

    def init_state(self, batch_size: int) -> Tuple[Array, Array, Array]:
        h = jnp.zeros((batch_size, self.hparams.recurrent_size))
        z_flat = jnp.zeros((batch_size, self.hparams.latents_flat))
        a0 = jnp.zeros((batch_size, self.act_dim))
        return h, z_flat, a0

    def feat(self, h: Array, z_flat: Array) -> Array:
        return jnp.concatenate([h, z_flat], axis=-1)

    def observe(self, obs_seq: Array, act_seq: Array, first_seq: Array) -> Tuple[
        Tuple[Array, Array], Tuple[Array, Array], Array, Array, Array, Array
    ]:
        """Runs the RSSM over an observed sequence, computing the posterior
        latent at every step and decoding obs/reward/term from it.

        Args:
            obs_seq (Array): `f32[B, L+1, obs_dim]`, flattened observations.
            act_seq (Array): `i32[B, L]`, actions taken between consecutive
                observations.
            first_seq (Array): `f32[B, L]`, `1.0` where `obs_seq[:, t + 1]`
                is a fresh post-autoreset observation (the first of a new
                episode) rather than a real consequence of `act_seq[:, t]`.
                NOTE the required shift relative to the buffer's `done`
                flags: navix *defers* autoreset to the next `env.step()`
                call (`Environment.step`'s `should_reset` looks at the
                INPUT timestep), so `done[t] == 1` means `obs[t + 1]` is
                the genuine TERMINAL observation - the goal cell actually
                reached, caused by `act_seq[:, t]`, carrying the episode's
                reward - and it's `obs[t + 2]` that is the exogenous reset.
                The correct mask for scan step `t` is therefore
                `done[t - 1]`, which `_sample_batch` slices as a
                one-step-shifted window over `done`. An earlier version
                passed `done` UNshifted, masking one step early: it zeroed
                the belief state and action exactly at the goal-reaching
                transition, so the reward/term heads were trained to
                associate the sparse reward and termination with a
                *blank-context* posterior of the goal observation - a
                latent that imagination (which always rolls forward with
                full context) never produces, so imagined rollouts never
                saw reward at all, value targets stayed at zero, and the
                actor's advantage signal was identically ~0 despite the
                reward head fitting its (mislabeled) training data well.
                Meanwhile the actual garbage transition - the action
                consumed by the reset, "causing" the teleport to
                `obs[t + 2]` - was trained unmasked. Sampled training
                sequences are sliced from the replay with no regard for
                episode boundaries, so sequences commonly straddle an
                autoreset; at scan step `t` (which pairs `act_seq[:, t]`
                with `embed(obs_seq[:, t + 1])`), both the incoming (h, z)
                carried from step `t - 1` and `act_seq[:, t]` itself are
                zeroed whenever `first_seq[:, t] == 1`, so the reset
                observation's posterior is computed from a blank slate
                instead of a stale, causally-unrelated belief plus an
                action that didn't really produce it - matching both the
                official implementation's `is_first` masking and what
                `collect_experience` does with its own carried latents (full-context
                belief at the terminal observation, blank slate at the
                reset one).

        Returns:
            `((h_seq, z_seq), (dyn_kls, rep_kls), feats, obs_pred, rew_logits,
            term_logits)`, all aligned with `obs_seq[:, 1:]` (`L` steps,
            `t=0..L-1`) except `(h_seq, z_seq)` and `feats`, which keep
            their own `(B, L, ...)` shape. `dyn_kls`/`rep_kls` are
            `f32[B, L]`, computed inside the scan body (not returned as
            distribution objects) because jax.lax.scan only restacks a
            distrax.Distribution's array *leaves* across the new leading
            axis; its batch_shape/event_shape (computed once from the
            un-stacked per-step arrays) go stale, so calling e.g.
            .kl_divergence() *after* the scan raises a shape mismatch deep
            inside distrax/tfp."""
        hp = self.hparams
        B, Lp1, _ = obs_seq.shape
        L = Lp1 - 1

        embed_all = jax.vmap(self.encoder)(obs_seq.reshape(B * (L + 1), -1)).reshape(
            B, L + 1, -1
        )
        # a_oh[:, t] = act_seq[t], the action taken FROM obs_seq[t] TO
        # obs_seq[t+1] - paired directly with embed_all[:, 1:] (embeddings
        # of obs_seq[1:]) below with NO shift. An earlier version prepended
        # a zero action and dropped the last one (`a_prev = concat([zeros,
        # a_oh[:-1]])`), pairing embed(obs_seq[t+1]) with act_seq[t-1]
        # instead of act_seq[t] - i.e. every posterior/prior at scan step
        # t was computed as if the action taken *before* act_seq[t] had
        # produced obs_seq[t+1], one step stale. This directly contradicts
        # both `posterior_step` (used for real-env collection: `h_next =
        # rssm(h, z, a_just_taken)` then posterior on `embed(new_obs)` -
        # the action and the observation it produced, paired together) and
        # `imagine`'s `rollout_step` (`h_next = rssm(h, z, a)` using the
        # action just sampled from the *current* state to reach the next
        # one) - both of which already use the correct, unshifted pairing.
        # Training `observe()` with actions mismatched by one timestep
        # from the observations/rewards/terminations they actually caused
        # silently degrades everything derived from it (obs/reward/term
        # losses, and - critically - the prior network `imagine()` samples
        # from), while `obs_loss`/`rew_loss` can still fit close to zero
        # by the end of training since consecutive actions/observations in
        # a grid-world are highly autocorrelated, masking the bug in the
        # aggregate loss curves alone.
        a_oh = jax.nn.one_hot(act_seq, self.act_dim)

        # Plain jax.lax.scan, not nn.scan: `step` closes over `self`
        # implicitly (via self.rssm/self.prior/self.post) rather than
        # taking it as an explicit first argument, which is what Flax's
        # lifted nn.scan requires when called as `nn.scan(step, ...)(carry,
        # xs)` - passing a plain closure that way makes Flax's transform
        # machinery treat `carry` itself as the (missing) `self` argument,
        # failing deep inside with a confusing `'tuple' object has no
        # attribute '_state'`. Since setup() already built every submodule
        # once, calling them repeatedly inside a plain scan just reuses
        # their closed-over params - no Flax-level parameter broadcasting
        # is needed here. The per-step sample rng is threaded through the
        # scan carry explicitly instead of nn.scan's split_rngs, since
        # plain jax.lax.scan doesn't auto-split Flax's rng streams.
        def step(carry, inputs):
            h_prev, z_prev_flat, rng = carry
            a_t, embed_t, is_first_t = inputs
            rng, key = jax.random.split(rng)
            # is_first_t == 1 means embed_t is a fresh post-autoreset
            # observation - zero the incoming (h, z) *and* the incoming
            # action before this step's RSSM update (matching the official
            # implementation's `is_first` masking, see the docstring above)
            # so the reset observation's posterior comes from a blank
            # slate rather than a stale, unrelated belief paired with an
            # action that didn't cause it.
            keep = (1.0 - is_first_t)[:, None]
            h_prev = h_prev * keep
            z_prev_flat = z_prev_flat * keep
            a_t = a_t * keep
            h_t = self.rssm(h_prev, z_prev_flat, a_t)
            prior_logits = self.prior(h_t)
            post_logits = self.post(h_t, embed_t)
            prior_dist = unimix_categorical(prior_logits, hp.unimix)
            post_dist = unimix_categorical(post_logits, hp.unimix)
            prior_dist_sg = distrax.Categorical(
                probs=jax.lax.stop_gradient(prior_dist.probs)
            )
            post_dist_sg = distrax.Categorical(
                probs=jax.lax.stop_gradient(post_dist.probs)
            )
            dyn_kl_t = categorical_kl(post_dist_sg, prior_dist)
            rep_kl_t = categorical_kl(post_dist, prior_dist_sg)
            z_t = straight_through_sample(post_dist, key)
            z_t_flat = z_t.reshape(*z_t.shape[:-2], -1)
            feat_t = self.feat(h_t, z_t_flat)
            return (h_t, z_t_flat, rng), (h_t, z_t_flat, dyn_kl_t, rep_kl_t, feat_t)

        # jax.lax.scan always scans over axis 0, but our sequences are
        # batch-major (B, L, ...) - swap to (L, B, ...) going in, and swap
        # the stacked outputs back to (B, L, ...) coming out.
        inputs = jax.tree.map(
            lambda x: jnp.swapaxes(x, 0, 1),
            (a_oh, embed_all[:, 1:], first_seq.astype(jnp.float32)),
        )
        init_rng = self.make_rng("sample")
        (h_last, z_last, _), (hs, zs, dyn_kls, rep_kls, feats) = jax.lax.scan(
            step,
            (
                jnp.zeros((B, hp.recurrent_size)),
                jnp.zeros((B, hp.latents_flat)),
                init_rng,
            ),
            inputs,
            length=L,
        )
        hs, zs, dyn_kls, rep_kls, feats = jax.tree.map(
            lambda x: jnp.swapaxes(x, 0, 1), (hs, zs, dyn_kls, rep_kls, feats)
        )

        feats_flat = feats.reshape(B * L, -1)
        obs_pred = self.decoder(feats_flat)
        rew_logits = self.reward(feats_flat)
        term_logits = self.term(feats_flat)

        return (hs, zs), (dyn_kls, rep_kls), feats, obs_pred, rew_logits, term_logits

    def posterior_step(
        self, h: Array, z_flat: Array, a_prev_oh: Array, obs: Array, rng: Array
    ) -> Tuple[Array, Array]:
        """One environment-collection step: advances the RSSM's belief with
        the previous action, then updates it to the posterior given the new
        observation. Encoder/RSSM/posterior are submodules bound to this
        WorldModel instance, so - unlike a bare Dense/Sequential module -
        they can only be called from inside a WorldModel.apply() trace,
        which is what this method (called via `self.world.apply(...,
        method=WorldModel.posterior_step)`) provides for `collect_experience`."""
        h_next = self.rssm(h, z_flat, a_prev_oh)
        embed = self.encoder(obs)
        post_dist = unimix_categorical(self.post(h_next, embed), self.hparams.unimix)
        z_next = straight_through_sample(post_dist, rng)
        z_next_flat = z_next.reshape(*z_next.shape[:-2], -1)
        return h_next, z_next_flat

    def prior_step(
        self, h: Array, z_flat: Array, a_oh: Array, rng: Array
    ) -> Tuple[Array, Array, Array]:
        """One imagined (prior/dynamics) transition with a caller-supplied
        action - the same mechanics as a single `imagine` rollout step,
        but decoupled from the actor. Returns `(feat_next, reward_mean,
        term_prob)`. Used for diagnosis: comparing the reward/term heads'
        predictions on prior-sampled latents at known real transitions
        (e.g. forcing the recorded goal-reaching action from the recorded
        pre-goal posterior state) against their predictions on posterior
        latents isolates whether a learned signal actually survives into
        imagination, where the actor is trained."""
        h_next = self.rssm(h, z_flat, a_oh)
        prior_dist = unimix_categorical(self.prior(h_next), self.hparams.unimix)
        z_next = straight_through_sample(prior_dist, rng)
        z_next_flat = z_next.reshape(*z_next.shape[:-2], -1)
        feat = jnp.concatenate([h_next, z_next_flat], axis=-1)
        rew = self.reward.mean(self.reward(feat))
        term_prob = jax.nn.sigmoid(jnp.squeeze(self.term(feat), -1))
        return feat, rew, term_prob

    def init_probe(self, obs: Array, a_prev_oh: Array) -> None:
        """Touches every submodule exactly once, purely so `WorldModel.
        init()` can discover every parameter's shape. Parameters are keyed
        by submodule path, not by how many times a submodule is called in
        the "real" forward pass, so a single non-scanned pass creates the
        identical parameter tree `observe()` would - deliberately used
        instead of `observe()` for `.init()`, because `.init()` does its
        own internal tracing to discover shapes, which doesn't compose
        safely with tracing through observe()'s internal jax.lax.scan
        while already inside an outer jax.jit (surfaces as a confusing
        `UnexpectedTracerError` pointing at an unrelated submodule)."""
        hp = self.hparams
        h = jnp.zeros((obs.shape[0], hp.recurrent_size))
        z_flat = jnp.zeros((obs.shape[0], hp.latents_flat))
        h_t = self.rssm(h, z_flat, a_prev_oh)
        self.prior(h_t)
        embed = self.encoder(obs)
        post_logits = self.post(h_t, embed)
        z_t = jax.nn.one_hot(jnp.zeros(obs.shape[0], dtype=jnp.int32), hp.num_classes)
        z_t = jnp.broadcast_to(z_t[:, None, :], (obs.shape[0], hp.num_latents, hp.num_classes))
        feat_t = self.feat(h_t, z_t.reshape(obs.shape[0], -1))
        self.decoder(feat_t)
        self.reward(feat_t)
        self.term(feat_t)

    def imagine(
        self,
        start_h: Array,
        start_z_flat: Array,
        actor_logits_fn: Callable[[Array], Array],
        horizon: int,
        rng: Array,
    ) -> Tuple[Array, Array, Array, Array, Array]:
        """Rolls the RSSM forward purely through its own prior (no real
        observations), sampling actions from `actor_logits_fn` at each
        step - the "imagined" trajectories the actor and critic train on.

        Returns `(feats, rewards, terms, actions, action_logps)`, all
        `(B, H, ...)` (batch-major). `rewards` is the reward head's
        decoded mean (plain array, not a TwoHotHead logits tensor);
        `actions`/`action_logps` are returned because the actor loss
        needs `log_prob(action)` under the *current* actor params for its
        REINFORCE term - recomputed at loss-value-and-grad time from the
        stored actions, not reused from here (this rollout's actor calls
        may run in a different tracing context)."""

        def rollout_step(carry, _):
            h, z_flat, rng = carry
            feat = jnp.concatenate([h, z_flat], axis=-1)
            logits = actor_logits_fn(feat)
            rng, key_a, key_z = jax.random.split(rng, 3)
            actor_dist = unimix_categorical(logits, self.hparams.actor_unimix)
            a = actor_dist.sample(seed=key_a)
            logp = actor_dist.log_prob(a)
            a_oh = jax.nn.one_hot(a, logits.shape[-1])
            h_next = self.rssm(h, z_flat, a_oh)
            prior_dist = unimix_categorical(self.prior(h_next), self.hparams.unimix)
            z_next = straight_through_sample(prior_dist, key_z)
            z_next_flat = z_next.reshape(*z_next.shape[:-2], -1)
            feat_next = jnp.concatenate([h_next, z_next_flat], axis=-1)
            rew_mean = self.reward.mean(self.reward(feat_next))
            term_logit = jnp.squeeze(self.term(feat_next), -1)
            return (h_next, z_next_flat, rng), (
                feat_next,
                rew_mean,
                term_logit,
                a,
                logp,
            )

        (hT, zT, _), (feats, rews, term_logits, actions, logps) = jax.lax.scan(
            rollout_step, (start_h, start_z_flat, rng), None, length=horizon
        )
        # (H, B, ...) -> (B, H, ...)
        feats, rews, term_logits, actions, logps = jax.tree.map(
            lambda x: jnp.swapaxes(x, 0, 1), (feats, rews, term_logits, actions, logps)
        )
        return feats, rews, term_logits, actions, logps


# -------------------------
# Actor & Critic
# -------------------------


class Actor(nn.Module):
    act_dim: int
    hidden: int = 200

    @nn.compact
    def __call__(self, feat: Array) -> Array:
        # Near-zero output init (the official implementation's small
        # actor `outscale`): the initial policy is near-uniform, so early
        # data collection genuinely explores instead of committing to
        # whatever the random init happens to prefer.
        net = nn.Sequential(
            [
                nn.Dense(self.hidden),
                nn.elu,
                nn.Dense(self.hidden),
                nn.elu,
                nn.Dense(self.act_dim, kernel_init=nn.initializers.orthogonal(0.01)),
            ]
        )
        return net(feat)

    def dist(self, feat: Array) -> distrax.Categorical:
        return distrax.Categorical(logits=self(feat))


class Critic(nn.Module):
    hidden: int = 200
    bins: int = 255
    low: float = -20.0
    high: float = 20.0

    @nn.compact
    def __call__(self, feat: Array) -> Array:
        return TwoHotHead(self.hidden, self.bins, self.low, self.high)(feat)


# -------------------------
# Buffer (shape-compatible with Agent.log_to_wandb)
# -------------------------


class Buffer(struct.PyTreeNode):
    done: Array
    termination: Array
    action: Array
    reward: Array
    log_prob: Array
    obs: Array
    info: Dict[str, Array]
    t: Array
    state: State


class Replay(struct.PyTreeNode):
    """A fixed-capacity FIFO replay buffer of whole collection rollouts
    ("blocks" of `num_steps` x `num_envs` transitions), carried inside
    `DreamerTrainState` so it lives through the `jax.lax.scan` training
    loop. DreamerV3 is a *replay-based* algorithm: the world model must
    keep training on past experience, not only the rollout just
    collected - with a sparse reward, a rare success transition seen only
    in the update it happened in is forgotten by the reward head one
    update later, so imagination goes back to predicting zero reward
    everywhere and the actor's learning signal vanishes as soon as it
    appears. (An earlier version had no replay at all - `_sample_batch`
    read directly from the latest rollout - which is exactly the failure
    mode that produced isolated windows of success that never
    consolidated.)

    Blocks are written whole (one per `Dreamer.update()` call) at a
    rolling index; sampled sequences never straddle two blocks, so the
    rolling overwrite boundary can't splice unrelated timelines together.
    Only what world-model training needs is stored (obs flattened, action,
    reward, done, termination) - not the logging-only fields of `Buffer`.
    """

    obs: Array  # (C, T, N, obs_dim), original observation dtype
    action: Array  # (C, T, N) i32
    reward: Array  # (C, T, N) f32
    done: Array  # (C, T, N) bool - episode ended (termination OR truncation)
    termination: Array  # (C, T, N) bool - true environment termination only
    idx: Array  # scalar i32, next block slot to (over)write
    size: Array  # scalar i32, number of filled block slots


# -------------------------
# Training state
# -------------------------


class DreamerTrainState(struct.PyTreeNode):
    """Wraps three *independent* `TrainState`s, one per network - the
    world model, actor, and critic each have their own optimizer,
    learning rate and step counter, and are updated via their own
    `apply_gradients()`. An earlier draft instead subclassed `TrainState`
    directly and shared one `tx`/`step` field meant to be swapped between
    the three networks' updates; the swap never actually took effect
    before each network's update ran, so actor and critic gradients were
    silently applied through the model's optimizer (and learning rate)
    instead of their own, and the step counter never incremented since
    `apply_gradients()` was never called. Three separate `TrainState`s
    make both bugs structurally impossible instead of relying on
    remembering to swap a shared field correctly.

    `slow_critic_params` is a plain EMA-tracked copy of the critic's
    params (not a fourth optimized TrainState - nothing ever computes a
    gradient w.r.t. it, it only ever gets updated by exponential
    averaging toward `critic.params`), used both to regularize the online
    critic's training and to compute the return-normalization statistics.
    """

    model: TrainState
    actor: TrainState
    critic: TrainState
    slow_critic_params: dict
    replay: Replay

    env_state: Timestep
    rng: Array
    frames: Array
    updates: Array
    return_norm_lo: Array  # EMA of the 5th percentile of returns
    return_norm_hi: Array  # EMA of the 95th percentile of returns

    # Latents per-env, carried across collection steps.
    h: Array  # (N, recurrent_size)
    z: Array  # (N, latents_flat)
    a_prev_oh: Array  # (N, act_dim)

    @classmethod
    def create(
        cls,
        rng: Array,
        hparams: DreamerHparams,
        env: Environment,
        world: WorldModel,
        actor: Actor,
        critic: Critic,
    ) -> "DreamerTrainState":
        """Builds an initial `DreamerTrainState`: inits the world model/
        actor/critic's params and optimizers, resets `hparams.num_envs`
        environments, and allocates an empty replay buffer sized to
        whichever is smaller of `hparams.replay_capacity` and what
        `hparams.budget` can actually fill. Mirrors the `flax.training.
        train_state.TrainState.create` convention PPO's own `TrainingState`
        relies on (construction logic lives on the state itself, not on
        the agent that produces it)."""
        hp = hparams
        obs_dim = world.obs_dim
        assert obs_dim == int(np.prod(env.observation_space.shape)), (
            "world's obs_dim must match the (flattened) observation space "
            f"of env - got obs_dim={obs_dim}, but env.observation_space."
            f"shape={env.observation_space.shape} flattens to "
            f"{int(np.prod(env.observation_space.shape))}. Construct "
            "WorldModel(obs_dim=int(np.prod(env.observation_space.shape)), "
            "act_dim=len(env.action_set), hparams=...) explicitly, the "
            "same way PPO expects ActorCritic(action_dim=len(env."
            "action_set)) pre-constructed rather than building it internally."
        )

        rng, wk1, wk2, ak, ck = jax.random.split(rng, 5)
        w_variables = world.init(
            {"params": wk1, "sample": wk2},
            jnp.zeros((1, obs_dim)),
            jnp.zeros((1, world.act_dim)),
            method=WorldModel.init_probe,
        )
        a_variables = actor.init(
            ak, jnp.zeros((1, hp.recurrent_size + hp.latents_flat))
        )
        c_variables = critic.init(
            ck, jnp.zeros((1, hp.recurrent_size + hp.latents_flat))
        )

        model_tx = optax.chain(
            optax.clip_by_global_norm(hp.max_grad_norm), optax.adam(hp.model_lr)
        )
        actor_tx = optax.chain(
            optax.clip_by_global_norm(hp.max_grad_norm), optax.adam(hp.actor_lr)
        )
        critic_tx = optax.chain(
            optax.clip_by_global_norm(hp.max_grad_norm), optax.adam(hp.critic_lr)
        )
        model_state = TrainState.create(
            apply_fn=world.apply, params=w_variables["params"], tx=model_tx
        )
        actor_state = TrainState.create(
            apply_fn=actor.apply, params=a_variables["params"], tx=actor_tx
        )
        critic_state = TrainState.create(
            apply_fn=critic.apply, params=c_variables["params"], tx=critic_tx
        )

        rng, rs = jax.random.split(rng)
        reset_rng = jax.random.split(rs, hp.num_envs)
        env_state = jax.vmap(env.reset)(reset_rng)
        h0, z0, a0 = world.apply(w_variables, hp.num_envs, method=WorldModel.init_state)

        # One replay block per update() call (num_steps x num_envs
        # frames); never allocate more blocks than the training budget
        # can actually fill.
        block_frames = hp.num_steps * hp.num_envs
        num_blocks = int(
            max(1, min(hp.replay_capacity // block_frames, hp.budget // block_frames))
        )
        obs_dtype = env_state.observation.dtype
        replay = Replay(
            obs=jnp.zeros(
                (num_blocks, hp.num_steps, hp.num_envs, obs_dim), dtype=obs_dtype
            ),
            action=jnp.zeros((num_blocks, hp.num_steps, hp.num_envs), dtype=jnp.int32),
            reward=jnp.zeros(
                (num_blocks, hp.num_steps, hp.num_envs), dtype=jnp.float32
            ),
            done=jnp.zeros((num_blocks, hp.num_steps, hp.num_envs), dtype=jnp.bool_),
            termination=jnp.zeros(
                (num_blocks, hp.num_steps, hp.num_envs), dtype=jnp.bool_
            ),
            idx=jnp.asarray(0, dtype=jnp.int32),
            size=jnp.asarray(0, dtype=jnp.int32),
        )

        return cls(
            model=model_state,
            actor=actor_state,
            critic=critic_state,
            slow_critic_params=c_variables["params"],
            replay=replay,
            env_state=env_state,
            rng=rng,
            frames=jnp.asarray(0, dtype=jnp.int32),
            updates=jnp.asarray(0, dtype=jnp.int32),
            return_norm_lo=jnp.asarray(0.0, dtype=jnp.float32),
            return_norm_hi=jnp.asarray(0.0, dtype=jnp.float32),
            h=h0,
            z=z0,
            a_prev_oh=a0,
        )


# -------------------------
# Agent
# -------------------------


class Dreamer(Agent):
    hparams: DreamerHparams
    world: WorldModel = struct.field(pytree_node=False)
    actor: Actor = struct.field(pytree_node=False)
    critic: Critic = struct.field(pytree_node=False)

    # ---------- Utilities ----------

    @staticmethod
    def _flatten_obs(obs: Array) -> Array:
        """Flattens every non-batch dimension of `obs` into one vector, so
        the Dense-based world model can consume any navix observation
        shape (`categorical`'s `(H, W)`, `rgb`'s `(H, W, 3)`, `symbolic`'s
        `(H, W, 3)`, ...) without the caller needing to pre-wrap the env,
        the way `baselines/ppo.py`'s `FlattenObsWrapper` does for `PPO`."""
        return obs.reshape(obs.shape[0], -1)

    @staticmethod
    def _lambda_returns(
        next_values: Array, rewards: Array, continues: Array, discount: float, lam: float
    ) -> Array:
        """shapes: (B, H), (B, H), (B, H) -> (B, H). Thin wrapper around
        `rlax.lambda_returns`, vmapped over the batch axis (that function
        itself scans over axis 0, which here is time): same recursion
        `Gₜ = rₜ + γₜ·((1−λ)vₜ + λGₜ₊₁)`, same boundary init `v[-1]`. Mirrors
        how ppo.py wraps `rlax.truncated_generalized_advantage_estimation`."""
        return jax.vmap(rlax.lambda_returns, in_axes=(0, 0, 0, None))(
            rewards, discount * continues, next_values, lam
        )

    # ---------- Collection ----------

    def collect_experience(
        self, ts: DreamerTrainState
    ) -> Tuple[DreamerTrainState, Buffer]:
        """Runs `hparams.num_steps` steps in `hparams.num_envs` parallel
        envs, carrying the per-env posterior latent (h, z, a_prev_oh)
        across steps so the policy always acts on an up-to-date belief."""
        hp = self.hparams

        def _env_step(carry, _):
            env_state, rng, h, z, a_prev_oh = carry

            feat = jnp.concatenate([h, z], axis=-1)
            logits = self.actor.apply({"params": ts.actor.params}, feat)
            rng, key_a = jax.random.split(rng)
            actor_dist = unimix_categorical(logits, hp.actor_unimix)
            a = actor_dist.sample(seed=key_a)
            log_prob = actor_dist.log_prob(a)

            new_env_state = jax.vmap(self.env.step, in_axes=(0, 0))(env_state, a)

            a_oh = jax.nn.one_hot(a, logits.shape[-1])
            flat_obs = self._flatten_obs(new_env_state.observation)
            rng, key_z = jax.random.split(rng)
            keys_z = jax.random.split(key_z, hp.num_envs)
            h_next, z_next = jax.vmap(
                lambda h_, z_, a_, o_, k_: self.world.apply(
                    {"params": ts.model.params},
                    h_,
                    z_,
                    a_,
                    o_,
                    k_,
                    method=WorldModel.posterior_step,
                )
            )(h, z, a_oh, flat_obs, keys_z)

            # Reset latents where the env just autoreset, so a new episode
            # doesn't start by conditioning on the previous one's belief.
            done = new_env_state.is_done()
            h_next = jnp.where(done[:, None], jnp.zeros_like(h_next), h_next)
            z_next = jnp.where(done[:, None], jnp.zeros_like(z_next), z_next)
            a_oh_next = jnp.where(done[:, None], jnp.zeros_like(a_oh), a_oh)

            transition = Buffer(
                done=new_env_state.is_done(),
                termination=new_env_state.is_termination(),
                action=a,
                reward=new_env_state.reward,
                log_prob=log_prob,
                obs=env_state.observation,
                info=new_env_state.info,
                t=env_state.t,
                state=env_state.state,
            )
            return (new_env_state, rng, h_next, z_next, a_oh_next), transition

        (env_state, rng, h, z, a_prev_oh), traj = jax.lax.scan(
            _env_step,
            (ts.env_state, ts.rng, ts.h, ts.z, ts.a_prev_oh),
            None,
            length=hp.num_steps,
        )
        ts = ts.replace(
            env_state=env_state,
            rng=rng,
            h=h,
            z=z,
            a_prev_oh=a_prev_oh,
            frames=ts.frames + hp.num_steps * hp.num_envs,
        )
        return ts, traj

    @staticmethod
    def _write_replay(replay: Replay, experience: Buffer) -> Replay:
        """Appends one whole collection rollout to the replay buffer as a
        block at the rolling write index (overwriting the oldest block
        once the buffer is full)."""
        obs = experience.obs.reshape(*experience.obs.shape[:2], -1)

        def put(buf, new):
            return jax.lax.dynamic_update_index_in_dim(
                buf, new.astype(buf.dtype), replay.idx, 0
            )

        capacity = replay.obs.shape[0]
        return replay.replace(
            obs=put(replay.obs, obs),
            action=put(replay.action, experience.action),
            reward=put(replay.reward, experience.reward),
            done=put(replay.done, experience.done),
            termination=put(replay.termination, experience.termination),
            idx=(replay.idx + 1) % capacity,
            size=jnp.minimum(replay.size + 1, capacity),
        )

    def _sample_batch(
        self, rng: Array, replay: Replay
    ) -> Tuple[Array, Array, Array, Array, Array]:
        """Samples `hparams.batch_size` sequences of `hparams.seq_len + 1`
        observations / `seq_len` actions uniformly from the *filled* part
        of the replay buffer (all blocks written so far, not just the
        latest rollout), for world-model training. Returns
        `(obs_seq, act_seq, rew_seq, first_seq, terminal_seq)`:
        `first_seq[t] = done[start + t - 1]` - the one-step-SHIFTED window
        over `done` that `observe()`'s boundary masking needs, because
        navix defers autoreset by one step: `done[t] == 1` marks
        `obs[t + 1]` as the genuine terminal observation, and it's
        `obs[t + 2]` that is the fresh reset (see `observe`'s docstring).
        Window starts are drawn from `[1, T - seq_len - 1]` (never 0) so
        the shifted window's first element always exists in-block.
        `terminal_seq` marks true environment termination only (the term
        head's target - a timeout truncation is NOT a terminal state, the
        episode was cut short exogenously, see `_model_loss`)."""
        hp = self.hparams
        C, T, N = replay.action.shape
        L = hp.seq_len
        max_start = jnp.maximum(T - (L + 1), 1)

        rng, key1, key2, key3 = jax.random.split(rng, 4)
        block_idx = jax.random.randint(
            key1, (hp.batch_size,), minval=0, maxval=jnp.maximum(replay.size, 1)
        )
        env_idx = jax.random.randint(key2, (hp.batch_size,), minval=0, maxval=N)
        start_idx = jax.random.randint(
            key3, (hp.batch_size,), minval=1, maxval=max_start + 1
        )

        def take_seq(bi, ei, si):
            obs_seq = jax.lax.dynamic_slice_in_dim(
                replay.obs[bi, :, ei].astype(jnp.float32), si, L + 1
            )
            act_seq = jax.lax.dynamic_slice_in_dim(replay.action[bi, :, ei], si, L)
            rew_seq = jax.lax.dynamic_slice_in_dim(replay.reward[bi, :, ei], si, L)
            first_seq = jax.lax.dynamic_slice_in_dim(
                replay.done[bi, :, ei].astype(jnp.float32), si - 1, L
            )
            terminal_seq = jax.lax.dynamic_slice_in_dim(
                replay.termination[bi, :, ei].astype(jnp.float32), si, L
            )
            return obs_seq, act_seq, rew_seq, first_seq, terminal_seq

        return jax.vmap(take_seq)(block_idx, env_idx, start_idx)

    # ---------- Losses ----------

    def _model_loss(self, params, obs_seq, act_seq, rew_seq, first_seq, terminal_seq, rng):
        (hs, zs), (dyn_kls, rep_kls), feats, obs_pred, rew_logits, term_logits = (
            self.world.apply(
                {"params": params},
                obs_seq,
                act_seq,
                first_seq,
                method=WorldModel.observe,
                rngs={"sample": rng},
            )
        )
        hp = self.hparams
        B, L = act_seq.shape
        dyn_loss = jnp.mean(jnp.maximum(dyn_kls, hp.free_nats))
        rep_loss = jnp.mean(jnp.maximum(rep_kls, hp.free_nats))

        obs_target = rlax.signed_logp1(obs_seq[:, 1:].reshape(B * L, -1))
        obs_loss = jnp.mean(jnp.square(obs_pred - obs_target))

        # TwoHotHead.loss/.mean are plain array math (no nn.Dense/params
        # of their own beyond what already produced `rew_logits` inside
        # observe()'s .apply() context above), so a standalone instance -
        # not bound to any params - can call them directly, the same
        # pattern _actor_loss/_critic_loss use.
        reward_head = TwoHotHead(hp.hidden_size, hp.bins, hp.bins_low, hp.bins_high)
        rew_loss = jnp.mean(reward_head.loss(rew_logits, rew_seq.reshape(-1)))
        # The term head's target is TRUE environment termination only,
        # never timeout truncation - imagination uses this head as
        # `continues = 1 - sigmoid(term_logit)`, i.e. "does the MDP end
        # here", and a truncated episode didn't end because of the state,
        # it was cut short exogenously by a step limit the agent can't
        # even observe (time isn't in the observation). An earlier
        # version trained it on `done` (termination OR truncation):
        # since most unsuccessful episodes end by timeout, that taught
        # the model that episodes die spontaneously at an unpredictable
        # background rate, deflating `continues` - and with it every
        # imagined value target and rollout weight - everywhere.
        term_loss = jnp.mean(
            optax.sigmoid_binary_cross_entropy(
                jnp.squeeze(term_logits, -1), terminal_seq.reshape(-1)
            )
        )

        loss = (
            hp.dyn_scale * dyn_loss
            + hp.rep_scale * rep_loss
            + obs_loss
            + rew_loss
            + term_loss
        )
        logs = {
            "diagnostics/model/dyn_kl": dyn_loss,
            "diagnostics/model/rep_kl": rep_loss,
            "diagnostics/model/obs_loss": obs_loss,
            "diagnostics/model/rew_loss": rew_loss,
            "diagnostics/model/term_loss": term_loss,
        }
        return loss, (feats, logs)

    def _actor_critic_rollout(self, model_params, actor_params, start_feats, rng):
        """Shared imagination rollout for both the actor and critic
        losses - both need the same `(feats, rewards, continues, actions,
        logps)`, so this factors it out rather than re-running `imagine()`
        (a full `imag_horizon`-step scan) twice per Dreamer.update()."""
        hp = self.hparams
        h0, z0 = start_feats[:, : hp.recurrent_size], start_feats[:, hp.recurrent_size :]

        def actor_logits_fn(feat):
            return self.actor.apply({"params": actor_params}, feat)

        feats, rews, term_logits, actions, logps = self.world.apply(
            {"params": model_params},
            h0,
            z0,
            actor_logits_fn,
            hp.imag_horizon,
            rng,
            method=WorldModel.imagine,
        )
        continues = 1.0 - jax.nn.sigmoid(term_logits)
        # Index conventions, spelled out because an off-by-one here is
        # exactly the bug this refactor fixed: `imagine`'s scan emits only
        # *resulting* states - feats[t] is the state REACHED by actions[t],
        # which was taken FROM feats_in[t] (the seed `start_feats` at t=0,
        # feats[t-1] after). The lambda-return recursion below therefore
        # produces targets[t] = return of feats_in[t], so everything the
        # actor/critic losses evaluate per-step - the policy distribution
        # actions[t] is scored under, the value baseline subtracted from
        # targets[t], the critic's regression input - must be feats_in[t],
        # not feats[t]. An earlier version used feats[t] for all three:
        # the policy gradient scored each action under the distribution of
        # the state it *led to*, the advantage subtracted the *next*
        # state's value, and (worst, see `weight` below) the step weight
        # included the outcome state's own continuation probability.
        feats_in = jnp.concatenate([start_feats[:, None, :], feats[:, :-1]], axis=1)
        # weight[t] = prob the imagined trajectory is still alive when
        # actions[t] is taken, times the discount accumulated getting
        # there: prod_{i<t}(discount * continues[i]) - continues of the
        # states *before* the action, never continues[t] itself, which is
        # the continuation of the action's own outcome. Including it (as
        # `cumprod(discount*continues)/discount` over these same arrays
        # did before) multiplied each action's whole loss term by
        # 1 - P(terminal | its outcome): in a sparse goal task the action
        # that reaches the goal makes the term head fire, so the one
        # policy gradient carrying the reward signal was scaled toward
        # zero exactly when it mattered - actions at the (real) start
        # states get full weight 1 instead.
        weight = jnp.concatenate(
            [
                jnp.ones_like(continues[:, :1]),
                jnp.cumprod(hp.discount * continues, axis=1)[:, :-1],
            ],
            axis=1,
        )
        return feats_in, feats, rews, continues, actions, weight

    def _actor_loss(
        self, actor_params, model_params, critic_params, start_feats, return_norm_scale, rng
    ):
        hp = self.hparams
        feats_in, feats, rews, continues, actions, weight = self._actor_critic_rollout(
            model_params, actor_params, start_feats, rng
        )
        # vals[t] = V(feats[t]) is the bootstrap for rews[t] (the reward
        # for *reaching* feats[t]), so the lambda-return recursion yields
        # targets[t] = return of feats_in[t], the state actions[t] was
        # taken FROM (see _actor_critic_rollout's index-convention
        # comment). The baseline subtracted from it must therefore be
        # V(feats_in[t]), and the policy distribution actions[t] is scored
        # under must be the one at feats_in[t] - not feats[t], the state
        # the action *led to*, which an earlier version used for both.
        head = TwoHotHead(hp.hidden_size, hp.bins, hp.bins_low, hp.bins_high)
        vals_logits = jax.vmap(self.critic.apply, in_axes=(None, 0))(
            {"params": critic_params}, feats
        )
        vals = head.mean(vals_logits)
        targets = jax.lax.stop_gradient(
            self._lambda_returns(vals, rews, continues, hp.discount, hp.lam)
        )
        baseline_logits = jax.vmap(self.critic.apply, in_axes=(None, 0))(
            {"params": critic_params}, feats_in
        )
        baseline = head.mean(baseline_logits)
        adv = jax.lax.stop_gradient((targets - baseline) / return_norm_scale)

        # REINFORCE: log_prob(sg(action)) * sg(advantage), not backprop
        # through the sampled discrete action - distrax.Categorical.
        # sample() has no gradient w.r.t. its logits (no reparameterization
        # for discrete distributions), so differentiating an imagined
        # return through a *sampled* action trains nothing through that
        # path. Recompute log_prob under the current actor_params (the
        # rollout's own logps were computed under a frozen snapshot for
        # the imagination pass, but grad must flow through actor_params
        # here).
        actor_logits = jax.vmap(
            lambda f: self.actor.apply({"params": actor_params}, f)
        )(feats_in)
        dist = unimix_categorical(actor_logits, hp.actor_unimix)
        logp = dist.log_prob(jax.lax.stop_gradient(actions))
        entropy = dist.entropy()
        policy_loss = jax.lax.stop_gradient(weight) * -(
            logp * adv + hp.actor_entropy * entropy
        )
        loss = policy_loss.mean()
        logs = {
            "diagnostics/actor/loss": loss,
            "diagnostics/actor/entropy": entropy.mean(),
            "diagnostics/actor/adv": adv.mean(),
            # Imagination health: if the world model is any good, imagined
            # reward should be on the same scale as the real collected
            # reward rate, and targets should track true policy value -
            # divergence between these and reality is the signature of the
            # actor optimizing against model error instead of the task.
            "diagnostics/imag/rew": rews.mean(),
            "diagnostics/imag/continues": continues.mean(),
            "diagnostics/imag/target": targets.mean(),
        }
        return loss, logs

    def _critic_loss(
        self, critic_params, model_params, actor_params, slow_critic_params, start_feats, rng
    ):
        hp = self.hparams
        feats_in, feats, rews, continues, actions, weight = self._actor_critic_rollout(
            model_params, actor_params, start_feats, rng
        )
        # targets[t] is the return of feats_in[t] (see _actor_critic_
        # rollout's index-convention comment), so the critic regresses
        # V(feats_in[t]) toward it - an earlier version regressed
        # V(feats[t]), training every state's value toward the *previous*
        # state's return, which (among other biases) taught the value of
        # a goal state to include the reward already collected by
        # *reaching* it. As a bonus, feats_in[0] is a real posterior
        # state (from replayed experience), so the critic also trains
        # directly on real states, as the official implementation does.
        head = TwoHotHead(hp.hidden_size, hp.bins, hp.bins_low, hp.bins_high)
        vals_logits = jax.vmap(self.critic.apply, in_axes=(None, 0))(
            {"params": critic_params}, feats
        )
        vals = jax.vmap(head.mean)(vals_logits)
        targets = jax.lax.stop_gradient(
            self._lambda_returns(vals, rews, continues, hp.discount, hp.lam)
        )

        pred_logits = jax.vmap(self.critic.apply, in_axes=(None, 0))(
            {"params": critic_params}, feats_in
        )
        regress_loss = jnp.mean(
            jax.lax.stop_gradient(weight) * jax.vmap(head.loss)(pred_logits, targets)
        )

        slow_logits = jax.lax.stop_gradient(
            jax.vmap(self.critic.apply, in_axes=(None, 0))(
                {"params": slow_critic_params}, feats_in
            )
        )
        slow_target = jax.lax.stop_gradient(head.mean(slow_logits))
        slow_reg_loss = jnp.mean(
            jax.lax.stop_gradient(weight) * jax.vmap(head.loss)(pred_logits, slow_target)
        )

        loss = regress_loss + hp.slow_critic_reg * slow_reg_loss
        logs = {
            "diagnostics/critic/loss": loss,
            "diagnostics/critic/regress_loss": regress_loss,
            "diagnostics/critic/slow_reg_loss": slow_reg_loss,
            "diagnostics/critic/value": jax.vmap(head.mean)(pred_logits).mean(),
        }
        return loss, logs

    # ---------- One update (collect + model/actor/critic) ----------

    def update(self, ts: DreamerTrainState, _) -> Tuple[DreamerTrainState, Dict]:
        hp = self.hparams

        ts, experience = self.collect_experience(ts)
        replay = self._write_replay(ts.replay, experience)
        ts = ts.replace(replay=replay)

        def model_step(carry, _):
            model, rng = carry
            rng, key_batch, key_loss = jax.random.split(rng, 3)
            obs_seq, act_seq, rew_seq, first_seq, terminal_seq = self._sample_batch(
                key_batch, replay
            )

            def loss_fn(p):
                return self._model_loss(
                    p, obs_seq, act_seq, rew_seq, first_seq, terminal_seq, key_loss
                )

            (loss, (feats, mlogs)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                model.params
            )
            grads = jax.tree.map(
                lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), grads
            )
            model = model.apply_gradients(grads=grads)
            return (model, rng), {"diagnostics/model": loss, **mlogs}

        (model, rng), mlogs = jax.lax.scan(
            model_step, (ts.model, ts.rng), None, length=hp.num_model_updates
        )
        mlogs = jax.tree.map(lambda x: jnp.mean(x), mlogs)

        # One fresh batch of features to seed the actor/critic imagination
        # rollouts from, reusing the just-updated model.
        rng, key_batch, key_observe = jax.random.split(rng, 3)
        obs_seq, act_seq, _, first_seq, _ = self._sample_batch(key_batch, replay)
        _, _, feats, _, _, _ = self.world.apply(
            {"params": model.params},
            obs_seq,
            act_seq,
            first_seq,
            method=WorldModel.observe,
            rngs={"sample": key_observe},
        )
        start_feats = jax.lax.stop_gradient(
            feats[:, :-1].reshape(-1, feats.shape[-1])
        )

        # Return-normalization scale: EMA-tracked 5th/95th percentile of
        # a quick lambda-return estimate under the current critic, so the
        # actor's advantage is on a comparable scale regardless of the
        # environment's raw reward magnitude. Split off a dedicated key
        # for this probe rollout - `rng` itself is reused just below as
        # the actor_step scan's carry, and reusing the same key for both
        # would make the probe's imagined trajectory and the first actor
        # gradient step draw from identical randomness.
        rng, key_probe = jax.random.split(rng)
        _, probe_feats, probe_rews, probe_continues, _, _ = self._actor_critic_rollout(
            model.params, ts.actor.params, start_feats, key_probe
        )
        head = TwoHotHead(hp.hidden_size, hp.bins, hp.bins_low, hp.bins_high)
        probe_vals_logits = jax.vmap(self.critic.apply, in_axes=(None, 0))(
            {"params": ts.critic.params}, probe_feats
        )
        probe_vals = jax.vmap(head.mean)(probe_vals_logits)
        probe_returns = self._lambda_returns(
            probe_vals, probe_rews, probe_continues, hp.discount, hp.lam
        )
        lo = jnp.percentile(probe_returns, 5)
        hi = jnp.percentile(probe_returns, 95)
        rate = hp.return_norm_rate
        return_norm_lo = (1 - rate) * ts.return_norm_lo + rate * lo
        return_norm_hi = (1 - rate) * ts.return_norm_hi + rate * hi
        return_norm_scale = jnp.maximum(
            return_norm_hi - return_norm_lo, hp.return_norm_limit
        )

        def actor_step(carry, _):
            actor, rng = carry
            rng, key = jax.random.split(rng)

            def loss_fn(p):
                return self._actor_loss(
                    p, model.params, ts.critic.params, start_feats, return_norm_scale, key
                )

            (loss, alogs), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                actor.params
            )
            grads = jax.tree.map(
                lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), grads
            )
            actor = actor.apply_gradients(grads=grads)
            return (actor, rng), {"diagnostics/actor": loss, **alogs}

        (actor, rng), alogs = jax.lax.scan(
            actor_step, (ts.actor, rng), None, length=hp.num_actor_updates
        )
        alogs = jax.tree.map(lambda x: jnp.mean(x), alogs)

        def critic_step(carry, _):
            critic, slow_critic_params, rng = carry
            rng, key = jax.random.split(rng)

            def loss_fn(p):
                return self._critic_loss(
                    p, model.params, actor.params, slow_critic_params, start_feats, key
                )

            (loss, clogs), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                critic.params
            )
            grads = jax.tree.map(
                lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), grads
            )
            critic = critic.apply_gradients(grads=grads)
            # The slow critic's EMA advances once per GRADIENT STEP (as in
            # the official implementation), not once per update() cycle.
            # With slow_critic_reg = 1.0 the online critic is regularized
            # toward the slow one as strongly as toward its actual
            # lambda-return targets, so a slow critic that only moved
            # rate=0.02 per *cycle* (i.e. per num_critic_updates gradient
            # steps - 32x more laggard than official) froze the pair near
            # their (zero) init: values were observed to crawl at ~1/10 of
            # their true scale over a whole training run, keeping
            # advantages - and with them the actor's learning signal -
            # near zero long after the world model had learned where the
            # reward is.
            slow_critic_params = jax.tree.map(
                lambda slow, online: (1 - hp.slow_critic_rate) * slow
                + hp.slow_critic_rate * online,
                slow_critic_params,
                critic.params,
            )
            return (critic, slow_critic_params, rng), {"diagnostics/critic": loss, **clogs}

        (critic, slow_critic_params, rng), clogs = jax.lax.scan(
            critic_step,
            (ts.critic, ts.slow_critic_params, rng),
            None,
            length=hp.num_critic_updates,
        )
        clogs = jax.tree.map(lambda x: jnp.mean(x), clogs)

        ts = ts.replace(
            model=model,
            actor=actor,
            critic=critic,
            slow_critic_params=slow_critic_params,
            rng=rng,
            updates=ts.updates + 1,
            return_norm_lo=return_norm_lo,
            return_norm_hi=return_norm_hi,
        )

        logs = {}
        logs.update(mlogs)
        logs.update(alogs)
        logs.update(clogs)
        logs["train/done_mask"] = experience.done
        logs["train/returns"] = experience.info["return"]
        logs["train/lengths"] = experience.t
        logs["train/frames"] = ts.frames
        logs["train/updates"] = ts.updates
        # Not guaranteed uniform across every navix agent (PPO/PQN
        # have one learning rate, not three - see Agent.train's
        # docstring) - diagnostics/*, not train/*.
        logs["diagnostics/model_lr"] = self.hparams.model_lr
        logs["diagnostics/actor_lr"] = self.hparams.actor_lr
        logs["diagnostics/critic_lr"] = self.hparams.critic_lr
        logs["diagnostics/return_norm_scale"] = return_norm_scale

        if self.hparams.log_render:
            from ..observations import rgb

            b = jax.random.randint(ts.rng, (), 0, self.hparams.num_envs)
            logs["render/human"] = jax.vmap(rgb)(
                jax.tree.map(lambda x: x[:, b], experience.state)
            ).transpose((0, 3, 1, 2))

        if self.hparams.debug:
            jax.debug.callback(self.log_to_wandb, logs, experience)

        return ts, logs

    # ---------- Train entry point ----------

    def train_first_update(self, rng: Array) -> Tuple[DreamerTrainState, Dict]:
        """Runs initialisation plus exactly one `update()` call. Mainly
        useful for tests/debugging, where running a full `train()` (sized
        by `hparams.budget`) is either unnecessary or awkward to size
        exactly."""
        ts = DreamerTrainState.create(
            rng, self.hparams, self.env, self.world, self.actor, self.critic
        )
        return self.update(ts, None)

    def train(self, rng: Array) -> Tuple[DreamerTrainState, Dict]:
        hp = self.hparams
        ts = DreamerTrainState.create(
            rng, self.hparams, self.env, self.world, self.actor, self.critic
        )
        num_updates = hp.budget // (hp.num_steps * hp.num_envs)

        # iter/fps and iter/wall_time are NOT set here - train() runs inside
        # a jax.jit trace (see Experiment.run), where time.time() only ever
        # fires once, at trace-build time. Experiment.run fills both in
        # itself, from real wall-clock timing measured outside any trace.
        ts, logs = jax.lax.scan(self.update, ts, None, length=num_updates)

        return ts, logs
