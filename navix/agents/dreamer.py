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

# A from-scratch DreamerV3 (Hafner et al., "Mastering Diverse Domains
# through World Models", https://arxiv.org/abs/2301.04104) agent: an RSSM
# world model with categorical latents, trained jointly with an actor and
# critic on imagined rollouts. Implements the paper's five headline
# robustness techniques, cross-checked directly against the official
# implementation (github.com/danijar/dreamerv3, dreamerv3/rssm.py and
# embodied/jax/agent.py) rather than assumed from the paper text alone:
#
#   1. Symlog inputs/reconstruction (`symlog`/`symexp` below).
#   2. Categorical latents (`stoch` independent categoricals of `classes`
#      each) with straight-through gradients and 1% "unimix" - mixing a
#      little uniform mass into every categorical, so no class ever gets
#      a literal zero probability - for both the prior and posterior.
#   3. KL balancing with free bits: two separate KL terms with different
#      stop-gradient placement and independent free-nats floors, not one
#      combined KL clamped by a single scalar.
#   4. Symexp-twohot regression for reward and value (`TwoHotHead`), not
#      a Gaussian/MSE head - a discrete classification loss over an
#      exponentially-spaced grid of bins, which is far less sensitive to
#      reward-scale outliers than a Gaussian likelihood.
#   5. Return normalization: an EMA-tracked 5th-95th percentile range of
#      returns rescales advantages, so the policy gradient's magnitude
#      stays stable across environments with very different reward
#      scales, without per-environment tuning.
#
# Also matches the official implementation's actor loss shape: REINFORCE
# (log_prob(action) * advantage + entropy bonus), not backpropagation
# through sampled discrete actions - discrete distrax.Categorical.sample()
# has no gradient path back to its logits, so naively differentiating an
# imagined-rollout return through a *sampled* discrete action (as an
# earlier draft of this agent did) trains nothing through that path at
# all - and an EMA "slow" target critic that the online critic is
# regularized toward, for training stability.
#
# Deliberate simplifications, kept for navix's small grid-world
# observations rather than image-scale ones: a plain nn.GRUCell for the
# deterministic recurrent state (not the official "block GRU", a
# parameter-efficiency optimization for much larger deter sizes); a plain
# symlog+MSE decoder for observation reconstruction (not the official
# implementation's image-specific CNN decoder); ELU activations and no
# RMSNorm (not load-bearing for correctness, just a smaller/simpler net);
# `stoch`/`classes` default to 8x8 rather than the paper's 32x32, sized
# for navix's small grids rather than Atari-scale observations. None of
# these are the algorithmic identity of DreamerV3 - the five techniques
# above are.

from __future__ import annotations
import time
from typing import Callable, Dict, Tuple

import numpy as np
import distrax
import jax
import jax.numpy as jnp
from jax import Array
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
from flax import struct

from .agent import Agent, HParams
from ..environments import Environment
from ..environments.environment import Timestep
from ..states import State


# -------------------------
# Symlog / symexp
# -------------------------


def symlog(x: Array) -> Array:
    """`sign(x) * log(1 + |x|)` - compresses large magnitudes so a network
    doesn't need to represent, e.g., both a reward of 1 and one of 1000 on
    the same linear scale. Self-inverse with `symexp`."""
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def symexp(x: Array) -> Array:
    """Inverse of `symlog`: `sign(x) * (exp(|x|) - 1)`."""
    return jnp.sign(x) * jnp.expm1(jnp.abs(x))


# -------------------------
# Categorical latents: unimix + straight-through sampling
# -------------------------


def unimix_categorical(logits: Array, unimix: float) -> distrax.Categorical:
    """Builds a `distrax.Categorical` from `logits` after mixing in a
    `unimix` fraction of uniform probability mass across the last axis -
    the "unimix" trick (1% by default in the paper): guarantees every
    class keeps at least `unimix / classes` probability, so neither the
    KL term nor the entropy can collapse to exactly zero, which keeps the
    prior from ever fully committing and losing gradient signal."""
    probs = jax.nn.softmax(logits, axis=-1)
    uniform = jnp.ones_like(probs) / probs.shape[-1]
    probs = (1.0 - unimix) * probs + unimix * uniform
    return distrax.Categorical(probs=probs)


def straight_through_sample(dist: distrax.Categorical, rng: Array) -> Array:
    """Samples a one-hot vector from `dist`, with a straight-through
    gradient: the forward value is a genuine (discrete) sample, but the
    backward gradient flows as if the output were `dist.probs` directly
    (`sg(onehot - probs) + probs` has forward value `onehot`, since
    `onehot - probs` is stop-gradiented, but its Jacobian w.r.t. upstream
    parameters is `probs`'s). This is what makes the RSSM's own latent
    trainable end-to-end despite being discrete - distinct from the
    *actor's* action sampling below, which deliberately does NOT use
    this (see the module docstring)."""
    idx = dist.sample(seed=rng)
    onehot = jax.nn.one_hot(idx, dist.probs.shape[-1])
    return jax.lax.stop_gradient(onehot - dist.probs) + dist.probs


def categorical_kl(post: distrax.Categorical, prior: distrax.Categorical) -> Array:
    """Sum of `KL(post_i || prior_i)` over the `stoch` independent
    categoricals (the second-to-last axis) - each categorical's own KL,
    summed, matching how DreamerV3 treats the full stochastic latent
    (`stoch` categoricals of `classes` each) as one joint distribution
    for the purposes of the free-nats floor."""
    return post.kl_divergence(prior).sum(axis=-1)


# -------------------------
# Symexp-twohot head (reward, value)
# -------------------------


def twohot_encode(x: Array, bin_centers: Array) -> Array:
    """Encodes scalar `x` (already in symlog space) as a soft one-hot
    vector over `bin_centers` (ascending, evenly spaced): all mass on the
    two bins bracketing `x`, split by linear interpolation. This is the
    "twohot" target the classification loss is computed against - lets a
    discrete softmax head represent a continuous value to sub-bin
    precision, rather than being limited to `bins`-many exact outputs."""
    K = bin_centers.shape[0]
    x = jnp.clip(x, bin_centers[0], bin_centers[-1])
    below = jnp.sum(bin_centers[None, ...] <= x[..., None], axis=-1) - 1
    below = jnp.clip(below, 0, K - 2)
    above = below + 1
    lo, hi = bin_centers[below], bin_centers[above]
    weight_hi = jnp.where(hi > lo, (x - lo) / (hi - lo), 0.0)
    onehot_lo = jax.nn.one_hot(below, K)
    onehot_hi = jax.nn.one_hot(above, K)
    return onehot_lo * (1.0 - weight_hi)[..., None] + onehot_hi * weight_hi[..., None]


class TwoHotHead(nn.Module):
    """A scalar-valued prediction head (reward, value) implemented as
    classification over `bins` evenly-spaced bins in symlog space, with a
    twohot cross-entropy loss - the "symexp twohot" head from the paper.
    Far more robust to reward/value outliers than a Gaussian/MSE head,
    since an extreme target only ever saturates the loss for its two
    nearest bins, not the whole (unbounded) squared-error term."""

    hidden_size: int
    bins: int = 255
    low: float = -20.0
    high: float = 20.0

    @nn.compact
    def __call__(self, feat: Array) -> Array:
        net = nn.Sequential(
            [
                nn.Dense(self.hidden_size),
                nn.elu,
                nn.Dense(self.hidden_size),
                nn.elu,
                nn.Dense(self.bins),
            ]
        )
        return net(feat)  # logits, (..., bins)

    def loss(self, logits: Array, target: Array) -> Array:
        bin_centers = jnp.linspace(self.low, self.high, self.bins)
        twohot = twohot_encode(symlog(target), bin_centers)
        logp = jax.nn.log_softmax(logits, axis=-1)
        return -jnp.sum(twohot * logp, axis=-1)

    def mean(self, logits: Array) -> Array:
        bin_centers = jnp.linspace(self.low, self.high, self.bins)
        probs = jax.nn.softmax(logits, axis=-1)
        return symexp(jnp.sum(probs * bin_centers, axis=-1))


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
    num_model_updates: int = struct.field(pytree_node=False, default=32)
    """Number of world-model gradient steps per update."""
    num_actor_updates: int = struct.field(pytree_node=False, default=32)
    """Number of actor gradient steps per update."""
    num_critic_updates: int = struct.field(pytree_node=False, default=32)
    """Number of critic gradient steps per update."""
    batch_size: int = struct.field(pytree_node=False, default=64)
    """Number of sequences sampled per world-model gradient step."""
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
    stoch: int = struct.field(pytree_node=False, default=8)
    """Number of independent categorical latent variables."""
    classes: int = struct.field(pytree_node=False, default=8)
    """Number of classes per categorical latent variable."""
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
    `_collect` (which samples actions from this same distribution) stops
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
    deter_size: int = 200
    """Size of the RSSM's deterministic (GRU) hidden state."""
    hidden_size: int = 200
    """Hidden layer size used throughout the model/actor/critic MLPs."""

    @property
    def stoch_flat(self) -> int:
        """Flattened size of the categorical latent (`stoch * classes`),
        i.e. how much of `feat = concat([h, z_flat])` the latent occupies."""
        return self.stoch * self.classes


# -------------------------
# World model: RSSM with categorical latents
# -------------------------


class Encoder(nn.Module):
    hidden_size: int
    embed_size: int

    @nn.compact
    def __call__(self, obs: Array) -> Array:
        net = nn.Sequential(
            [
                nn.Dense(self.hidden_size),
                nn.elu,
                nn.Dense(self.hidden_size),
                nn.elu,
                nn.Dense(self.embed_size),
            ]
        )
        return net(symlog(obs))


class Decoder(nn.Module):
    hidden_size: int
    obs_dim: int

    @nn.compact
    def __call__(self, feat: Array) -> Array:
        """Returns a prediction of `symlog(obs)` directly (not a
        distribution) - reconstruction loss is plain MSE against
        `symlog(obs)`, the paper's "symlog MSE" observation head, simpler
        than the image-specific decoder the official implementation uses
        since navix's observations here are flat vectors, not pixels."""
        net = nn.Sequential(
            [
                nn.Dense(self.hidden_size),
                nn.elu,
                nn.Dense(self.hidden_size),
                nn.elu,
                nn.Dense(self.obs_dim),
            ]
        )
        return net(feat)


class RSSM(nn.Module):
    deter_size: int

    @nn.compact
    def __call__(self, h_prev: Array, z_prev_flat: Array, a_prev_oh: Array) -> Array:
        x = jnp.concatenate([z_prev_flat, a_prev_oh], axis=-1)
        gru = nn.GRUCell(features=self.deter_size)
        h, _ = gru(h_prev, x)  # GRUCell returns (new_carry, y); y == new_carry
        return h


class PriorNet(nn.Module):
    hidden_size: int
    stoch: int
    classes: int

    @nn.compact
    def __call__(self, h: Array) -> Array:
        """Returns raw logits, `(..., stoch, classes)` - unimix is applied
        by the caller (`unimix_categorical`), not baked in here, so every
        caller treats prior and posterior identically."""
        x = nn.elu(nn.Dense(self.hidden_size)(h))
        logits = nn.Dense(self.stoch * self.classes)(x)
        return logits.reshape(*h.shape[:-1], self.stoch, self.classes)


class PostNet(nn.Module):
    hidden_size: int
    stoch: int
    classes: int

    @nn.compact
    def __call__(self, h: Array, embed: Array) -> Array:
        x = jnp.concatenate([h, embed], axis=-1)
        x = nn.elu(nn.Dense(self.hidden_size)(x))
        logits = nn.Dense(self.stoch * self.classes)(x)
        return logits.reshape(*h.shape[:-1], self.stoch, self.classes)


class WorldModel(nn.Module):
    obs_dim: int
    act_dim: int
    hparams: DreamerHparams

    def setup(self):
        hp = self.hparams
        self.encoder = Encoder(hp.hidden_size, hp.embed_size)
        self.rssm = RSSM(hp.deter_size)
        self.prior = PriorNet(hp.hidden_size, hp.stoch, hp.classes)
        self.post = PostNet(hp.hidden_size, hp.stoch, hp.classes)
        self.decoder = Decoder(hp.hidden_size, self.obs_dim)
        self.reward = TwoHotHead(hp.hidden_size, hp.bins, hp.bins_low, hp.bins_high)
        self.term = nn.Sequential(
            [
                nn.Dense(hp.hidden_size),
                nn.elu,
                nn.Dense(hp.hidden_size),
                nn.elu,
                nn.Dense(1),
            ]
        )

    def init_state(self, batch_size: int) -> Tuple[Array, Array, Array]:
        h = jnp.zeros((batch_size, self.hparams.deter_size))
        z_flat = jnp.zeros((batch_size, self.hparams.stoch_flat))
        a0 = jnp.zeros((batch_size, self.act_dim))
        return h, z_flat, a0

    def feat(self, h: Array, z_flat: Array) -> Array:
        return jnp.concatenate([h, z_flat], axis=-1)

    def observe(self, obs_seq: Array, act_seq: Array, term_seq: Array) -> Tuple[
        Tuple[Array, Array], Tuple[Array, Array], Array, Array, Array, Array
    ]:
        """Runs the RSSM over an observed sequence, computing the posterior
        latent at every step and decoding obs/reward/term from it.

        Args:
            obs_seq (Array): `f32[B, L+1, obs_dim]`, flattened observations.
            act_seq (Array): `i32[B, L]`, actions taken between consecutive
                observations.
            term_seq (Array): `f32[B, L]`, `1.0` where `act_seq[:, t]` ended
                an episode (matching `experience.done`/the `_model_loss`
                term-head target) - `term_seq[:, t] == 1` means
                `obs_seq[:, t + 1]` is a fresh post-autoreset observation,
                *not* a real consequence of `act_seq[:, t]`. Sampled
                training sequences are sliced from a fixed-size buffer with
                no regard for episode boundaries (`_sample_batch`), so a
                sequence commonly straddles an autoreset; without this,
                every such step would condition the reset observation's
                posterior on the *previous* episode's (h, z) and pair it
                with an action that didn't actually cause it - a genuine
                transition the RSSM cannot learn (the reset is exogenous),
                and, left unhandled, the resulting garbage (h, z) then
                propagates forward through every subsequent step of the new
                episode still inside this sequence, not just the one
                boundary step. Matches the official implementation's
                `is_first` masking in `RSSM.observe` (`embodied/jax/
                rssm.py`): at scan step `t` (which pairs `act_seq[:, t]`
                with `embed(obs_seq[:, t + 1])`), both the incoming (h, z)
                carried from step `t - 1` and `act_seq[:, t]` itself are
                zeroed whenever `term_seq[:, t] == 1`, so `obs_seq[:, t +
                1]`'s posterior is computed from a blank slate instead of a
                stale, causally-unrelated belief plus an action that
                didn't really produce it.

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
            (a_oh, embed_all[:, 1:], term_seq.astype(jnp.float32)),
        )
        init_rng = self.make_rng("sample")
        (h_last, z_last, _), (hs, zs, dyn_kls, rep_kls, feats) = jax.lax.scan(
            step,
            (
                jnp.zeros((B, hp.deter_size)),
                jnp.zeros((B, hp.stoch_flat)),
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
        method=WorldModel.posterior_step)`) provides for `_collect`."""
        h_next = self.rssm(h, z_flat, a_prev_oh)
        embed = self.encoder(obs)
        post_dist = unimix_categorical(self.post(h_next, embed), self.hparams.unimix)
        z_next = straight_through_sample(post_dist, rng)
        z_next_flat = z_next.reshape(*z_next.shape[:-2], -1)
        return h_next, z_next_flat

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
        h = jnp.zeros((obs.shape[0], hp.deter_size))
        z_flat = jnp.zeros((obs.shape[0], hp.stoch_flat))
        h_t = self.rssm(h, z_flat, a_prev_oh)
        self.prior(h_t)
        embed = self.encoder(obs)
        post_logits = self.post(h_t, embed)
        z_t = jax.nn.one_hot(jnp.zeros(obs.shape[0], dtype=jnp.int32), hp.classes)
        z_t = jnp.broadcast_to(z_t[:, None, :], (obs.shape[0], hp.stoch, hp.classes))
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
        net = nn.Sequential(
            [
                nn.Dense(self.hidden),
                nn.elu,
                nn.Dense(self.hidden),
                nn.elu,
                nn.Dense(self.act_dim),
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
    action: Array
    reward: Array
    log_prob: Array
    obs: Array
    info: Dict[str, Array]
    t: Array
    state: State


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

    env_state: Timestep
    rng: Array
    frames: Array
    updates: Array
    return_norm_lo: Array  # EMA of the 5th percentile of returns
    return_norm_hi: Array  # EMA of the 95th percentile of returns

    # Latents per-env, carried across collection steps.
    h: Array  # (N, deter_size)
    z: Array  # (N, stoch_flat)
    a_prev_oh: Array  # (N, act_dim)


# -------------------------
# Agent
# -------------------------


class Dreamer(Agent):
    hparams: DreamerHparams
    env: Environment
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
        """shapes: (B, H), (B, H), (B, H) -> (B, H). `continues` is
        `1 - terminal`. jax.lax.scan only ever scans axis 0, but these are
        batch-major (B, H) with H (time) as axis 1 - swap to (H, B) in and
        out around the scan itself."""

        def scan_fn(carry, inputs):
            v_lambda_next = carry
            r_t, cont_t, v_tp1 = inputs
            td = r_t + cont_t * (1.0 - lam) * discount * v_tp1
            v_lambda = td + cont_t * v_lambda_next * lam * discount
            return v_lambda, v_lambda

        init = next_values[:, -1]
        rewards_t, continues_t, next_values_t = jax.tree.map(
            lambda x: jnp.swapaxes(x, 0, 1)[::-1], (rewards, continues, next_values)
        )
        _, vals_t = jax.lax.scan(
            scan_fn, init, (rewards_t, continues_t, next_values_t), length=rewards.shape[1]
        )
        return jnp.swapaxes(vals_t[::-1], 0, 1)

    # ---------- Collection ----------

    def _collect(self, ts: DreamerTrainState) -> Tuple[DreamerTrainState, Buffer]:
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

    def _sample_batch(
        self, rng: Array, experience: Buffer
    ) -> Tuple[Array, Array, Array, Array]:
        """Samples `hparams.batch_size` sequences of length `hparams.seq_len
        + 1` observations / `seq_len` actions from the most recent
        rollout, for world-model training."""
        hp = self.hparams
        T, N = experience.obs.shape[0], experience.obs.shape[1]
        L = hp.seq_len
        max_start = jnp.maximum(T - (L + 1), 0)

        rng, key1, key2 = jax.random.split(rng, 3)
        env_idx = jax.random.randint(key1, (hp.batch_size,), minval=0, maxval=N)
        start_idx = jax.random.randint(
            key2, (hp.batch_size,), minval=0, maxval=jnp.maximum(1, max_start + 1)
        )

        def take_seq(ei, si):
            obs_seq = jax.lax.dynamic_slice_in_dim(experience.obs[:, ei], si, L + 1)
            act_seq = jax.lax.dynamic_slice_in_dim(experience.action[:, ei], si, L)
            rew_seq = jax.lax.dynamic_slice_in_dim(experience.reward[:, ei], si, L)
            term_seq = jax.lax.dynamic_slice_in_dim(
                experience.done[:, ei].astype(jnp.float32), si, L
            )
            return obs_seq, act_seq, rew_seq, term_seq

        obs_seq, act_seq, rew_seq, term_seq = jax.vmap(take_seq)(env_idx, start_idx)
        return self._flatten_obs(obs_seq.reshape(-1, *obs_seq.shape[2:])).reshape(
            hp.batch_size, L + 1, -1
        ), act_seq, rew_seq, term_seq

    # ---------- Losses ----------

    def _model_loss(self, params, obs_seq, act_seq, rew_seq, term_seq, rng):
        (hs, zs), (dyn_kls, rep_kls), feats, obs_pred, rew_logits, term_logits = (
            self.world.apply(
                {"params": params},
                obs_seq,
                act_seq,
                term_seq,
                method=WorldModel.observe,
                rngs={"sample": rng},
            )
        )
        hp = self.hparams
        B, L = act_seq.shape
        dyn_loss = jnp.mean(jnp.maximum(dyn_kls, hp.free_nats))
        rep_loss = jnp.mean(jnp.maximum(rep_kls, hp.free_nats))

        obs_target = symlog(obs_seq[:, 1:].reshape(B * L, -1))
        obs_loss = jnp.mean(jnp.square(obs_pred - obs_target))

        # TwoHotHead.loss/.mean are plain array math (no nn.Dense/params
        # of their own beyond what already produced `rew_logits` inside
        # observe()'s .apply() context above), so a standalone instance -
        # not bound to any params - can call them directly, the same
        # pattern _actor_loss/_critic_loss use.
        reward_head = TwoHotHead(hp.hidden_size, hp.bins, hp.bins_low, hp.bins_high)
        rew_loss = jnp.mean(reward_head.loss(rew_logits, rew_seq.reshape(-1)))
        term_loss = jnp.mean(
            optax.sigmoid_binary_cross_entropy(
                jnp.squeeze(term_logits, -1), term_seq.reshape(-1)
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
            "agent/model/dyn_kl": dyn_loss,
            "agent/model/rep_kl": rep_loss,
            "agent/model/obs_loss": obs_loss,
            "agent/model/rew_loss": rew_loss,
            "agent/model/term_loss": term_loss,
        }
        return loss, (feats, logs)

    def _actor_critic_rollout(self, model_params, actor_params, start_feats, rng):
        """Shared imagination rollout for both the actor and critic
        losses - both need the same `(feats, rewards, continues, actions,
        logps)`, so this factors it out rather than re-running `imagine()`
        (a full `imag_horizon`-step scan) twice per Dreamer.update()."""
        hp = self.hparams
        h0, z0 = start_feats[:, : hp.deter_size], start_feats[:, hp.deter_size :]

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
        # weight[t] down-weights step t by how much discount/continuation
        # probability has decayed by the time it's reached - later imagined
        # steps count for less both because they're discounted more, and
        # because the rollout may well have "really" terminated earlier
        # than this imagined continuation pretends (continues < 1 there).
        # Matches the official implementation's `cumprod(disc*con)/disc`.
        weight = jnp.cumprod(hp.discount * continues, axis=1) / hp.discount
        return feats, rews, continues, actions, logps, weight

    def _actor_loss(
        self, actor_params, model_params, critic_params, start_feats, return_norm_scale, rng
    ):
        hp = self.hparams
        feats, rews, continues, actions, _, weight = self._actor_critic_rollout(
            model_params, actor_params, start_feats, rng
        )
        # feats[t]/rews[t] are already aligned to the same imagined
        # transition (rews[t] is the reward *for reaching* feats[t]), so
        # vals[t] = V(feats[t]) is exactly the right bootstrap for
        # rews[t] - next_values=vals, rewards=rews need no shift. An
        # earlier version paired rews[:-1] with vals[1:], bootstrapping
        # every reward with the value *two* imagined steps ahead instead
        # of one - a systematic bias that made the advantage strongly,
        # persistently negative from the very first update regardless of
        # entropy coefficient or gradient-clip norm (verified: varying
        # either by 3+ orders of magnitude left the advantage trajectory
        # essentially unchanged, since the bug is in what's being
        # computed, not in how big a step is taken from it).
        head = TwoHotHead(hp.hidden_size, hp.bins, hp.bins_low, hp.bins_high)
        vals_logits = jax.vmap(self.critic.apply, in_axes=(None, 0))(
            {"params": critic_params}, feats
        )
        vals = head.mean(vals_logits)
        targets = jax.lax.stop_gradient(
            self._lambda_returns(vals, rews, continues, hp.discount, hp.lam)
        )
        adv = jax.lax.stop_gradient((targets - vals) / return_norm_scale)

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
        )(feats)
        dist = unimix_categorical(actor_logits, hp.actor_unimix)
        logp = dist.log_prob(jax.lax.stop_gradient(actions))
        entropy = dist.entropy()
        policy_loss = jax.lax.stop_gradient(weight) * -(
            logp * adv + hp.actor_entropy * entropy
        )
        loss = policy_loss.mean()
        logs = {
            "agent/actor/loss": loss,
            "agent/actor/entropy": entropy.mean(),
            "agent/actor/adv": adv.mean(),
        }
        return loss, logs

    def _critic_loss(
        self, critic_params, model_params, actor_params, slow_critic_params, start_feats, rng
    ):
        hp = self.hparams
        feats, rews, continues, actions, _, weight = self._actor_critic_rollout(
            model_params, actor_params, start_feats, rng
        )
        head = TwoHotHead(hp.hidden_size, hp.bins, hp.bins_low, hp.bins_high)
        vals_logits = jax.vmap(self.critic.apply, in_axes=(None, 0))(
            {"params": critic_params}, feats
        )
        vals = jax.vmap(head.mean)(vals_logits)
        targets = jax.lax.stop_gradient(
            self._lambda_returns(vals, rews, continues, hp.discount, hp.lam)
        )

        pred_logits = jax.vmap(self.critic.apply, in_axes=(None, 0))(
            {"params": critic_params}, feats
        )
        regress_loss = jnp.mean(
            jax.lax.stop_gradient(weight) * jax.vmap(head.loss)(pred_logits, targets)
        )

        slow_logits = jax.lax.stop_gradient(
            jax.vmap(self.critic.apply, in_axes=(None, 0))(
                {"params": slow_critic_params}, feats
            )
        )
        slow_target = jax.lax.stop_gradient(head.mean(slow_logits))
        slow_reg_loss = jnp.mean(
            jax.lax.stop_gradient(weight) * jax.vmap(head.loss)(pred_logits, slow_target)
        )

        loss = regress_loss + hp.slow_critic_reg * slow_reg_loss
        logs = {
            "agent/critic/loss": loss,
            "agent/critic/regress_loss": regress_loss,
            "agent/critic/slow_reg_loss": slow_reg_loss,
            "agent/critic/value": vals.mean(),
        }
        return loss, logs

    # ---------- One update (collect + model/actor/critic) ----------

    def update(self, ts: DreamerTrainState, _) -> Tuple[DreamerTrainState, Dict]:
        hp = self.hparams

        ts, experience = self._collect(ts)

        def model_step(carry, _):
            model, rng = carry
            rng, key_batch, key_loss = jax.random.split(rng, 3)
            obs_seq, act_seq, rew_seq, term_seq = self._sample_batch(
                key_batch, experience
            )

            def loss_fn(p):
                return self._model_loss(p, obs_seq, act_seq, rew_seq, term_seq, key_loss)

            (loss, (feats, mlogs)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                model.params
            )
            grads = jax.tree.map(
                lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), grads
            )
            model = model.apply_gradients(grads=grads)
            return (model, rng), {"loss/model": loss, **mlogs}

        (model, rng), mlogs = jax.lax.scan(
            model_step, (ts.model, ts.rng), None, length=hp.num_model_updates
        )
        mlogs = jax.tree.map(lambda x: jnp.mean(x), mlogs)

        # One fresh batch of features to seed the actor/critic imagination
        # rollouts from, reusing the just-updated model.
        rng, key_batch, key_observe = jax.random.split(rng, 3)
        obs_seq, act_seq, _, term_seq = self._sample_batch(key_batch, experience)
        _, _, feats, _, _, _ = self.world.apply(
            {"params": model.params},
            obs_seq,
            act_seq,
            term_seq,
            method=WorldModel.observe,
            rngs={"sample": key_observe},
        )
        start_feats = jax.lax.stop_gradient(
            feats[:, :-1].reshape(-1, feats.shape[-1])
        )

        # Return-normalization scale: EMA-tracked 5th/95th percentile of
        # a quick lambda-return estimate under the current critic, so the
        # actor's advantage is on a comparable scale regardless of the
        # environment's raw reward magnitude.
        probe_feats, probe_rews, probe_continues, _, _, _ = self._actor_critic_rollout(
            model.params, ts.actor.params, start_feats, rng
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
            return (actor, rng), {"loss/actor": loss, **alogs}

        (actor, rng), alogs = jax.lax.scan(
            actor_step, (ts.actor, rng), None, length=hp.num_actor_updates
        )
        alogs = jax.tree.map(lambda x: jnp.mean(x), alogs)

        def critic_step(carry, _):
            critic, rng = carry
            rng, key = jax.random.split(rng)

            def loss_fn(p):
                return self._critic_loss(
                    p, model.params, actor.params, ts.slow_critic_params, start_feats, key
                )

            (loss, clogs), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                critic.params
            )
            grads = jax.tree.map(
                lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), grads
            )
            critic = critic.apply_gradients(grads=grads)
            return (critic, rng), {"loss/critic": loss, **clogs}

        (critic, rng), clogs = jax.lax.scan(
            critic_step, (ts.critic, rng), None, length=hp.num_critic_updates
        )
        clogs = jax.tree.map(lambda x: jnp.mean(x), clogs)

        slow_critic_params = jax.tree.map(
            lambda slow, online: (1 - hp.slow_critic_rate) * slow
            + hp.slow_critic_rate * online,
            ts.slow_critic_params,
            critic.params,
        )

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
        logs["done_mask"] = experience.done
        logs["returns"] = experience.info["return"]
        logs["lengths"] = experience.t
        logs["iter/frames"] = ts.frames
        logs["iter/updates"] = ts.updates
        logs["iter/model_lr"] = self.hparams.model_lr
        logs["iter/actor_lr"] = self.hparams.actor_lr
        logs["iter/critic_lr"] = self.hparams.critic_lr
        logs["agent/return_norm_scale"] = return_norm_scale

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

    def _init_train_state(self, rng: Array) -> DreamerTrainState:
        hp = self.hparams
        obs_dim = self.world.obs_dim
        assert obs_dim == int(np.prod(self.env.observation_space.shape)), (
            "self.world's obs_dim must match the (flattened) observation "
            f"space of self.env - got obs_dim={obs_dim}, but "
            f"env.observation_space.shape={self.env.observation_space.shape} "
            f"flattens to {int(np.prod(self.env.observation_space.shape))}. "
            "Construct WorldModel(obs_dim=int(np.prod(env.observation_space."
            "shape)), act_dim=len(env.action_set), hparams=...) explicitly, "
            "the same way PPO expects ActorCritic(action_dim=len(env."
            "action_set)) pre-constructed rather than building it internally."
        )

        rng, wk1, wk2, ak, ck = jax.random.split(rng, 5)
        w_variables = self.world.init(
            {"params": wk1, "sample": wk2},
            jnp.zeros((1, obs_dim)),
            jnp.zeros((1, self.world.act_dim)),
            method=WorldModel.init_probe,
        )
        a_variables = self.actor.init(ak, jnp.zeros((1, hp.deter_size + hp.stoch_flat)))
        c_variables = self.critic.init(ck, jnp.zeros((1, hp.deter_size + hp.stoch_flat)))

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
            apply_fn=self.world.apply, params=w_variables["params"], tx=model_tx
        )
        actor_state = TrainState.create(
            apply_fn=self.actor.apply, params=a_variables["params"], tx=actor_tx
        )
        critic_state = TrainState.create(
            apply_fn=self.critic.apply, params=c_variables["params"], tx=critic_tx
        )

        rng, rs = jax.random.split(rng)
        reset_rng = jax.random.split(rs, hp.num_envs)
        env_state = jax.vmap(self.env.reset)(reset_rng)
        h0, z0, a0 = self.world.apply(
            w_variables, hp.num_envs, method=WorldModel.init_state
        )

        return DreamerTrainState(
            model=model_state,
            actor=actor_state,
            critic=critic_state,
            slow_critic_params=c_variables["params"],
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

    def train_first_update(self, rng: Array) -> Tuple[DreamerTrainState, Dict]:
        """Runs initialisation plus exactly one `update()` call. Mainly
        useful for tests/debugging, where running a full `train()` (sized
        by `hparams.budget`) is either unnecessary or awkward to size
        exactly."""
        ts = self._init_train_state(rng)
        return self.update(ts, None)

    def train(self, rng: Array) -> Tuple[DreamerTrainState, Dict]:
        hp = self.hparams
        ts = self._init_train_state(rng)
        num_updates = hp.budget // (hp.num_steps * hp.num_envs)

        start_time = time.time()
        ts, logs = jax.lax.scan(self.update, ts, None, length=num_updates)
        elapsed = time.time() - start_time
        logs["iter/fps"] = jnp.asarray([ts.frames / elapsed] * num_updates)
        logs["iter/wall_time"] = jnp.asarray([elapsed] * num_updates)

        return ts, logs
