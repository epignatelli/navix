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

# A lightweight, MLP-based Dreamer-style world-model agent (RSSM + actor +
# critic trained on imagined rollouts). This is a simplified reproduction
# inspired by Dreamer/DreamerV2/DreamerV3 (Hafner et al.), not a faithful
# port of any single one of those papers - the world model uses a unit
# -variance Gaussian decoder and a single free-bits KL scalar rather than
# per-dimension free bits, categorical latents, symlog transforms, or
# their two-hot reward/value heads. Originally written for a NeurIPS
# rebuttal on the `neurips` research branch; ported here as a first-class
# `navix.agents` module per issue #142, with three correctness fixes made
# during the port (see below).

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
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from flax import struct

from .agent import Agent, HParams
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

    # Loss scales
    kl_scale: float = 1.0
    """Weight of the KL term in the world-model loss."""
    free_kl: float = 1.0
    """Free-bits floor on the (batch-mean) KL term - the model isn't
    penalised for KL below this value."""

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
    stoch_size: int = 30
    """Size of the RSSM's stochastic latent."""
    hidden_size: int = 200
    """Hidden layer size used throughout the model/actor/critic MLPs."""


# -------------------------
# World Model (lightweight Dreamer-style RSSM)
# -------------------------


def _diag_gaussian(loc: Array, scale: Array) -> distrax.MultivariateNormalDiag:
    return distrax.MultivariateNormalDiag(loc=loc, scale_diag=scale)


def _softplus_std(x: Array, min_std: float = 0.1) -> Array:
    return nn.softplus(x) + min_std


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
        return net(obs)


class Decoder(nn.Module):
    hidden_size: int
    obs_dim: int

    @nn.compact
    def __call__(self, feat: Array) -> distrax.MultivariateNormalDiag:
        net = nn.Sequential(
            [
                nn.Dense(self.hidden_size),
                nn.elu,
                nn.Dense(self.hidden_size),
                nn.elu,
                nn.Dense(self.obs_dim),
            ]
        )
        mean = net(feat)
        scale = jnp.ones_like(mean)  # unit-variance decoder (simple & stable)
        return _diag_gaussian(mean, scale)


class RewardHead(nn.Module):
    hidden_size: int

    @nn.compact
    def __call__(self, feat: Array) -> distrax.Normal:
        net = nn.Sequential(
            [
                nn.Dense(self.hidden_size),
                nn.elu,
                nn.Dense(self.hidden_size),
                nn.elu,
                nn.Dense(1),
            ]
        )
        mean = jnp.squeeze(net(feat), -1)
        scale = jnp.ones_like(mean)
        return distrax.Normal(loc=mean, scale=scale)


class TermHead(nn.Module):
    hidden_size: int

    @nn.compact
    def __call__(self, feat: Array) -> distrax.Bernoulli:
        net = nn.Sequential(
            [
                nn.Dense(self.hidden_size),
                nn.elu,
                nn.Dense(self.hidden_size),
                nn.elu,
                nn.Dense(1),
            ]
        )
        logits = jnp.squeeze(net(feat), -1)
        return distrax.Bernoulli(logits=logits)


class PriorNet(nn.Module):
    hidden_size: int
    stoch_size: int

    @nn.compact
    def __call__(self, h: Array) -> distrax.MultivariateNormalDiag:
        x = nn.elu(nn.Dense(self.hidden_size)(h))
        loc = nn.Dense(self.stoch_size)(x)
        pre = nn.Dense(self.stoch_size)(x)
        scale = _softplus_std(pre)
        return _diag_gaussian(loc, scale)


class PostNet(nn.Module):
    hidden_size: int
    stoch_size: int

    @nn.compact
    def __call__(self, h: Array, embed: Array) -> distrax.MultivariateNormalDiag:
        x = jnp.concatenate([h, embed], axis=-1)
        x = nn.elu(nn.Dense(self.hidden_size)(x))
        loc = nn.Dense(self.stoch_size)(x)
        pre = nn.Dense(self.stoch_size)(x)
        scale = _softplus_std(pre)
        return _diag_gaussian(loc, scale)


class RSSM(nn.Module):
    deter_size: int

    @nn.compact
    def __call__(self, h_prev: Array, z_prev: Array, a_prev_oh: Array) -> Array:
        x = jnp.concatenate([z_prev, a_prev_oh], axis=-1)
        gru = nn.GRUCell(features=self.deter_size)
        h, _ = gru(h_prev, x)  # GRUCell returns (new_carry, y); y == new_carry
        return h


class WorldModel(nn.Module):
    obs_dim: int
    act_dim: int
    hparams: DreamerHparams

    def setup(self):
        hp = self.hparams
        self.encoder = Encoder(hp.hidden_size, hp.embed_size)
        self.rssm = RSSM(hp.deter_size)
        self.prior = PriorNet(hp.hidden_size, hp.stoch_size)
        self.post = PostNet(hp.hidden_size, hp.stoch_size)
        self.decoder = Decoder(hp.hidden_size, self.obs_dim)
        self.reward = RewardHead(hp.hidden_size)
        self.term = TermHead(hp.hidden_size)

    def init_state(self, batch_size: int) -> Tuple[Array, Array, Array]:
        h = jnp.zeros((batch_size, self.hparams.deter_size))
        z = jnp.zeros((batch_size, self.hparams.stoch_size))
        a0 = jnp.zeros((batch_size, self.act_dim))
        return h, z, a0

    def feat(self, h: Array, z: Array) -> Array:
        return jnp.concatenate([h, z], axis=-1)

    def observe(self, obs_seq: Array, act_seq: Array) -> Tuple[
        Tuple[Array, Array],
        Array,
        Tuple[distrax.Distribution, distrax.Distribution, distrax.Distribution],
        Array,
    ]:
        """Runs the RSSM over an observed sequence, computing the posterior
        latent at every step and decoding obs/reward/termination from it.

        Args:
            obs_seq (Array): `f32[B, L+1, obs_dim]`, flattened observations.
            act_seq (Array): `i32[B, L]`, actions taken between consecutive
                observations.

        Returns:
            `((h_seq, z_seq), kls, (obs_dist, rew_dist, term_dist), feats)`,
            all aligned with `obs_seq[:, 1:]` (i.e. `L` steps, `t=0..L-1`).
            `kls` is `f32[B, L]`, the per-step `KL(post || prior)` - computed
            inside the scan body rather than returned as `(priors, posts)`
            distribution objects, because jax.lax.scan only restacks a
            distrax.Distribution's array *leaves* across the new leading
            (L) axis; its batch_shape/event_shape (computed once from the
            un-stacked per-step arrays) go stale, so calling
            post.kl_divergence(prior) *after* the scan raises a shape
            mismatch deep inside distrax/tfp. Computing it per-step, while
            the distributions' shapes are still consistent with their own
            metadata, sidesteps that entirely."""
        B, Lp1, _ = obs_seq.shape
        L = Lp1 - 1

        embed_all = jax.vmap(self.encoder)(obs_seq.reshape(B * (L + 1), -1)).reshape(
            B, L + 1, -1
        )
        a_oh = jax.nn.one_hot(act_seq, self.act_dim)
        a_prev = jnp.concatenate(
            [jnp.zeros((B, 1, self.act_dim)), a_oh[:, :-1]], axis=1
        )

        # Plain jax.lax.scan, not nn.scan: `step` closes over `self`
        # implicitly (via self.rssm/self.prior/self.post) rather than
        # taking it as an explicit first argument, which is what Flax's
        # lifted nn.scan requires when called as `nn.scan(step, ...)(carry,
        # xs)` - passing a plain closure that way makes Flax's transform
        # machinery treat `carry` itself as the (missing) `self` argument,
        # failing deep inside with a confusing `'tuple' object has no
        # attribute '_state'`. `imagine()` below already uses plain
        # jax.lax.scan correctly for the same reason; this mirrors it.
        # Since setup() already built every submodule once, calling them
        # repeatedly inside a plain scan just reuses their closed-over
        # params - no Flax-level parameter broadcasting is needed here.
        # The per-step posterior sample rng is threaded through the scan
        # carry explicitly instead of nn.scan's split_rngs, since plain
        # jax.lax.scan doesn't auto-split Flax's rng streams.
        def step(carry, inputs):
            h_prev, z_prev, rng = carry
            a_prev_t, embed_t = inputs
            rng, key = jax.random.split(rng)
            h_t = self.rssm(h_prev, z_prev, a_prev_t)
            prior_t = self.prior(h_t)
            post_t = self.post(h_t, embed_t)
            kl_t = post_t.kl_divergence(prior_t)  # (B,) - see observe()'s docstring
            z_t = post_t.sample(seed=key)
            feat_t = self.feat(h_t, z_t)
            return (h_t, z_t, rng), (h_t, z_t, kl_t, feat_t)

        # jax.lax.scan always scans over axis 0, but our sequences are
        # batch-major (B, L, ...) - swap to (L, B, ...) going in, and swap
        # the stacked outputs back to (B, L, ...) coming out, since
        # downstream code (feats.reshape(B * L, -1) below,
        # start_feats = feats[:, :-1]... in update()) expects batch-major.
        inputs = jax.tree.map(
            lambda x: jnp.swapaxes(x, 0, 1), (a_prev[:, :L], embed_all[:, 1:])
        )
        hp = self.hparams
        init_rng = self.make_rng("sample")
        (h_last, z_last, _), (hs, zs, kls, feats) = jax.lax.scan(
            step,
            (jnp.zeros((B, hp.deter_size)), jnp.zeros((B, hp.stoch_size)), init_rng),
            inputs,
            length=L,
        )
        hs, zs, kls, feats = jax.tree.map(
            lambda x: jnp.swapaxes(x, 0, 1), (hs, zs, kls, feats)
        )

        feats_flat = feats.reshape(B * L, -1)
        obs_dist = self.decoder(feats_flat)
        rew_dist = self.reward(feats_flat)
        term_dist = self.term(feats_flat)

        return (hs, zs), kls, (obs_dist, rew_dist, term_dist), feats

    def posterior_step(
        self, h: Array, z: Array, a_prev_oh: Array, obs: Array, rng: Array
    ) -> Tuple[Array, Array]:
        """One environment-collection step: advances the RSSM's belief with
        the previous action, then updates it to the posterior given the new
        observation. Encoder/RSSM/posterior are submodules bound to this
        WorldModel instance, so - unlike a bare Dense/Sequential module -
        they can only be called from inside a WorldModel.apply() trace,
        which is what this method (called via `self.world.apply(...,
        method=WorldModel.posterior_step)`) provides for `_collect`."""
        h_next = self.rssm(h, z, a_prev_oh)
        embed = self.encoder(obs)
        post = self.post(h_next, embed)
        z_next = post.sample(seed=rng)
        return h_next, z_next

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
        h = jnp.zeros((obs.shape[0], self.hparams.deter_size))
        z = jnp.zeros((obs.shape[0], self.hparams.stoch_size))
        h_t = self.rssm(h, z, a_prev_oh)
        self.prior(h_t)
        embed = self.encoder(obs)
        post_t = self.post(h_t, embed)
        feat_t = self.feat(h_t, post_t.mean())
        self.decoder(feat_t)
        self.reward(feat_t)
        self.term(feat_t)

    def imagine(
        self,
        start_h: Array,
        start_z: Array,
        actor_logits_fn: Callable[[Array], Array],
        horizon: int,
        rng: Array,
    ) -> Tuple[Array, Array, Array]:
        """Rolls the RSSM forward purely through its own prior (no real
        observations), sampling actions from `actor_logits_fn` at each
        step - the "imagined" trajectories the actor and critic train on.

        Returns `(feats, rewards, terms)`, all `(B, H, ...)` (batch-major).
        `rewards`/`terms` are plain arrays (each step's reward/term
        distribution mean), not distrax.Distribution objects - as in
        observe(), a distrax.Distribution's batch_shape/event_shape are
        computed once from its un-stacked per-step arrays, and go stale
        under jax.lax.scan's leaf-only restacking, so any post-scan method
        call on it (including .mean()) is unsafe. Reducing to a mean
        before accumulating sidesteps that."""

        def rollout_step(carry, _):
            h, z, rng = carry
            feat = jnp.concatenate([h, z], axis=-1)
            logits = actor_logits_fn(feat)
            rng, key = jax.random.split(rng)
            a = distrax.Categorical(logits=logits).sample(seed=key)
            a_oh = jax.nn.one_hot(a, logits.shape[-1])
            h_next = self.rssm(h, z, a_oh)
            prior = self.prior(h_next)
            rng, key = jax.random.split(rng)
            z_next = prior.sample(seed=key)
            feat_next = jnp.concatenate([h_next, z_next], axis=-1)
            rew_mean = self.reward(feat_next).mean()
            term_mean = self.term(feat_next).mean()
            return (h_next, z_next, rng), (feat_next, rew_mean, term_mean)

        (hT, zT, _), (feats, rews, terms) = jax.lax.scan(
            rollout_step, (start_h, start_z, rng), None, length=horizon
        )
        # (H, B, ...) -> (B, H, ...)
        feats, rews, terms = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), (feats, rews, terms))
        return feats, rews, terms


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
                nn.Dense(
                    self.act_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
                ),
            ]
        )
        return net(feat)

    def dist(self, feat: Array) -> distrax.Categorical:
        return distrax.Categorical(logits=self(feat))


class Critic(nn.Module):
    hidden: int = 200

    @nn.compact
    def __call__(self, feat: Array) -> Array:
        net = nn.Sequential(
            [
                nn.Dense(self.hidden),
                nn.elu,
                nn.Dense(self.hidden),
                nn.elu,
                nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0)),
            ]
        )
        return jnp.squeeze(net(feat), -1)


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
    """Unlike `flax.training.train_state.TrainState` (used by `PPO`), this
    wraps three *independent* `TrainState`s, one per network - the world
    model, actor, and critic each have their own optimizer, learning rate
    and step counter, and are updated via their own `apply_gradients()`.

    An earlier draft of this agent instead subclassed `TrainState` directly
    and hand-rolled `tx.update()` + `optax.apply_updates()` for all three
    networks while sharing a single `tx`/`step` field meant to be swapped
    between updates - the swap never actually took effect before each
    network's update ran, so actor and critic gradients were silently
    applied through the model's optimizer (and learning rate) instead of
    their own, and since `apply_gradients()` was never called, the step
    counter never incremented at all. Three separate `TrainState`s make
    both bugs structurally impossible instead of relying on remembering to
    swap a shared field correctly."""

    model: TrainState
    actor: TrainState
    critic: TrainState

    env_state: Timestep
    rng: Array
    frames: Array
    updates: Array

    # Latents per-env, carried across collection steps.
    h: Array  # (N, deter_size)
    z: Array  # (N, stoch_size)
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
    def _discount_weights(gamma: float, length: int) -> Array:
        if length <= 0:
            return jnp.ones((0,), dtype=jnp.float32)
        d = jnp.cumprod(jnp.full((length - 1,), gamma, dtype=jnp.float32))
        return jnp.concatenate([jnp.ones((1,), dtype=jnp.float32), d], axis=0)

    @staticmethod
    def _compute_lambda_values(
        next_values: Array,
        rewards: Array,
        terminals: Array,
        discount: float,
        lam: float,
    ) -> Array:
        # shapes: (B, H), (B, H), (B, H) - batch-major, but jax.lax.scan
        # only ever scans axis 0, so time (H, the axis this function
        # actually recurs over) needs to be axis 0 for the scan itself;
        # swap in and back out around it.
        def scan_fn(carry, inputs):
            v_lambda_next = carry
            r_t, term_t, v_tp1 = inputs
            td = r_t + (1.0 - term_t) * (1.0 - lam) * discount * v_tp1
            v_lambda = td + v_lambda_next * lam * discount
            return v_lambda, v_lambda

        init = jnp.zeros_like(next_values[:, -1])
        rewards_t, terminals_t, next_values_t = jax.tree.map(
            lambda x: jnp.swapaxes(x, 0, 1)[::-1], (rewards, terminals, next_values)
        )
        _, vals_t = jax.lax.scan(
            scan_fn,
            init,
            (rewards_t, terminals_t, next_values_t),
            length=next_values.shape[1],
        )
        return jnp.swapaxes(vals_t[::-1], 0, 1)

    # ---------- Collection ----------

    def _collect(
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
            a = distrax.Categorical(logits=logits).sample(seed=key_a)
            log_prob = distrax.Categorical(logits=logits).log_prob(a)

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
        (hs, zs), kls, (obs_dist, rew_dist, term_dist), feats = self.world.apply(
            {"params": params},
            obs_seq,
            act_seq,
            method=WorldModel.observe,
            rngs={"sample": rng},
        )
        B, L = act_seq.shape
        kl = jnp.mean(kls)
        kl = jnp.maximum(kl, self.hparams.free_kl)
        obs_ll = jnp.mean(obs_dist.log_prob(obs_seq[:, 1:].reshape(B * L, -1)))
        rew_ll = jnp.mean(rew_dist.log_prob(rew_seq.reshape(-1)))
        term_ll = jnp.mean(term_dist.log_prob(term_seq.reshape(-1)))
        loss = self.hparams.kl_scale * kl - obs_ll - rew_ll - term_ll
        logs = {
            "agent/model/kl": kl,
            "agent/model/log_p_obs": -obs_ll,
            "agent/model/log_p_rew": -rew_ll,
            "agent/model/log_p_term": -term_ll,
        }
        return loss, (feats, logs)

    def _actor_loss(self, actor_params, model_params, critic_params, start_feats, rng):
        hp = self.hparams
        h0, z0 = start_feats[:, : hp.deter_size], start_feats[:, hp.deter_size :]

        def actor_logits_fn(feat):
            return self.actor.apply({"params": actor_params}, feat)

        feats, rews, terms = self.world.apply(
            {"params": model_params},
            h0,
            z0,
            actor_logits_fn,
            hp.imag_horizon,
            rng,
            method=WorldModel.imagine,
        )
        # vals_tp1 = V(feats[1:]) bootstraps each reward with the *next*
        # imagined state's value, so it only covers H-1 pairs (there's no
        # feats[H] within this rollout to bootstrap the last reward with)
        # - truncate rews/terms to match rather than the full H-length
        # rollout.
        vals_tp1 = jax.vmap(self.critic.apply, in_axes=(None, 0))(
            {"params": critic_params}, feats[:, 1:]
        )
        lam_vals = self._compute_lambda_values(
            vals_tp1, rews[:, :-1], terms[:, :-1], hp.discount, hp.lam
        )
        disc = self._discount_weights(hp.discount, hp.imag_horizon - 1)
        loss = -(lam_vals * disc).mean()
        entropy = (
            distrax.Categorical(
                logits=self.actor.apply({"params": actor_params}, start_feats)
            )
            .entropy()
            .mean()
        )
        logs = {"agent/actor/loss": loss, "agent/actor/entropy": entropy}
        return loss, logs

    def _critic_loss(self, critic_params, model_params, actor_params, start_feats, rng):
        hp = self.hparams
        h0, z0 = start_feats[:, : hp.deter_size], start_feats[:, hp.deter_size :]

        def actor_logits_fn(feat):
            return self.actor.apply({"params": actor_params}, feat)

        feats, rews, terms = self.world.apply(
            {"params": model_params},
            h0,
            z0,
            actor_logits_fn,
            hp.imag_horizon,
            rng,
            method=WorldModel.imagine,
        )
        vals_tp1 = jax.vmap(self.critic.apply, in_axes=(None, 0))(
            {"params": critic_params}, feats[:, 1:]
        )
        targets = jax.lax.stop_gradient(
            self._compute_lambda_values(
                vals_tp1, rews[:, :-1], terms[:, :-1], hp.discount, hp.lam
            )
        )
        # preds must align with targets' H-1 length, hence feats[:, :-1]
        # (dropping the last imagined state, which targets has no
        # lambda-return for - see _actor_loss's vals_tp1 comment).
        preds = jax.vmap(self.critic.apply, in_axes=(None, 0))(
            {"params": critic_params}, feats[:, :-1]
        )
        disc = self._discount_weights(hp.discount, hp.imag_horizon - 1)
        loss = jnp.mean(((preds - targets) ** 2) * disc)
        return loss, {"agent/critic/loss": loss}

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
        obs_seq, act_seq, _, _ = self._sample_batch(key_batch, experience)
        _, _, _, feats = self.world.apply(
            {"params": model.params},
            obs_seq,
            act_seq,
            method=WorldModel.observe,
            rngs={"sample": key_observe},
        )
        start_feats = jax.lax.stop_gradient(
            feats[:, :-1].reshape(-1, feats.shape[-1])
        )

        def actor_step(carry, _):
            actor, rng = carry
            rng, key = jax.random.split(rng)

            def loss_fn(p):
                return self._actor_loss(p, model.params, ts.critic.params, start_feats, key)

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
                return self._critic_loss(p, model.params, actor.params, start_feats, key)

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

        ts = ts.replace(
            model=model,
            actor=actor,
            critic=critic,
            rng=rng,
            updates=ts.updates + 1,
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
        a_variables = self.actor.init(ak, jnp.zeros((1, hp.deter_size + hp.stoch_size)))
        c_variables = self.critic.init(ck, jnp.zeros((1, hp.deter_size + hp.stoch_size)))

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
            env_state=env_state,
            rng=rng,
            frames=jnp.asarray(0, dtype=jnp.int32),
            updates=jnp.asarray(0, dtype=jnp.int32),
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
