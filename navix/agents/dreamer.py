# dreamer_navix.py
# Dreamer agent adapted for NAVIX (JAX), fully-jittable training loop.
# Uses Flax/Linen modules, Distrax distributions, Optax optimizers.
# Assumes discrete action space (Categorical actor). Designed to mirror your PPO agent structure.

from __future__ import annotations
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, Tuple

import distrax
import jax
import jax.numpy as jnp
from jax import Array
import optax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from flax import struct

from navix.agents.agent import Agent, HParams
from navix.environments import Environment
from navix.environments.environment import Timestep
from navix.states import State

# -------------------------
# Hyperparameters
# -------------------------


class DreamerHparams(HParams):
    # Training schedule
    budget: int = struct.field(pytree_node=False, default=1_000_000)
    num_envs: int = struct.field(pytree_node=False, default=16)
    num_steps: int = struct.field(pytree_node=False, default=128)  # steps per update
    num_model_updates: int = struct.field(
        pytree_node=False, default=32
    )  # world model updates / update
    num_actor_updates: int = struct.field(pytree_node=False, default=32)
    num_critic_updates: int = struct.field(pytree_node=False, default=32)
    batch_size: int = struct.field(
        pytree_node=False, default=64
    )  # sequences per update step
    seq_len: int = struct.field(
        pytree_node=False, default=32
    )  # sequence length for training
    imag_horizon: int = struct.field(
        pytree_node=False, default=15
    )  # imagination rollout length
    prefill: int = struct.field(pytree_node=False, default=10_000)  # frames
    discount: float = 0.99
    lam: float = 0.95  # lambda-returns

    # Loss scales
    kl_scale: float = 1.0
    free_kl: float = 1.0

    # Opt/grad
    model_lr: float = 3e-4
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    max_grad_norm: float = 100.0

    # Model sizes
    embed_size: int = 128
    deter_size: int = 200  # GRU hidden
    stoch_size: int = 30  # latent z size
    hidden_size: int = 200


# -------------------------
# World Model (lightweight Dreamer-style RSSM)
# -------------------------


def _mlp(hidden: int, out: int, act: Callable = nn.elu):
    return nn.Sequential([nn.Dense(hidden), act, nn.Dense(hidden), act, nn.Dense(out)])


def _diag_gaussian(loc: Array, scale: Array):
    return distrax.MultivariateNormalDiag(loc=loc, scale_diag=scale)


def _softplus_std(x, min_std=0.1):
    return nn.softplus(x) + min_std


class Encoder(nn.Module):
    hidden_size: int
    embed_size: int

    @nn.compact
    def __call__(self, obs: Array) -> Array:
        # obs: (..., obs_dim) -> embed
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
        scale = jnp.ones_like(mean)  # unit variance decoder (simple & stable)
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
        gru = nn.GRUCell(features=self.deter_size)  # ← required
        h, _ = gru(h_prev, x)  # returns (new_carry, y) where y == new_carry
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
        a0 = jnp.zeros((batch_size, self.act_dim))  # one-hot(0)
        return h, z, a0

    def feat(self, h: Array, z: Array) -> Array:
        return jnp.concatenate([h, z], axis=-1)

    def observe(self, obs_seq: Array, act_seq: Array) -> Tuple[
        Tuple[Array, Array],  # (h_seq, z_seq)
        Tuple[distrax.Distribution, distrax.Distribution],  # (priors, posts)
        Tuple[
            distrax.Distribution,  # obs_dist (e.g., MultivariateNormalDiag)
            distrax.Distribution,  # rew_dist (Normal)
            distrax.Distribution,  # term_dist (Bernoulli)
        ],
        Array,  # feats (B, L, F)
    ]:
        """
        obs_seq: (B, L+1, obs_dim); act_seq: (B, L) int actions
        Returns sequences from t=0..L-1 that align with obs_seq[:,1:]
        """
        B, Lp1, _ = obs_seq.shape
        L = Lp1 - 1
        hp = self.hparams

        # Prepare inputs
        embed_all = jax.vmap(self.encoder)(obs_seq.reshape(B * (L + 1), -1)).reshape(
            B, L + 1, -1
        )
        # prev actions as one-hot, padded at start with zeros
        a_oh = jax.nn.one_hot(act_seq, self.act_dim)
        a_prev = jnp.concatenate(
            [jnp.zeros((B, 1, self.act_dim)), a_oh[:, :-1]], axis=1
        )

        def step(carry, inputs):
            h_prev, z_prev = carry
            a_prev_t, embed_t = inputs
            h_t = self.rssm(h_prev, z_prev, a_prev_t)
            prior_t = self.prior(h_t)
            post_t = self.post(h_t, embed_t)
            z_t = post_t.sample(seed=self.make_rng("sample"))
            feat_t = self.feat(h_t, z_t)
            return (h_t, z_t), (h_t, z_t, prior_t, post_t, feat_t)

        # scan over t = 0..L-1 using obs_seq[:,1:] and a_prev[:,0:L]
        inputs = (a_prev[:, :L], embed_all[:, 1:])
        (h_last, z_last), (hs, zs, priors, posts, feats) = nn.scan(
            step,
            variable_broadcast="params",
            split_rngs={"sample": True},
            length=L,
        )((jnp.zeros((B, hp.deter_size)), jnp.zeros((B, hp.stoch_size))), inputs)

        # decode / predict on feats
        feats_flat = feats.reshape(B * L, -1)
        obs_dist = self.decoder(feats_flat)
        rew_dist = self.reward(feats_flat)
        term_dist = self.term(feats_flat)

        return (hs, zs), (priors, posts), (obs_dist, rew_dist, term_dist), feats

    def imagine(
        self,
        start_h: Array,
        start_z: Array,
        actor_logits_fn: Callable[[Array], Array],
        horizon: int,
        rng: Array,
    ):
        """Generate imagined rollouts from (h,z), using prior and actor (Categorical)."""

        def rollout_step(carry, _):
            h, z, rng = carry
            feat = jnp.concatenate([h, z], axis=-1)
            logits = actor_logits_fn(feat)  # (B, act_dim)
            rng, key = jax.random.split(rng)
            a = distrax.Categorical(logits=logits).sample(seed=key)
            a_oh = jax.nn.one_hot(a, logits.shape[-1])
            h_next = self.rssm(h, z, a_oh)
            prior = self.prior(h_next)
            rng, key = jax.random.split(rng)
            z_next = prior.sample(seed=key)
            feat_next = jnp.concatenate([h_next, z_next], axis=-1)
            rew = self.reward(feat_next)
            term = self.term(feat_next)
            return (h_next, z_next, rng), (feat_next, rew, term)

        (hT, zT, _), (feats, rews, terms) = jax.lax.scan(
            rollout_step, (start_h, start_z, rng), None, length=horizon
        )
        # feats: (H, B, F) -> (B, H, F)
        feats = jnp.swapaxes(feats, 0, 1)
        return (
            feats,
            rews,
            terms,
        )  # rews/terms are distrax.Normal / distrax.Bernoulli per step (B,H)


# -------------------------
# Actor & Critic (Flax)
# -------------------------


class Actor(nn.Module):
    act_dim: int
    hidden: int = 200

    @nn.compact
    def __call__(self, feat: Array) -> Array:
        # Returns logits for Categorical policy
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
# Buffer compatible with PPO logging (for Experiment/Agent.log)
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
# Training State
# -------------------------


class DreamerTrainState(TrainState):
    # We won’t use TrainState.apply_fn for multiple nets; instead, keep params & opt state separately.
    env_state: Timestep
    rng: Array
    frames: Array
    epoch: Array

    # Latents per-env carried across collection
    h: Array  # (N, deter_size)
    z: Array  # (N, stoch_size)
    a_prev_oh: Array  # (N, act_dim)

    # Parameters and opt-states for each component
    model_params: nn.FrozenDict
    model_opt_state: optax.OptState
    actor_params: nn.FrozenDict
    actor_opt_state: optax.OptState
    critic_params: nn.FrozenDict
    critic_opt_state: optax.OptState


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
        # shapes: (B, H), (B, H), (B, H)
        def scan_fn(carry, inputs):
            v_lambda_next = carry
            r_t, term_t, v_tp1 = inputs
            td = r_t + (1.0 - term_t) * (1.0 - lam) * discount * v_tp1
            v_lambda = td + v_lambda_next * lam * discount
            return v_lambda, v_lambda

        # reverse scan
        init = jnp.zeros_like(next_values[:, -1])
        _, vals = jax.lax.scan(
            scan_fn,
            init,
            (rewards[:, ::-1], terminals[:, ::-1], next_values[:, ::-1]),
            length=next_values.shape[1],
        )
        return vals[:, ::-1]

    # ---------- Collection ----------

    def _collect(
        self, ts: DreamerTrainState
    ) -> Tuple[DreamerTrainState, Buffer, Tuple[Array, Array, Array]]:
        """Run self.hparams.num_steps in parallel envs; maintain and return per-env latent (h,z,a_prev_oh)."""
        hp = self.hparams

        def _env_step(carry, _):
            env_state, rng, h, z, a_prev_oh = carry

            # Inference for current observation -> posterior state
            # Policy uses features from posterior at time t
            feat = jnp.concatenate([h, z], axis=-1)
            logits = jax.vmap(self.actor)(feat)  # (N, act_dim)
            rng, key_a = jax.random.split(rng)
            a = distrax.Categorical(logits=logits).sample(seed=key_a)  # (N,)
            log_prob = distrax.Categorical(logits=logits).log_prob(a)

            # step env
            new_env_state = jax.vmap(self.env.step, in_axes=(0, 0))(env_state, a)

            # Update posterior using new observation (o_{t+1}) and previous action a_t
            # Build one-hot for the action we just took
            a_oh = jax.nn.one_hot(a, logits.shape[-1])
            # Encode new obs and update RSSM/posterior for next step
            embed = jax.vmap(self.world.encoder)(new_env_state.observation)
            h_next = jax.vmap(self.world.rssm)(h, z, a_oh)
            post = jax.vmap(self.world.post)(h_next, embed)
            rng, key_z = jax.random.split(rng)
            z_next = post.sample(seed=key_z)

            # Reset latents where env is done
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
            carry_next = (new_env_state, rng, h_next, z_next, a_oh_next)
            return carry_next, transition

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
        return ts, traj, (h, z, a_prev_oh)

    # ---------- Sampling sequences from the latest rollout ----------

    def _sample_batch(
        self, rng: Array, experience: Buffer
    ) -> Tuple[Array, Array, Array, Array]:
        """
        Sample B sequences of length L from the most recent rollout.
        Returns:
          obs_seq: (B, L+1, obs_dim)
          act_seq: (B, L) int
          rew_seq: (B, L)
          term_seq:(B, L)
        """
        hp = self.hparams
        T, N = experience.obs.shape[0], experience.obs.shape[1]
        L = hp.seq_len
        max_start = T - (L + 1)
        max_start = jnp.maximum(max_start, 0)

        # Sample (env_idx, start_idx) pairs
        rng, key1, key2 = jax.random.split(rng, 3)
        env_idx = jax.random.randint(key1, (hp.batch_size,), minval=0, maxval=N)
        start_idx = jax.random.randint(
            key2, (hp.batch_size,), minval=0, maxval=jnp.maximum(1, max_start + 1)
        )

        def take_seq(ei, si):
            obs_seq = experience.obs[si : si + L + 1, ei]
            act_seq = experience.action[si : si + L, ei]
            rew_seq = experience.reward[si : si + L, ei]
            term_seq = experience.done[si : si + L, ei].astype(jnp.float32)
            return obs_seq, act_seq, rew_seq, term_seq

        obs_seq, act_seq, rew_seq, term_seq = jax.vmap(take_seq)(env_idx, start_idx)
        return obs_seq, act_seq, rew_seq, term_seq

    # ---------- Losses & Updates ----------

    def _model_loss(self, params, obs_seq, act_seq, rew_seq, term_seq):
        # Forward through observe() with current params
        (hs, zs), (priors, posts), (obs_dist, rew_dist, term_dist), feats = (
            self.world.apply(
                {"params": params},
                obs_seq,
                act_seq,
                method=WorldModel.observe,
                rngs={"sample": jax.random.PRNGKey(0)},
            )
        )
        B, L = act_seq.shape
        # KL (posterior || prior)
        kl = jnp.mean(distrax.kl_divergence(posts, priors))
        kl = jnp.maximum(kl, self.hparams.free_kl)
        # Observation likelihood (reconstruct obs_seq[:,1:])
        obs_ll = jnp.mean(obs_dist.log_prob(obs_seq[:, 1:].reshape(B * L, -1)))
        # Reward likelihood
        rew_ll = jnp.mean(rew_dist.log_prob(rew_seq.reshape(-1)))
        # Terminal likelihood (Bernoulli)
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
        # start_feats: (B0, F) -> split to h,z
        hp = self.hparams
        F = start_feats.shape[-1]
        h_dim = hp.deter_size
        h0, z0 = start_feats[:, :h_dim], start_feats[:, h_dim:]

        def actor_logits_fn(feat):
            logits = self.actor.apply({"params": actor_params}, feat)
            return logits

        feats, rew_dists, term_dists = self.world.apply(
            {"params": model_params},
            h0,
            z0,
            partial(actor_logits_fn),
            hp.imag_horizon,
            rng,
            method=WorldModel.imagine,
        )
        # Values for steps 1..H
        vals_tp1 = jax.vmap(self.critic.apply, in_axes=(None, 0))(
            {"params": critic_params}, feats[:, 1:]
        )
        # Rewards/terminals for steps 0..H-1
        rews = rew_dists.mean()  # (B, H)
        terms = term_dists.mean()  # (B, H)
        lam_vals = self._compute_lambda_values(
            vals_tp1, rews, terms, hp.discount, hp.lam
        )
        disc = self._discount_weights(hp.discount, hp.imag_horizon)  # (H,)
        loss = -(lam_vals * disc).mean()
        entropy = (
            distrax.Categorical(
                logits=self.actor.apply({"params": actor_params}, start_feats)
            )
            .entropy()
            .mean()
        )
        logs = {
            "agent/actor/loss": loss,
            "agent/actor/entropy": entropy,
        }
        return loss, logs

    def _critic_loss(self, critic_params, model_params, actor_params, start_feats, rng):
        hp = self.hparams
        F = start_feats.shape[-1]
        h_dim = hp.deter_size
        h0, z0 = start_feats[:, :h_dim], start_feats[:, h_dim:]

        def actor_logits_fn(feat):
            return self.actor.apply({"params": actor_params}, feat)

        feats, rew_dists, term_dists = self.world.apply(
            {"params": model_params},
            h0,
            z0,
            partial(actor_logits_fn),
            hp.imag_horizon,
            rng,
            method=WorldModel.imagine,
        )
        vals_tp1 = jax.vmap(self.critic.apply, in_axes=(None, 0))(
            {"params": critic_params}, feats[:, 1:]
        )
        rews = rew_dists.mean()
        terms = term_dists.mean()
        targets = self._compute_lambda_values(
            vals_tp1, rews, terms, hp.discount, hp.lam
        )
        preds = jax.vmap(self.critic.apply, in_axes=(None, 0))(
            {"params": critic_params}, feats[:, :-1]
        )  # align with targets length H
        disc = self._discount_weights(hp.discount, hp.imag_horizon)
        loss = jnp.mean(((preds - targets) ** 2) * disc)
        logs = {
            "agent/critic/loss": loss,
        }
        return loss, logs

    # ---------- One update (collect + model/actor/critic) ----------

    def update(self, ts: DreamerTrainState, _) -> Tuple[DreamerTrainState, Dict]:
        hp = self.hparams

        # 1) Collect
        ts, experience, _ = self._collect(ts)

        # 2) Model updates (scan)
        def model_step(carry, _):
            ts, rng = carry
            rng, key = jax.random.split(rng)
            obs_seq, act_seq, rew_seq, term_seq = self._sample_batch(key, experience)

            def loss_fn(p):
                return self._model_loss(p, obs_seq, act_seq, rew_seq, term_seq)

            (loss, (feats, mlogs)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                ts.model_params
            )
            grads = jax.tree.map(
                lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), grads
            )
            updates, new_opt = ts.tx.update(
                grads, ts.model_opt_state, ts.model_params
            )  # tx from TrainState
            new_params = optax.apply_updates(ts.model_params, updates)
            ts = ts.replace(model_params=new_params, model_opt_state=new_opt)
            return (ts, rng), {"loss/model": loss, **mlogs}

        (ts, _), mlogs = jax.lax.scan(
            model_step, (ts, ts.rng), None, length=hp.num_model_updates
        )
        mlogs = jax.tree.map(lambda x: jnp.mean(x), mlogs)

        # Extract features from one fresh batch for actor/critic (to avoid re-running observe repeatedly)
        key = ts.rng
        key, k1 = jax.random.split(key)
        obs_seq, act_seq, _, _ = self._sample_batch(k1, experience)
        (hs, zs), _, _, feats = self.world.apply(
            {"params": ts.model_params},
            obs_seq,
            act_seq,
            method=WorldModel.observe,
            rngs={"sample": jax.random.PRNGKey(0)},
        )
        # Start features are feats[:, :-1] (B,L-1,F) -> flatten
        start_feats = feats[:, :-1].reshape(-1, feats.shape[-1])

        # 3) Actor updates
        def actor_step(carry, _):
            ts, rng = carry
            rng, key = jax.random.split(rng)

            def loss_fn(p):
                return self._actor_loss(
                    p, ts.model_params, ts.critic_params, start_feats, key
                )

            (loss, alogs), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                ts.actor_params
            )
            grads = jax.tree.map(
                lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), grads
            )
            updates, new_opt = ts.tx.update(grads, ts.actor_opt_state, ts.actor_params)
            new_params = optax.apply_updates(ts.actor_params, updates)
            ts = ts.replace(actor_params=new_params, actor_opt_state=new_opt)
            return (ts, rng), {"loss/actor": loss, **alogs}

        (ts, _), alogs = jax.lax.scan(
            actor_step, (ts, key), None, length=hp.num_actor_updates
        )
        alogs = jax.tree.map(lambda x: jnp.mean(x), alogs)

        # 4) Critic updates
        def critic_step(carry, _):
            ts, rng = carry
            rng, key = jax.random.split(rng)

            def loss_fn(p):
                return self._critic_loss(
                    p, ts.model_params, ts.actor_params, start_feats, key
                )

            (loss, clogs), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                ts.critic_params
            )
            grads = jax.tree.map(
                lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), grads
            )
            updates, new_opt = ts.tx.update(
                grads, ts.critic_opt_state, ts.critic_params
            )
            new_params = optax.apply_updates(ts.critic_params, updates)
            ts = ts.replace(critic_params=new_params, critic_opt_state=new_opt)
            return (ts, rng), {"loss/critic": loss, **clogs}

        (ts, rng), clogs = jax.lax.scan(
            critic_step, (ts, key), None, length=hp.num_critic_updates
        )
        clogs = jax.tree.map(lambda x: jnp.mean(x), clogs)

        # finalize state
        ts = ts.replace(rng=rng, epoch=ts.epoch + 1)

        # logs compatible with Agent.log
        logs = {}
        logs.update(mlogs)
        logs.update(alogs)
        logs.update(clogs)
        logs["done_mask"] = experience.done
        logs["returns"] = experience.info["return"]
        logs["lengths"] = experience.t
        logs["iter/frames"] = ts.frames
        logs["iter/epochs"] = ts.epoch
        logs["iter/updates"] = ts.step

        if self.hparams.log_render:
            # Render one env from the collected rollout (T, N)
            b = jax.random.randint(ts.rng, (), 0, self.hparams.num_envs)
            from navix.observations import rgb

            logs["render/human"] = jax.vmap(rgb)(
                jax.tree.map(lambda x: x[:, b], experience.state)
            ).transpose((0, 3, 1, 2))

        # Optional debug callback
        if self.hparams.debug:
            jax.debug.callback(self.log, logs, experience)

        return ts, logs

    # ---------- Train entry point ----------

    def train(self, rng: Array) -> Tuple[DreamerTrainState, Dict]:
        hp = self.hparams
        # Dummy observation to init
        rng, key_init = jax.random.split(rng)
        init_obs = self.env.observation_space.sample(key_init)
        obs_dim = init_obs.shape[-1]
        act_dim = len(self.env.action_set)

        # Instantiate modules
        world = WorldModel(obs_dim=obs_dim, act_dim=act_dim, hparams=hp)
        actor = self.actor
        critic = self.critic

        # Init params
        rng, wk, ak, ck = jax.random.split(rng, 4)
        w_params = world.init(
            wk,
            jnp.zeros((1, hp.seq_len + 1, obs_dim)),
            jnp.zeros((1, hp.seq_len), dtype=jnp.int32),
            method=WorldModel.observe,
        )
        a_params = actor.init(
            ak, jnp.zeros((1, hp.deter_size + hp.stoch_size))
        )  # logits(feat)
        c_params = critic.init(
            ck, jnp.zeros((1, hp.deter_size + hp.stoch_size))
        )  # value(feat)

        # Optax
        model_tx = optax.chain(
            optax.clip_by_global_norm(hp.max_grad_norm), optax.adam(hp.model_lr)
        )
        actor_tx = optax.chain(
            optax.clip_by_global_norm(hp.max_grad_norm), optax.adam(hp.actor_lr)
        )
        critic_tx = optax.chain(
            optax.clip_by_global_norm(hp.max_grad_norm), optax.adam(hp.critic_lr)
        )

        # Init opt states
        m_opt_state = model_tx.init(w_params["params"])
        a_opt_state = actor_tx.init(a_params["params"])
        c_opt_state = critic_tx.init(c_params["params"])

        # Init Envs
        rng, rs = jax.random.split(rng)
        reset_seeds = jax.random.split(rs, hp.num_envs)
        env_state = jax.vmap(self.env.reset)(reset_seeds)

        # Init latents per env
        h0, z0, a0 = world.apply(w_params, hp.num_envs, method=WorldModel.init_state)

        # Compose training state
        num_updates = hp.budget // (hp.num_steps * hp.num_envs)
        ts = DreamerTrainState.create(
            apply_fn=lambda p, x: x,  # unused by this agent
            params=w_params["params"],  # keep something to satisfy TrainState API
            tx=model_tx,  # reuse ts.tx inside update() for model updates; actor/critic use explicit tx below
            env_state=env_state,
            rng=rng,
            frames=jnp.asarray(0, dtype=jnp.int32),
            epoch=jnp.asarray(0, dtype=jnp.int32),
            h=h0,
            z=z0,
            a_prev_oh=a0,
            model_params=w_params["params"],
            model_opt_state=m_opt_state,
            actor_params=a_params["params"],
            actor_opt_state=a_opt_state,
            critic_params=c_params["params"],
            critic_opt_state=c_opt_state,
        )

        # Monkey-patch tx for other nets inside update (simple & jit-friendly)
        # We'll swap ts.tx before each block via replace()
        def set_tx(ts, tx):
            return ts.replace(tx=tx)

        def body(ts, _):
            # model
            ts_m = set_tx(ts, model_tx)
            ts_m, logs = self.update(ts_m, None)
            # swap in actor/critic tx for next calls within update()
            ts_m = ts_m.replace(tx=actor_tx)  # actor/critic use ts.tx internally
            ts_m = ts_m.replace(tx=critic_tx)
            return ts_m, logs

        start_time = jax.numpy.asarray(0.0)  # we’ll compute FPS in the driver script
        ts, logs = jax.lax.scan(body, ts, None, length=num_updates)
        return ts, logs
