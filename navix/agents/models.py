"""Shared neural-network building blocks used across navix's agents.

Two groups live here: PPO's encoder/actor-critic components
(`MLPEncoder`, `ConvEncoder`, `ActorCritic`), and Dreamer's world-model
components (categorical-latent utilities, the symexp-twohot head, and
the RSSM's encoder/decoder/prior/posterior networks) - the reusable
pieces `navix.agents.dreamer.WorldModel` wires together into an RSSM.
See `navix.agents.dreamer`'s module docstring for the algorithm these
latter components implement."""

from functools import partial
from typing import Callable, Sequence, Tuple
from jax import Array
import jax
import jax.numpy as jnp
import distrax
import rlax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal


class MLPEncoder(nn.Module):
    hidden_size: int = 64

    @nn.compact
    def __call__(self, x):
        return nn.Sequential(
            [
                nn.Dense(self.hidden_size),
                nn.tanh,
                nn.Dense(self.hidden_size),
                nn.tanh,
            ]
        )(x)


class ConvEncoder(nn.Module):
    hidden_size: int = 64

    @nn.compact
    def __call__(self, x):
        return nn.Sequential(
            [
                nn.Conv(16, kernel_size=(2, 2)),
                nn.relu,
                nn.Conv(32, kernel_size=(2, 2)),
                nn.relu,
                nn.Conv(64, kernel_size=(2, 2)),
                nn.relu,
                jnp.ravel,
                nn.Dense(self.hidden_size),
                nn.relu,
            ]
        )(x)


class ActorCritic(nn.Module):
    action_dim: int
    actor_encoder: nn.Module = MLPEncoder()
    critic_encoder: nn.Module = MLPEncoder()

    def setup(self):
        self.actor = nn.Sequential(
            [
                self.actor_encoder,
                nn.Dense(
                    self.action_dim,
                    kernel_init=orthogonal(0.01),
                    bias_init=constant(0.0),
                ),
                # lambda x: distrax.Categorical(logits=x),
            ]
        )

        self.critic = nn.Sequential(
            [
                self.critic_encoder,
                nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0)),
                # lambda x: jnp.squeeze(x, axis=-1),
            ]
        )

    def __call__(self, x: Array) -> Tuple[distrax.Distribution, Array]:
        return distrax.Categorical(self.actor(x)), jnp.squeeze(self.critic(x), -1)

    def policy(self, x: Array) -> distrax.Distribution:
        return distrax.Categorical(logits=self.actor(x))

    def value(self, x: Array) -> Array:
        return jnp.squeeze(self.critic(x), -1)


# -------------------------
# Dreamer: categorical latents - unimix + straight-through sampling
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
    trainable end-to-end despite being discrete - distinct from Dreamer's
    *actor* action sampling, which deliberately does NOT use this (see
    `navix.agents.dreamer`'s module docstring)."""
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
# Dreamer: symexp-twohot head (reward, value)
# -------------------------


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
        # The output layer is ZERO-initialized (official implementation's
        # `outscale: 0.0` for the reward/critic heads): all-zero logits ->
        # uniform bin probabilities -> mean symexp(sum(p*bins)) = symexp(0)
        # = 0 exactly, so an untrained head predicts 0, not symexp of
        # whatever random logits happen to sum to over bins spanning
        # symlog +-20 - which can reach +-e^20. Without this, the very
        # first imagination rollouts feed the actor advantages of garbage
        # magnitude; the return-normalization EMA (rate 0.01) is far too
        # slow to absorb that, and ~64 gradient steps of it were observed
        # to slam the actor into a deterministic policy (entropy pinned at
        # the unimix floor) before the world model had learned anything -
        # after which exploration never recovers.
        net = nn.Sequential(
            [
                nn.Dense(self.hidden_size),
                nn.elu,
                nn.Dense(self.hidden_size),
                nn.elu,
                nn.Dense(self.bins, kernel_init=nn.initializers.zeros),
            ]
        )
        return net(feat)  # logits, (..., bins)

    def loss(self, logits: Array, target: Array) -> Array:
        twohot = rlax.transform_to_2hot(
            rlax.signed_logp1(target), self.low, self.high, self.bins
        )
        logp = jax.nn.log_softmax(logits, axis=-1)
        return -jnp.sum(twohot * logp, axis=-1)

    def mean(self, logits: Array) -> Array:
        probs = jax.nn.softmax(logits, axis=-1)
        return rlax.signed_expm1(
            rlax.transform_from_2hot(probs, self.low, self.high, self.bins)
        )


# -------------------------
# Dreamer: RSSM building blocks (encoder/decoder/prior/posterior)
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
        return net(rlax.signed_logp1(obs))


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
