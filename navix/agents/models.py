"""Shared neural-network building blocks used across navix's agents.

Three groups live here: PPO's encoder/actor-critic components
(`MLPEncoder`, `ConvEncoder`, `TransformerEncoder`, `ActorCritic`),
Dreamer's world-model components (categorical-latent utilities, the
symexp-twohot head, and the RSSM's encoder/decoder/prior/posterior
networks) - the reusable pieces `navix.agents.dreamer.WorldModel` wires
together into an RSSM - and PQN's normalized Q-network (`QNetwork`). See
`navix.agents.dreamer` and `navix.agents.pqn`'s module docstrings for the
algorithms these latter components implement.

## The encoder contract (carry-based)

Every PPO/PQN feature extractor subclasses `Encoder` and implements the
same two-method interface, so an agent's training loop is written once and
works for both fully- and partially-observable settings by swapping only
the encoder:

- `initial_carry(obs_shape, dtype=float32) -> carry` - the encoder's
  hidden state at an episode boundary. `Encoder`'s default is stateless
  (`()`, ignoring both args); a stateful encoder overrides it. `dtype` is
  the observation's own dtype, so a raw-frame carry isn't silently upcast
  (`uint8` pixels -> `float32`).
- `__call__(carry, obs, is_first) -> (carry, features)` - consume one
  observation, emit the next carry and a feature vector. `is_first` (a
  bool, broadcast per batch element) marks `obs` as the first frame of a
  fresh episode, so a stateful encoder re-initialises its carry there
  rather than reading history that belongs to the episode that just
  ended.

The stateless encoders (`MLPEncoder`, `ConvEncoder`, and PQN's
`QMLPEncoder`/`QConvEncoder`) carry `()` and ignore `is_first`: threading
a carry through an agent that uses them is inert, and their output is
identical to the pre-carry versions. `TransformerEncoder` is the stateful
one (issue #169): its carry is a raw window of the last `context` frames,
so a `pomdp` observation function's single-frame stream becomes a
history-conditioned feature without the agent, the environment, or the
observation function changing. (Dreamer's RSSM encoder, `SymlogEncoder`,
is a separate thing - different `__call__` shape, no carry.)"""

from functools import partial
from typing import Any, Callable, Sequence, Tuple
from jax import Array
import jax
import jax.numpy as jnp
import distrax
import rlax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal


class Encoder(nn.Module):
    """Base for the PPO/PQN feature extractors - the carry-based encoder
    contract (see this module's docstring). Subclasses implement

        __call__(carry, obs, is_first) -> (carry, features)

    and, if they hold history across steps, override `initial_carry`. The
    default here is stateless: an empty carry, so threading it through an
    agent is inert. Not an `nn.Module` you instantiate directly.

    The agents that store a per-step carry and replay it in their loss
    (`PPO`, `PQN` - "Option 1") rely on the carry being a pure function of
    the observation stream, *independent of the encoder's parameters* - as
    `TransformerEncoder`'s raw-frame window is. An encoder whose carry is
    a learned state (a GRU/SSM hidden state) breaks that: the stored carry
    was produced under stale parameters, so replaying it is no longer what
    the current network would compute. Such an encoder needs the agent to
    recompute the carry over the sequence in the loss instead."""

    def initial_carry(
        self, obs_shape: Sequence[int], dtype: Any = jnp.float32
    ) -> Any:
        """The carry at an episode boundary. Stateless by default (`()`,
        ignoring both args); `TransformerEncoder` overrides this with a
        zeroed frame window of shape `(context, *obs_shape)` and dtype
        `dtype` - callers pass the observation's own dtype so the window
        isn't silently upcast (e.g. `uint8` pixels -> `float32`)."""
        return ()

    def __call__(self, carry: Any, obs: Array, is_first: Array):
        raise NotImplementedError


class MLPEncoder(Encoder):
    """Two `tanh` `Dense` layers - the default `ActorCritic` encoder for
    a flat (fully-observable / pre-flattened) observation. Stateless (see
    `Encoder`); its output is `hidden_size`-wide."""

    hidden_size: int = 64

    @nn.compact
    def __call__(self, carry, x, is_first):
        # Stateless: `carry` (always `()`) passes straight through and
        # `is_first` is ignored, so this is identical to the pre-carry
        # `MLPEncoder(x)` for any agent that threads a carry.
        features = nn.Sequential(
            [
                nn.Dense(self.hidden_size),
                nn.tanh,
                nn.Dense(self.hidden_size),
                nn.tanh,
            ]
        )(x)
        return carry, features


class ConvEncoder(Encoder):
    """`strides=(2, 2)` on every layer, not flax's `nn.Conv` default of
    1 (full-resolution, no downsampling) - an RL rollout batch is huge
    (`num_steps * num_envs` samples backprop'd through at once), and
    without downsampling each layer's activations stay at the input's
    full spatial resolution while channels grow 16->32->64, which blows
    up to tens of GB for a modest image (e.g. a 56x56 partially-
    observable render) at a real rollout batch size - confirmed via an
    actual OOM (single-op 24.5GB allocation) benchmarking navix's own
    PPO with this encoder on `observations.rgb_first_person`. Standard
    strided downsampling (every real CNN vision encoder in RL, e.g.
    Nature DQN's) keeps each layer's activation footprint bounded
    instead of constant-times-growing-channels."""

    hidden_size: int = 64

    @nn.compact
    def __call__(self, carry, x, is_first):
        # Stateless (see `MLPEncoder.__call__`).
        features = nn.Sequential(
            [
                nn.Conv(16, kernel_size=(2, 2), strides=(2, 2)),
                nn.relu,
                nn.Conv(32, kernel_size=(2, 2), strides=(2, 2)),
                nn.relu,
                nn.Conv(64, kernel_size=(2, 2), strides=(2, 2)),
                nn.relu,
                jnp.ravel,
                nn.Dense(self.hidden_size),
                nn.relu,
            ]
        )(x)
        return carry, features


class TransformerBlock(nn.Module):
    """One pre-LN transformer encoder block (self-attention + MLP, each
    with a residual connection and `LayerNorm` applied *before* the
    sub-layer, not after) - the standard modern choice (GPT-2 onwards)
    over the original "Attention Is All You Need" post-LN block, which
    needs a learning-rate warmup schedule to train stably; none of
    navix's other components add one, so pre-LN avoids relying on it."""

    hidden_size: int
    num_heads: int = 4
    mlp_ratio: int = 4

    @nn.compact
    def __call__(self, x: Array) -> Array:
        y = nn.LayerNorm()(x)
        y = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(y)
        x = x + y

        y = nn.LayerNorm()(x)
        y = nn.Dense(self.hidden_size * self.mlp_ratio)(y)
        y = nn.gelu(y)
        y = nn.Dense(self.hidden_size)(y)
        x = x + y
        return x


class TransformerEncoder(Encoder):
    """Issue #169: a `pomdp`-mode observation function (`rgb_first_person`/
    `categorical_first_person`/`symbolic_first_person`) returns a single
    current-step frame - no history. A lone frame doesn't disambiguate
    states that look identical from the agent's current viewpoint but
    differ in what led there (e.g. which direction something moved before
    this frame reached it), so a policy conditioned on it only sees a
    Markovian *approximation* of the true partially-observable state. This
    encoder instead attends over a short window of the last `context`
    frames, so the feature it hands to `ActorCritic` is conditioned on a
    real (if bounded) piece of history.

    History lives in the encoder's **carry**, not in the environment, the
    observation function, or the agent's training state. The carry is the
    literal `(context, *frame_shape)` window of the last `context` raw
    observations, oldest first; `__call__` rolls the new `obs` in (or, on
    `is_first`, refills the whole window with `obs` so it never reaches
    back into the episode that just ended) and returns the updated window
    *unchanged* as the next carry. Keeping the carry as raw frames - not
    per-frame embeddings - is deliberate: the window then has no
    dependence on the encoder's parameters, so an agent that stores it at
    collection time and reuses it in its loss (rather than re-deriving it)
    gets the exact same gradient, not an approximation.

    `frame_encoder` is applied to each of the `context` frames with
    *shared* weights (same submodule instance, called once per frame -
    `context` is a static field, so this unrolls at trace time and
    repeated calls reuse its parameters rather than creating `context`
    copies). Its output must already be `hidden_size`-wide, the same
    implicit dimension contract `ActorCritic` places on its
    `actor_encoder`/`critic_encoder`.

    A learned positional embedding is added per frame position before
    attention (plain per-position table, not sinusoidal - the context
    length is fixed and small). The output is the *last* token's
    post-attention feature (the current frame, now contextualised by the
    ones before it), not a pool over all positions, which would blur the
    current frame's signal into older, less relevant ones."""

    frame_encoder: nn.Module
    hidden_size: int = 64
    num_heads: int = 4
    num_layers: int = 2
    context: int = 4

    def initial_carry(
        self, obs_shape: Sequence[int], dtype: Any = jnp.float32
    ) -> Array:
        """The frame window at an episode boundary: `context` zero frames,
        `(context, *obs_shape)`. `obs_shape` is a single observation's
        shape (no batch axis) - the caller `vmap`s `__call__` over the
        batch, so it `vmap`s an `initial_carry`-shaped leaf the same way.
        `dtype` is the observation's own dtype: the window stores raw
        frames, so keeping it (e.g. `uint8` for navix's first-person
        pixel/symbolic observations) rather than upcasting to `float32`
        keeps `Buffer.carry` from being `context`-frames-wide *and* 4x
        per element."""
        return jnp.zeros((self.context, *tuple(obs_shape)), dtype=dtype)

    @nn.compact
    def __call__(
        self, carry: Array, obs: Array, is_first: Array
    ) -> Tuple[Array, Array]:
        # carry: (context, *frame_shape), oldest first, in the
        # observation's own dtype. Roll `obs` in at the end; on a fresh
        # episode, refill the window with `obs` so no position holds a
        # frame from the previous episode.
        obs = obs.astype(carry.dtype)
        rolled = jnp.concatenate([carry[1:], obs[None]], axis=0)
        fresh = jnp.broadcast_to(obs, carry.shape)
        window = jnp.where(is_first, fresh, rolled)  # next carry, obs dtype

        # `frame_encoder` follows the same encoder contract: a stateless
        # spatial encoder, called per frame with its own blank carry and
        # no reset flag. Cast to float here (not in the stored window) so
        # `nn.Dense`/`nn.Conv` see floats without widening what's buffered.
        frames = window.astype(jnp.float32)
        fe_carry = self.frame_encoder.initial_carry(frames.shape[1:])
        not_first = jnp.asarray(False)
        embed = jnp.stack(
            [
                self.frame_encoder(fe_carry, frames[t], not_first)[1]
                for t in range(self.context)
            ],
            axis=0,
        )  # (context, hidden_size) - shared frame_encoder, one call per frame
        pos_embedding = self.param(
            "pos_embedding",
            nn.initializers.normal(0.02),
            (self.context, self.hidden_size),
        )
        embed = embed + pos_embedding
        for _ in range(self.num_layers):
            embed = TransformerBlock(
                hidden_size=self.hidden_size, num_heads=self.num_heads
            )(embed)
        embed = nn.LayerNorm()(embed)
        return window, embed[-1]  # (context, *frame), (hidden_size,)


class ActorCritic(nn.Module):
    """PPO's network: two independent `Encoder` towers (actor, critic)
    each followed by a linear head - a categorical policy over
    `action_dim` and a scalar value. Swap `actor_encoder` /
    `critic_encoder` to change what the agent sees (e.g.
    `TransformerEncoder` for frame history); the training loop is
    unchanged.

    Attributes:
        action_dim: number of discrete actions (`len(env.action_set)`).
        actor_encoder: `Encoder` for the policy tower.
        critic_encoder: `Encoder` for the value tower. Must produce the
            same carry shape as `actor_encoder` (they share one carry).
    """

    action_dim: int
    actor_encoder: Encoder = MLPEncoder()
    critic_encoder: Encoder = MLPEncoder()

    def initial_carry(
        self, obs_shape: Sequence[int], dtype: Any = jnp.float32
    ) -> Any:
        """A single carry, shared by the actor and critic encoders. This
        assumes the two encoders derive their carry the same way from the
        same observation stream - true for the encoders here (a stateless
        `()`, or `TransformerEncoder`'s raw-frame window, which doesn't
        depend on the encoder's parameters) - so threading one carry and
        advancing it once per step is correct and avoids the actor's and
        critic's windows drifting apart when only one of `policy`/`value`
        runs (as in `PPO.collect_experience`, which calls `policy` only).
        `()` for the stateless encoders.

        Raises `ValueError` if the actor and critic encoders don't agree
        on the carry (e.g. a stateless actor with a stateful critic, or
        two `TransformerEncoder`s with different `context`) - otherwise
        the mismatch only surfaces as an opaque shape error deep in a
        later `.apply()` trace."""
        actor_carry = self.actor_encoder.initial_carry(obs_shape, dtype)
        critic_carry = self.critic_encoder.initial_carry(obs_shape, dtype)
        a_shapes = [x.shape for x in jax.tree_util.tree_leaves(actor_carry)]
        c_shapes = [x.shape for x in jax.tree_util.tree_leaves(critic_carry)]
        if a_shapes != c_shapes:
            raise ValueError(
                "ActorCritic threads one shared encoder carry, so "
                "actor_encoder and critic_encoder must produce the same "
                f"carry shape - got actor {a_shapes}, critic {c_shapes}."
            )
        return actor_carry

    def setup(self):
        # `layers_0` is an identity passthrough, not the encoder: the
        # encoder is now called separately (it returns `(carry,
        # features)`, which `nn.Sequential` can't thread). Keeping the
        # head at `nn.Sequential` slot `layers_1` preserves its parameter
        # path (`actor/layers_1`, `critic/layers_1`) - and therefore its
        # init RNG - byte-for-byte against the pre-carry `ActorCritic`, so
        # a fixed-seed run with a stateless encoder is unchanged.
        self.actor = nn.Sequential(
            [
                lambda x: x,
                nn.Dense(
                    self.action_dim,
                    kernel_init=orthogonal(0.01),
                    bias_init=constant(0.0),
                ),
            ]
        )
        self.critic = nn.Sequential(
            [
                lambda x: x,
                nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0)),
            ]
        )

    def __call__(
        self, carry: Any, x: Array, is_first: Array
    ) -> Tuple[Any, Tuple[distrax.Distribution, Array]]:
        # Both encoders advance the *same* carry from the same `x`; for
        # the encoders here they produce the identical next carry, so
        # returning the actor's is well-defined (see `initial_carry`).
        next_carry, actor_feat = self.actor_encoder(carry, x, is_first)
        _, critic_feat = self.critic_encoder(carry, x, is_first)
        pi = distrax.Categorical(self.actor(actor_feat))
        value = jnp.squeeze(self.critic(critic_feat), -1)
        return next_carry, (pi, value)

    def policy(
        self, carry: Any, x: Array, is_first: Array
    ) -> Tuple[Any, distrax.Distribution]:
        next_carry, actor_feat = self.actor_encoder(carry, x, is_first)
        pi = distrax.Categorical(logits=self.actor(actor_feat))
        return next_carry, pi

    def value(
        self, carry: Any, x: Array, is_first: Array
    ) -> Tuple[Any, Array]:
        next_carry, critic_feat = self.critic_encoder(carry, x, is_first)
        value = jnp.squeeze(self.critic(critic_feat), -1)
        return next_carry, value


# -------------------------
# Dreamer: categorical latents - unimix + straight-through sampling
# -------------------------


def unimix_categorical(logits: Array, unimix: float) -> distrax.Categorical:
    """Builds a `distrax.Categorical` from `logits` after mixing in a
    `unimix` fraction of uniform probability mass across the last axis -
    the "unimix" trick (1% by default in the paper): guarantees every
    class keeps at least `unimix / num_classes` probability, so neither
    the KL term nor the entropy can collapse to exactly zero, which keeps
    the prior from ever fully committing and losing gradient signal."""
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
    """Sum of `KL(post_i || prior_i)` over the `num_latents` independent
    categoricals (the second-to-last axis) - each categorical's own KL,
    summed, matching how DreamerV3 treats the full stochastic latent
    (`num_latents` categoricals of `num_classes` each, "stoch"/"classes"
    in the official implementation) as one joint distribution for the
    purposes of the free-nats floor."""
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


class SymlogEncoder(nn.Module):
    """Dreamer's RSSM observation encoder: a symlog-input MLP mapping a
    raw observation to a `embed_size` embedding for the posterior. Named
    for its distinctive `rlax.signed_logp1` (symlog) input transform -
    not part of the PPO/PQN `Encoder` carry-contract family above (it has
    a different `__call__` shape and no carry)."""

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
        # rlax.signed_logp1 (unlike the hand-rolled symlog it replaced)
        # asserts its input is already float via chex.assert_type - navix
        # observations can be uint8 (e.g. `categorical`/`symbolic`), so
        # cast explicitly rather than relying on implicit promotion.
        return net(rlax.signed_logp1(obs.astype(jnp.float32)))


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
    recurrent_size: int
    """Size of the GRU's deterministic hidden state ("deter" in the
    official implementation)."""

    @nn.compact
    def __call__(self, h_prev: Array, z_prev_flat: Array, a_prev_oh: Array) -> Array:
        x = jnp.concatenate([z_prev_flat, a_prev_oh], axis=-1)
        gru = nn.GRUCell(features=self.recurrent_size)
        h, _ = gru(h_prev, x)  # GRUCell returns (new_carry, y); y == new_carry
        return h


class PriorNet(nn.Module):
    hidden_size: int
    """Hidden layer size of the prior network's MLP."""
    num_latents: int
    """Number of independent categorical latent variables ("stoch" in
    the official implementation)."""
    num_classes: int
    """Number of classes per categorical latent variable ("classes" in
    the official implementation)."""

    @nn.compact
    def __call__(self, h: Array) -> Array:
        """Returns raw logits, `(..., num_latents, num_classes)` - unimix
        is applied by the caller (`unimix_categorical`), not baked in
        here, so every caller treats prior and posterior identically."""
        x = nn.elu(nn.Dense(self.hidden_size)(h))
        logits = nn.Dense(self.num_latents * self.num_classes)(x)
        return logits.reshape(*h.shape[:-1], self.num_latents, self.num_classes)


class PostNet(nn.Module):
    hidden_size: int
    """Hidden layer size of the posterior network's MLP."""
    num_latents: int
    """Number of independent categorical latent variables ("stoch" in
    the official implementation)."""
    num_classes: int
    """Number of classes per categorical latent variable ("classes" in
    the official implementation)."""

    @nn.compact
    def __call__(self, h: Array, embed: Array) -> Array:
        x = jnp.concatenate([h, embed], axis=-1)
        x = nn.elu(nn.Dense(self.hidden_size)(x))
        logits = nn.Dense(self.num_latents * self.num_classes)(x)
        return logits.reshape(*h.shape[:-1], self.num_latents, self.num_classes)


# -------------------------
# PQN: normalized Q-network
# -------------------------


class QMLPEncoder(Encoder):
    """`QNetwork`'s default (MDP, fully-observable/flattened) feature
    extractor: Dense/LayerNorm/ReLU stacked twice. LayerNorm after every
    hidden layer is not incidental here the way it might be in `MLPEncoder`
    above: it's the specific regularizer the PQN paper shows keeps online
    Q-learning convergent with no replay buffer and no target network
    (see `navix.agents.pqn`'s module docstring) - so unlike `ActorCritic`'s
    encoders, `QNetwork`'s own encoders (this and `QConvEncoder`) always
    keep it, rather than leaving it out like the shared `MLPEncoder`/
    `ConvEncoder` do. Stateless (`Encoder`'s `()` carry)."""

    hidden_size: int = 64

    @nn.compact
    def __call__(self, carry, x, is_first):
        # navix observations can be any shape (a (H, W) grid, (H, W, 3)
        # RGB/symbolic, ...) and this is always called on one example at
        # a time (the caller vmaps over the env/batch axis - PQN never
        # calls this with a leading batch dim of its own). Flattening
        # here means PQN's own env doesn't need external wrapping (e.g.
        # `examples/ppo.py`'s FlattenObsWrapper) the way PPO's does -
        # same ergonomics as Dreamer's `_flatten_obs`.
        # Stateless: `carry` (`()`) passes through, `is_first` is ignored.
        x = jnp.ravel(x)
        features = nn.Sequential(
            [
                nn.Dense(
                    self.hidden_size,
                    kernel_init=orthogonal(jnp.sqrt(2.0)),
                    bias_init=constant(0.0),
                ),
                nn.LayerNorm(),
                nn.relu,
                nn.Dense(
                    self.hidden_size,
                    kernel_init=orthogonal(jnp.sqrt(2.0)),
                    bias_init=constant(0.0),
                ),
                nn.LayerNorm(),
                nn.relu,
            ]
        )(x)
        return carry, features


class QConvEncoder(Encoder):
    """`QNetwork`'s POMDP (partially-observable pixel) feature extractor:
    same strided-downsampling conv stack as `ConvEncoder` (see its
    docstring for why the stride matters), projected through a Dense/
    LayerNorm/ReLU head to match `QMLPEncoder`'s regularization - PQN's
    LayerNorm-for-stability argument applies to the features `QNetwork`
    regresses Q-values from regardless of whether they came from a Dense
    or Conv stack, so this keeps it rather than dropping it for pixels.
    Stateless (`Encoder`'s `()` carry)."""

    hidden_size: int = 64

    @nn.compact
    def __call__(self, carry, x, is_first):
        # Stateless: `carry` (`()`) passes through, `is_first` is ignored.
        features = nn.Sequential(
            [
                nn.Conv(16, kernel_size=(2, 2), strides=(2, 2)),
                nn.relu,
                nn.Conv(32, kernel_size=(2, 2), strides=(2, 2)),
                nn.relu,
                nn.Conv(64, kernel_size=(2, 2), strides=(2, 2)),
                nn.relu,
                jnp.ravel,
                nn.Dense(
                    self.hidden_size,
                    kernel_init=orthogonal(jnp.sqrt(2.0)),
                    bias_init=constant(0.0),
                ),
                nn.LayerNorm(),
                nn.relu,
            ]
        )(x)
        return carry, features


class QNetwork(nn.Module):
    """The Q-network PQN regresses towards Q(lambda) targets - a
    pluggable `Encoder` feature extractor (`QMLPEncoder` by default for
    MDP/flattened observations, swappable for `QConvEncoder` for POMDP/
    pixel observations, or `TransformerEncoder` for frame history - same
    encoder-swap pattern `ActorCritic` uses) followed by a linear head
    over `action_dim` raw Q-values (no output activation). Threads the
    encoder carry through like `ActorCritic`."""

    action_dim: int
    encoder: Encoder = QMLPEncoder()

    def initial_carry(
        self, obs_shape: Sequence[int], dtype: Any = jnp.float32
    ) -> Any:
        """The encoder's carry (`()` for the stateless Q-encoders)."""
        return self.encoder.initial_carry(obs_shape, dtype)

    @nn.compact
    def __call__(self, carry: Any, x: Array, is_first: Array) -> Tuple[Any, Array]:
        carry, feat = self.encoder(carry, x, is_first)
        # Unlike PPO's actor head (small-std output init) or Dreamer's
        # zero-init heads, the reference PQN implementation uses the
        # same orthogonal(sqrt(2)) init on the output layer as every
        # hidden layer - no special small-scale treatment for the
        # Q-value outputs.
        q = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(jnp.sqrt(2.0)),
            bias_init=constant(0.0),
        )(feat)  # raw Q-values, (action_dim,)
        return carry, q
