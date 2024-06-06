import functools
from typing import Tuple
from jax import Array
import jax
import jax.numpy as jnp
import distrax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal


RNNState = tuple


class DenseRNN(nn.Dense, nn.RNNCellBase):
    """A linear module that returns an empty RNN state,
    which makes it behave like an RNN layer."""

    @nn.compact
    def __call__(self, carry: RNNState, x: Array) -> Tuple[RNNState, Array]:
        return (), super().__call__(x)

    @nn.nowrap
    def initialize_carry(
        self, rng: Array, input_shape: Tuple[int, ...]
    ) -> Tuple[Array, Array]:
        return (jnp.asarray(()), jnp.asarray(()))

    @property
    def num_feature_axes(self) -> int:
        return 1


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


class ActorCriticRNN(nn.Module):
    action_dim: int
    actor_encoder: nn.Module = MLPEncoder()
    critic_encoder: nn.Module = MLPEncoder()
    hidden_size: int = 64
    recurrent: bool = False

    def setup(self):
        if self.recurrent:
            self.core_actor = nn.LSTMCell(self.hidden_size)
            self.core_critic = nn.LSTMCell(self.hidden_size)
        else:
            self.core_actor = DenseRNN(self.hidden_size)
            self.core_critic = DenseRNN(self.hidden_size)

        self.actor_head = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )
        self.critic_head = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    def __call__(
        self, x: Array, carry: Tuple[RNNState, RNNState], done=None
    ) -> Tuple[RNNState, Tuple[distrax.Distribution, Array]]:
        if done is None:
            done = jnp.zeros(x.shape[0], dtype=jnp.bool_)

        # TODO(epignatelli): Implement reset

        carry_actor, pi = self.policy(carry, x)
        carry_critic, v = self.value(carry, x)
        return (carry_actor, carry_critic), (pi, v)

    @nn.nowrap
    def initialize_carry(
        self, rng: Array, input_shape: Tuple[int, ...]
    ) -> Tuple[RNNState, RNNState]:
        carry_actor = self.core_actor.initialize_carry(rng, input_shape)
        carry_critic = self.core_critic.initialize_carry(rng, input_shape)
        return (carry_actor, carry_critic)

    def policy(
        self, carry: Tuple[RNNState, RNNState], x: Array
    ) -> Tuple[RNNState, distrax.Distribution]:
        carry_actor, carry_critic = carry
        actor_embed = self.actor_encoder(x)
        carry_actor, actor_embed = self.core_actor(carry_actor, actor_embed)
        logits = self.actor_head(actor_embed)
        carry = (carry_actor, carry_critic)
        return carry, distrax.Categorical(logits=logits)

    def value(
        self, carry: Tuple[RNNState, RNNState], x: Array
    ) -> Tuple[RNNState, Array]:
        carry_actor, carry_critic = carry
        critic_embed = self.critic_encoder(x)
        carry_critic, critic_embed = self.core_critic(carry_critic, critic_embed)
        value = self.critic_head(critic_embed)
        carry = (carry_actor, carry_critic)
        return carry, jnp.squeeze(value, axis=-1)
