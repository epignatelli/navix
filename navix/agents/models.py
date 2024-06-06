from typing import Tuple
from jax import Array
import jax.numpy as jnp
import distrax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal


RNNState = tuple


class DenseRNN(nn.Dense):
    """A linear module that returns an empty RNN state,
    which makes it behave like an RNN layer."""

    @nn.compact
    def __call__(self, carry: RNNState, x: Array) -> Tuple[RNNState, Array]:
        return (), super().__call__(x)


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


class ActorCriticRnn(nn.Module):
    action_dim: int
    actor_encoder: nn.Module = MLPEncoder()
    critic_encoder: nn.Module = MLPEncoder()
    hidden_size: int = 64
    recurrent: bool = False

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

        if self.recurrent:
            self.core_actor = nn.LSTMCell(self.hidden_size)
            self.core_critic = nn.LSTMCell(self.hidden_size)
        else:
            self.core_actor = DenseRNN(self.hidden_size)
            self.core_critic = DenseRNN(self.hidden_size)

    def __call__(
        self, x: Array, carry: RNNState
    ) -> Tuple[RNNState, Tuple[distrax.Distribution, Array]]:
        pi = distrax.Categorical(logits=self.actor(x))
        v = jnp.squeeze(self.critic(x), -1)
        return carry, (pi, v)

    def policy(
        self, carry: RNNState, x: Array
    ) -> Tuple[RNNState, distrax.Distribution]:
        actor_embed = self.actor_encoder(x)
        carry, actor_embed = self.core_actor(carry, actor_embed)
        return carry, distrax.Categorical(logits=self.actor(x))

    def value(self, x: Array, carry: RNNState) -> Tuple[RNNState, Array]:
        critic_embed = self.critic_encoder(x)
        carry, critic_embed = self.core_critic(carry, critic_embed)
        return carry, jnp.squeeze(self.critic(x), axis=-1)
