from __future__ import annotations

import functools
from typing import Tuple
from jax import Array
import jax
import jax.numpy as jnp
import distrax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal


RNNState = Tuple[Array, Array]
ACHiddenState = Tuple[RNNState, RNNState]


class RNNWrapper(nn.RNNCellBase):
    module: nn.Module

    @nn.compact
    def __call__(self, carry, x):
        return self.module(x)

    def initialize_carry(self, rng, input_shape):
        return (jnp.asarray(()), jnp.asarray(()))

    @property
    def num_feature_axes(self) -> int:
        """Returns the number of feature axes of the RNN cell."""
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
            ]
        )

        self.critic = nn.Sequential(
            [
                self.critic_encoder,
                nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0)),
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
    recurrent: bool = False
    episodic_reset: bool = True
    embedding_size: int = 64

    def setup(self):
        if self.recurrent:
            actor_mlp = nn.LSTMCell(self.embedding_size)
            critic_mlp = nn.LSTMCell(self.embedding_size)
        else:
            actor_mlp = RNNWrapper(nn.Dense(self.embedding_size))
            critic_mlp = RNNWrapper(nn.Dense(self.embedding_size))

        self.actor = nn.Sequential(
            [
                RNNWrapper(self.actor_encoder),
                actor_mlp,
                RNNWrapper(
                    nn.Dense(
                        self.action_dim,
                        kernel_init=orthogonal(0.01),
                        bias_init=constant(0.0),
                    )
                ),
            ]
        )
        self.critic = nn.Sequential(
            [
                RNNWrapper(self.critic_encoder),
                critic_mlp,
                RNNWrapper(
                    nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))
                ),
            ]
        )

    def initialize_carry(
        self, rng: Array, input_shape: Tuple[int, ...]
    ) -> ACHiddenState:
        if self.recurrent:
            batch_dims = input_shape[:-1]
            mem_shape = batch_dims + (self.embedding_size,)
            carry_each = tuple(jnp.zeros((2,) + mem_shape, self.param_dtype))
            carry = (carry_each, carry_each)
        else:
            carry_each = (jnp.asarray(()), jnp.asarray(()))
            carry = (carry_each, carry_each)
        return carry

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    def __call__(
        self, carry: ACHiddenState, x: Array, done: Array
    ) -> Tuple[ACHiddenState, Tuple[distrax.Distribution, Array]]:
        if self.episodic_reset:
            carry = jax.tree.map(lambda x: jnp.where(done, 0.0, x), carry)

        carry, pi = self.policy(carry, x, done)
        carry, v = self.value(carry, x, done)
        return carry, (pi, v)

    def policy(
        self, carry: ACHiddenState, x: Array, done: Array
    ) -> Tuple[ACHiddenState, distrax.Distribution]:
        carry_actor, carry_critic = carry
        if self.episodic_reset:
            carry_actor = jax.tree.map(lambda x: jnp.where(done, 0.0, x), carry_actor)
        carry_actor, logits = self.actor(carry, x)
        return (carry_actor, carry_critic), distrax.Categorical(logits=logits)

    def value(
        self, carry: ACHiddenState, x: Array, done: Array
    ) -> Tuple[ACHiddenState, Array]:
        carry_actor, carry_critic = carry
        if self.episodic_reset:
            carry_critic = jax.tree.map(lambda x: jnp.where(done, 0.0, x), carry_critic)
        carry_critic, value = self.critic(carry, x)
        return (carry_actor, carry_critic), jnp.squeeze(value, -1)
