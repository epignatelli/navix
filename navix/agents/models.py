from functools import partial
from typing import Callable, Sequence, Tuple
from jax import Array
import jax
import jax.numpy as jnp
import distrax
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
