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

    @nn.compact
    def __call__(self, x):
        actor_repr = self.actor_encoder(x)
        logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(actor_repr)
        pi = distrax.Categorical(logits=logits)

        critic_repr = self.critic_encoder(x)
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic_repr
        )
        return pi, jnp.squeeze(value, axis=-1)
