from dataclasses import dataclass
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import navix as nx
from navix.environments.environment import Environment
from navix.agents import PPO, PPOHparams


def FlattenObsWrapper(env: Environment):
    flatten_obs_fn = lambda x: jnp.ravel(env.observation_fn(x))
    flatten_obs_shape = (int(np.prod(env.observation_space.shape)),)
    return env.replace(
        observation_fn=flatten_obs_fn,
        observation_space=env.observation_space.replace(shape=flatten_obs_shape),
    )


@dataclass
class Args:
    project_name = "navix-baselines"
    budget: int = 10_000_000
    debug: bool = True


if __name__ == "__main__":
    for env_id in nx.registry():
        env = nx.make(env_id)
        print(env_id)
        env = FlattenObsWrapper(env)

        network = nn.Sequential(
            [
                nn.Dense(
                    64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
                ),
                nn.tanh,
                nn.Dense(
                    64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
                ),
                nn.tanh,
            ]
        )

        hparams = PPOHparams(
            
        )
