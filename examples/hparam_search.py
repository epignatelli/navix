from dataclasses import dataclass, field, replace
import copy

import distrax
import jax
import tyro
import numpy as np
import jax.numpy as jnp
import navix as nx
from navix import observations
from navix.agents import PPO, PPOHparams, ActorCritic
from navix.environments.environment import Environment


@dataclass
class Args:
    project_name = "navix-debug"
    population_size: int = 10
    # env
    env_id: str = "Navix-DoorKey-Random-6x6-v0"
    discount: float = 0.99
    # ppo
    ppo_config: PPOHparams = field(default_factory=PPOHparams)


class CategoricalUniform(distrax.Categorical):
    def __init__(self, domain: tuple, dtype=jnp.int32):
        self.domain = jnp.asarray(domain)
        super().__init__(logits=jnp.zeros(len(domain)), dtype=dtype)

    def sample(self, rng):
        samples = super().sample(seed=rng, sample_shape=())
        return self.domain[samples]

    def sample_n(self, rng, n):
        samples = super().sample(seed=rng, sample_shape=(n,))
        return self.domain[samples]


if __name__ == "__main__":
    args = tyro.cli(Args)

    def FlattenObsWrapper(env: Environment):
        flatten_obs_fn = lambda x: jnp.ravel(env.observation_fn(x))
        flatten_obs_shape = (int(np.prod(env.observation_space.shape)),)
        return env.replace(
            observation_fn=flatten_obs_fn,
            observation_space=env.observation_space.replace(shape=flatten_obs_shape),
        )

    env = nx.make(
        args.env_id,
        observation_fn=observations.symbolic,
        gamma=args.discount,
    )
    env = FlattenObsWrapper(env)

    hparams_distr = {
        "n_steps": CategoricalUniform((128, 256)),
        "n_epochs": CategoricalUniform((1, 3)),
        "clip_ratio": CategoricalUniform((0.1, 0.2)),
        "entropy_coef": CategoricalUniform((0.001, 0.01, 0.1)),
        "learning_rate": CategoricalUniform((1e-4, 2.5e-4)),
        "gae_lambda": CategoricalUniform((0.7, 0.95, 0.99)),
    }

    agents = []
    for seed in range(args.population_size):
        print(f"Seed: {seed}")
        print("Hparams:")
        hparams = copy.deepcopy(args.ppo_config)
        key = jax.random.PRNGKey(seed)
        for k, distr in hparams_distr.items():
            hparams = replace(hparams, k=distr.sample(key).item())
        agent = PPO(
            hparams=hparams,
            network=ActorCritic(
                action_dim=len(env.action_set),
            ),
            env=env,
        )
        agents.append(agent)

    
