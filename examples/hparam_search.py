from dataclasses import dataclass, field
from typing import Dict

import distrax
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
    seed: int = 0
    # env
    env_id: str = "Navix-DoorKey-Random-6x6-v0"
    discount: float = 0.99
    # ppo
    ppo_config: PPOHparams = field(default_factory=PPOHparams)


class CategoricalUniform(distrax.Categorical):
    def __init__(self, domain: tuple, dtype=jnp.int32):
        self.domain = jnp.asarray(domain)
        super().__init__(logits=jnp.zeros(len(domain)), dtype=dtype)

    def sample(self, *, seed, sample_shape=()):
        samples = super().sample(seed=seed, sample_shape=sample_shape)
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

    # static hparams
    ppo_config = args.ppo_config.replace(anneal_lr=False)

    hparams_distr: Dict[str, distrax.Distribution] = {
        "gae_lambda": CategoricalUniform((0.7, 0.95, 0.99)),
        "clip_eps": CategoricalUniform((0.1, 0.2)),
        "ent_coef": CategoricalUniform((0.001, 0.01, 0.1)),
        "vf_coef": CategoricalUniform((0.1, 0.5, 0.9)),
        "lr": CategoricalUniform((1e-3, 2.5e-4, 1e-4, 1e-5)),
    }

    base_hparams = args.ppo_config
    experiment = nx.Experiment(
        name=args.project_name,
        agent=PPO(base_hparams, ActorCritic(len(env.action_set)), env),
        env=env,
        env_id=args.env_id,
        seeds=(args.seed,),
    )

    experiment.run_hparam_search(hparams_distr, args.population_size)
