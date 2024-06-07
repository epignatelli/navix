from dataclasses import dataclass
from typing import Tuple
import wandb

import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import tyro
import navix as nx
from navix.environments.environment import Environment
from navix.agents import PPO, PPOHparams, ActorCritic


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
    seeds_range: Tuple[int, int, int] = (0, 10, 1)
    ppo: PPOHparams = PPOHparams()


if __name__ == "__main__":
    args = tyro.cli(Args)

    # create environments
    for env_id in nx.registry():
        # init logging
        config = {**vars(args), **{"observations": "symbolic"}, **{"algo": "ppo"}}
        wandb.init(project=args.project_name, config=config)

        # init environment
        env = nx.make(env_id)
        env = FlattenObsWrapper(env)

        # create agent
        agent = PPO(
            hparams=args.ppo,
            network=ActorCritic(action_dim=len(env.action_set)),
            env=env,
        )

        # run experiment
        experiment = nx.Experiment(
            name=args.project_name,
            agent=agent,
            env=env,
            env_id=env_id,
            seeds=tuple(range(*args.seeds_range)),
        )
        experiment.run()
