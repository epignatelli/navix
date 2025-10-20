# train_dreamer.py
from dataclasses import dataclass
from typing import Tuple
import wandb
import numpy as np
import jax.numpy as jnp
import tyro

import navix as nx
from navix.environments.environment import Environment
from navix.agents.dreamer import Dreamer, DreamerHparams, WorldModel, Actor, Critic


def FlattenObsWrapper(env: Environment):
    # Same as PPO example: make observations 1D
    flatten_obs_fn = lambda x: jnp.ravel(env.observation_fn(x))
    shape = (int(np.prod(env.observation_space.shape)),)
    return env.replace(
        observation_fn=flatten_obs_fn,
        observation_space=env.observation_space.replace(shape=shape),
    )


@dataclass
class Args:
    project_name: str = "navix-baselines"
    seeds_range: Tuple[int, int, int] = (0, 10, 1)
    dreamer: DreamerHparams = DreamerHparams()


if __name__ == "__main__":
    args = tyro.cli(Args)

    # Loop through NAVIX registry like your PPO script
    for env_id in nx.registry():
        # init logging
        config = {**vars(args), **{"observations": "symbolic"}, **{"algo": "dreamer"}}
        wandb.init(project=args.project_name, config=config)

        # init environment (flattened obs)
        env = nx.make(env_id)
        env = FlattenObsWrapper(env)

        # action/obs dims for modules
        action_dim = len(env.action_set)
        # Create agent
        agent = Dreamer(
            hparams=args.dreamer,
            env=env,
            world=WorldModel(
                obs_dim=env.observation_space.shape[0],
                act_dim=action_dim,
                hparams=args.dreamer,
            ),
            actor=Actor(act_dim=action_dim, hidden=args.dreamer.hidden_size),
            critic=Critic(hidden=args.dreamer.hidden_size),
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
