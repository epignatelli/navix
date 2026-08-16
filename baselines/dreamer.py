from dataclasses import dataclass
from typing import Tuple

import numpy as np
import wandb
import tyro

import navix as nx
from navix.agents import Dreamer, DreamerHparams, WorldModel, DreamerActor, DreamerCritic


@dataclass
class Args:
    project_name = "navix-baselines"
    seeds_range: Tuple[int, int, int] = (0, 10, 1)
    dreamer: DreamerHparams = DreamerHparams()


if __name__ == "__main__":
    args = tyro.cli(Args)

    for env_id in nx.registry():
        config = {**vars(args), **{"observations": "symbolic"}, **{"algo": "dreamer"}}
        wandb.init(project=args.project_name, config=config)

        # Unlike PPO's ActorCritic (which needs a pre-flattened
        # observation via FlattenObsWrapper, see baselines/ppo.py),
        # Dreamer's world model flattens whatever shape self.env.
        # observation_fn returns internally - obs_dim just needs to match
        # the flattened size of that raw shape.
        env = nx.make(env_id)
        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = len(env.action_set)
        agent = Dreamer(
            hparams=args.dreamer,
            env=env,
            world=WorldModel(obs_dim=obs_dim, act_dim=act_dim, hparams=args.dreamer),
            actor=DreamerActor(act_dim=act_dim, hidden=args.dreamer.hidden_size),
            critic=DreamerCritic(hidden=args.dreamer.hidden_size),
        )

        experiment = nx.Experiment(
            name=args.project_name,
            agent=agent,
            env=env,
            env_id=env_id,
            seeds=tuple(range(*args.seeds_range)),
        )
        experiment.run()
