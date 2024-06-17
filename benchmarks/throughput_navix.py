# Make sure this is called before jax runs any operations!

from dataclasses import dataclass, field
import json

import jax

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache/")

import tyro
import numpy as np
import jax.numpy as jnp
import navix as nx
from navix import observations
from navix.agents import PPO, PPOHparams, ActorCritic
from navix.environments.environment import Environment


@dataclass
class Args:
    env_id: str = "Navix-Empty-5x5-v0"
    project_name = "navix-benchmarks"
    ppo_config: PPOHparams = field(default_factory=PPOHparams)
    n_seeds: int = 5


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
    )
    env = FlattenObsWrapper(env)

    ppo_config = args.ppo_config.replace(budget=1_000_000)
    agent = PPO(
        hparams=ppo_config,
        network=ActorCritic(
            action_dim=len(env.action_set),
        ),
        env=env,
    )

    agent = PPO(
        hparams=ppo_config,
        network=ActorCritic(
            action_dim=len(env.action_set),
        ),
        env=env,
    )
    results = {}
    for n_envs in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048):
        for seed in range(args.n_seeds):
            experiment = nx.Experiment(
                name=args.project_name,
                agent=agent,
                env=env,
                env_id=args.env_id,
                seeds=tuple(range(0, n_envs)),
                group="navix",
            )
            train_state, logs = experiment.run(do_log=False)
            results[n_envs] = results.get(n_envs, []) + [logs["bench/training_time"][0]]

    with open("results.json", "w") as f:
        json.dump(results, f)
