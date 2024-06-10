from dataclasses import asdict, dataclass, field
import time
import jax
import tyro
import numpy as np
import jax.numpy as jnp
import wandb
import navix as nx
from navix import observations
from navix.agents import PPO, PPOHparams, ActorCritic
from navix.environments.environment import Environment

# set persistent compilation cache directory
# jax.config.update("jax_compilation_cache_dir", "/tmp/jax-cache/")


@dataclass
class Args:
    project_name = "navix-benchmarks"
    seeds_offset: int = 0
    n_seeds: int = 1
    # env
    env_id: str = "Navix-DoorKey-Random-8x8-v0"
    discount: float = 0.99
    # ppo
    ppo_config: PPOHparams = field(default_factory=PPOHparams)


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
        observation_fn=observations.symbolic_first_person,
        gamma=args.discount,
    )
    env = FlattenObsWrapper(env)

    ppo_config = args.ppo_config.replace(budget=10_000_000)
    agent = PPO(
        hparams=ppo_config,
        network=ActorCritic(
            action_dim=len(env.action_set),
        ),
        env=env,
    )

    experiment = nx.Experiment(
        name=args.project_name,
        agent=agent,
        env=env,
        env_id=args.env_id,
        seeds=tuple(range(args.seeds_offset, args.seeds_offset + args.n_seeds)),
        group="navix",
    )
    train_state, logs = experiment.run(do_log=False)

    print("Logging final results to wandb...")
    start_time = time.time()
    # average over seeds
    logs_avg = jax.tree.map(lambda x: x.mean(axis=0), logs)
    config = {**vars(experiment), **asdict(agent.hparams)}
    wandb.init(project=experiment.name, config=config, group=experiment.group)
    agent.log_on_train_end(logs_avg)
    wandb.finish()
    logging_time = time.time() - start_time
    print(f"Logging time cost: {logging_time}")
