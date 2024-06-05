from dataclasses import asdict, dataclass
import time
import wandb

import jax
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
    budget: int = 10_000_000
    seeds_offset: int = 0
    n_seeds: int = 10


if __name__ == "__main__":
    args = tyro.cli(Args)

    ppo_hparams = PPOHparams(budget=args.budget)
    # create environments
    for env_id in nx.registry():
        # init logging
        config = {**vars(args), **asdict(ppo_hparams)}
        wandb.init(project=args.project_name, config=config)

        # init environment
        env = FlattenObsWrapper(nx.make(env_id))

        # create agent
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
        agent = PPO(
            hparams=ppo_hparams,
            network=ActorCritic(action_dim=len(env.action_set)),
            env=env,
        )

        # train agent
        seeds = range(args.seeds_offset, args.seeds_offset + args.n_seeds)
        rngs = jnp.asarray([jax.random.PRNGKey(seed) for seed in seeds])
        train_fn = jax.vmap(agent.train)

        print("Compiling training function...")
        start_time = time.time()
        train_fn = jax.jit(train_fn).lower(rngs).compile()
        compilation_time = time.time() - start_time
        print(f"Compilation time cost: {compilation_time}")

        print("Training agent...")
        start_time = time.time()
        train_state, logs = train_fn(rngs)
        training_time = time.time() - start_time
        print(f"Training time cost: {training_time}")

        print("Logging final results to wandb...")
        start_time = time.time()
        # transpose logs tree
        logs = jax.tree_map(lambda *args: jnp.stack(args), *logs)
        for log in logs:
            agent.log_on_train_end(log)
        logging_time = time.time() - start_time
        print(f"Logging time cost: {logging_time}")

        print("Training complete")
        print(f"Compilation time cost: {compilation_time}")
        print(f"Training time cost: {training_time}")
        total_time = compilation_time + training_time
        print(f"Logging time cost: {logging_time}")
        total_time += logging_time
        print(f"Total time cost: {total_time}")
