from dataclasses import dataclass
from typing import Any, Dict, Type

import tyro
import jax
import jax.experimental
import jax.numpy as jnp
from rejax import PPO, get_algo
from rejax.algos.ppo import PPOConfig
import wandb

import navix as nx
from navix import observations
from navix.environments.wrappers import ToGymnax


@dataclass
class Args:
    agent_kwargs: Dict[str, Any] = {"activation": "relu"}
    total_timesteps: int = 4_000_000
    eval_freq: int = 100_000
    num_envs: int = 16
    num_steps: int = 128
    num_epochs: int = 1
    num_minibatches: int = 8
    learning_rate: float = 0.00025
    max_grad_norm: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01


if __name__ == "__main__":
    args = tyro.cli(Args)
    config = vars(args)
    
    wandb.init(project="navix-examples-rejax", config=config)

    env = nx.make(
        "Navix-Empty-Random-5x5-v0", observation_fn=observations.symbolic, gamma=0.99
    )
    env = ToGymnax(env)

    # Get train function and initialize config for training
    algo = get_algo("ppo")
    agent_cls: Type[PPO] = algo[0]
    config_cls: Type[PPOConfig] = algo[1]
    config = config_cls.create(env=env, **config)
    eval_callback = config.eval_callback

    def wandb_callback(config, train_state, rng):
        lengths, returns = eval_callback(config, train_state, rng)

        def log(step, data):
            # io_callback returns np.array, which wandb does not like.
            # In jax 0.4.27, this becomes a jax array, should check when upgrading...
            step = step.item()
            wandb.log(data, step=step)

        jax.experimental.io_callback(
            log,
            (),  # result_shape_dtypes (wandb.log returns None)
            train_state.global_step,
            {
                "episode_length": lengths.mean(),
                "return": returns.mean(),
                "frames": train_state.global_step
                * config.num_envs
                * config.num_steps
                // (config.num_minibatches * config.num_epochs),
            },
        )

        # Since we log to wandb, we don't want to return anything that is collected
        # throughout training
        return jnp.array(())

    config = config.replace(eval_callback=wandb_callback)

    # Jit the training function
    train_fn = jax.jit(jax.vmap(agent_cls().train, in_axes=(None, 0)))

    # Train n_seeds agents!
    n_seeds = 5
    keys = jax.random.split(jax.random.PRNGKey(0), n_seeds)
    train_state, evaluation = train_fn(config, keys)