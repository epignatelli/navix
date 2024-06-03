from dataclasses import dataclass, field
import tyro
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import navix as nx
from navix import observations
from navix.agents import PPO, PPOHparams, ActorCritic
from navix.environments.environment import Environment


@dataclass
class Args:
    project_name = "navix-examples"
    env_id: str = "Navix-Empty-Random-5x5-v0"
    seed: int = 0
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
        max_steps=100,
        observation_fn=observations.symbolic,
    )
    env = FlattenObsWrapper(env)

    encoder = nn.Sequential(
        [
            nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
            nn.tanh,
            nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
            nn.tanh,
        ]
    )
    agent = PPO(
        hparams=args.ppo_config,
        network=ActorCritic(
            # actor_encoder=encoder.copy(),
            # critic_encoder=encoder.copy(),
            action_dim=len(env.action_set),
        ),
        env=env,
    )

    experiment = nx.Experiment(
        name=args.project_name,
        budget=40_000_000,
        agent=agent,
        env=env,
        seed=args.seed,
        debug=args.ppo_config.debug,
    )
    train_state, logs = experiment.run()
