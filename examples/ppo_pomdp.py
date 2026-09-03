"""Same `PPO` as `examples/ppo.py`, on a first-person (`pomdp`)
observation - the only change is the encoder. A single first-person frame
is not a Markovian observation (issue #169); `TransformerEncoder` gives
the policy a short window of history instead, and it manages that window
itself (as its carry), so the agent, the environment, and the observation
function are all identical to the fully-observable case.
"""

from dataclasses import dataclass, field
import tyro
import numpy as np
import jax.numpy as jnp
import navix as nx
from navix import observations
from navix.agents import PPO, PPOHparams, ActorCritic, MLPEncoder, TransformerEncoder
from navix.environments.environment import Environment


@dataclass
class Args:
    project_name = "navix-examples"
    seeds_offset: int = 0
    n_seeds: int = 1
    env_id: str = "Navix-DoorKey-Random-6x6-v0"
    discount: float = 0.99
    context: int = 4
    """Number of frames the encoder attends over."""
    ppo_config: PPOHparams = field(default_factory=lambda: PPOHparams(budget=5_000))
    """Small budget - a quick usage demo, not a convergence benchmark."""


if __name__ == "__main__":
    args = tyro.cli(Args)

    def FlattenObsWrapper(env: Environment):
        # per-frame flatten: the encoder attends over a window of these.
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

    def encoder() -> TransformerEncoder:
        return TransformerEncoder(
            frame_encoder=MLPEncoder(hidden_size=64),
            hidden_size=64,
            context=args.context,
        )

    agent = PPO(
        hparams=args.ppo_config,
        # the ONLY difference from examples/ppo.py: the encoder.
        network=ActorCritic(
            action_dim=len(env.action_set),
            actor_encoder=encoder(),
            critic_encoder=encoder(),
        ),
        env=env,
    )

    experiment = nx.Experiment(
        name=args.project_name,
        agent=agent,
        env=env,
        env_id=args.env_id,
        seeds=tuple(range(args.seeds_offset, args.seeds_offset + args.n_seeds)),
    )
    train_state, logs = experiment.run()
