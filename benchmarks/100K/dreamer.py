"""Scores navix's own Dreamer against the Navix100K benchmark preset."""
from dataclasses import dataclass
import subprocess

import numpy as np
import tyro

from navix.agents import Dreamer, DreamerHparams, WorldModel, DreamerActor, DreamerCritic
from navix.benchmarks import AlgorithmEntry, Navix100K
from navix.environments.environment import Environment


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def make_dreamer(env: Environment, hparams: DreamerHparams) -> Dreamer:
    # Unlike PPO's ActorCritic (which needs a pre-flattened observation),
    # Dreamer's world model flattens whatever shape env.observation_fn
    # returns internally - obs_dim just needs to match the flattened
    # size of that raw shape.
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = len(env.action_set)
    return Dreamer(
        hparams=hparams,
        env=env,
        world=WorldModel(obs_dim=obs_dim, act_dim=act_dim, hparams=hparams),
        actor=DreamerActor(act_dim=act_dim, hidden=hparams.hidden_size),
        critic=DreamerCritic(
            hidden=hparams.hidden_size,
            bins=hparams.bins,
            low=hparams.bins_low,
            high=hparams.bins_high,
        ),
    )


@dataclass
class Args:
    author: str = "navix"
    """Author of this implementation, for the AlgorithmEntry's provenance."""
    log_to_wandb: bool = True
    dreamer: DreamerHparams = DreamerHparams()


if __name__ == "__main__":
    args = tyro.cli(Args)
    commit_sha = _git_sha()

    entry = AlgorithmEntry(
        name="Dreamer",
        author=args.author,
        paper_url="https://arxiv.org/abs/2301.04104",
        commit_sha=commit_sha,
        requirements_url=(
            f"https://raw.githubusercontent.com/epignatelli/navix/{commit_sha}/requirements.txt"
        ),
        agent_factory=lambda env: make_dreamer(env, args.dreamer),
    )

    result = Navix100K(entry).run(log_to_wandb=args.log_to_wandb)
    print(f"{Navix100K.name} results:")
    print(f"  success_rate:   {result.success_rate}")
    print(f"  returns:        {result.returns}")
    print(f"  episode_length: {result.episode_length}")
    print(f"  fps:            {result.fps}")
    print(f"  wall_time:      {result.wall_time}")
