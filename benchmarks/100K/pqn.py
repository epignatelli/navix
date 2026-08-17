"""Scores navix's own PQN against the Navix100K benchmark preset."""
from dataclasses import dataclass
import subprocess

import tyro

from navix.agents import PQN, PQNHparams
from navix.agents.models import QNetwork
from navix.benchmarks import AlgorithmEntry, Navix100K
from navix.environments.environment import Environment


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def make_pqn(env: Environment, hparams: PQNHparams) -> PQN:
    # QNetwork flattens whatever shape env.observation_fn returns
    # internally - no FlattenObsWrapper needed, same ergonomics as
    # Dreamer's world model.
    network = QNetwork(action_dim=len(env.action_set), hidden_size=hparams.hidden_size)
    return PQN(hparams=hparams, network=network, env=env)


@dataclass
class Args:
    author: str = "navix"
    """Author of this implementation, for the AlgorithmEntry's provenance."""
    log_to_wandb: bool = True
    pqn: PQNHparams = PQNHparams()


if __name__ == "__main__":
    args = tyro.cli(Args)
    commit_sha = _git_sha()

    entry = AlgorithmEntry(
        name="PQN",
        author=args.author,
        paper_url="https://arxiv.org/abs/2407.04811",
        commit_sha=commit_sha,
        requirements_url=(
            f"https://raw.githubusercontent.com/epignatelli/navix/{commit_sha}/requirements.txt"
        ),
        agent_factory=lambda env: make_pqn(env, args.pqn),
    )

    result = Navix100K(entry).run(log_to_wandb=args.log_to_wandb)
    print(f"{Navix100K.name} results:")
    print(f"  success_rate:   {result.success_rate}")
    print(f"  returns:        {result.returns}")
    print(f"  episode_length: {result.episode_length}")
    print(f"  fps:            {result.fps}")
    print(f"  wall_time:      {result.wall_time}")
