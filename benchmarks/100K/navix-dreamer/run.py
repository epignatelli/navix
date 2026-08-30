"""Scores navix's own Dreamer against the Navix100K benchmark preset.

Reproduction: `python run.py` from any directory, no arguments - this
entry's static metadata lives in the sibling `config.yml`, its pinned
dependencies in the sibling `requirements.txt`."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from navix.agents import Dreamer, DreamerHparams, WorldModel, DreamerActor, DreamerCritic
from navix.agents.agent import masked_mean
from navix.benchmarks import AlgorithmEntry, Navix100K, TrainingCurve
from navix.environments.registry import make

HERE = Path(__file__).resolve().parent


class DreamerEntry(AlgorithmEntry):
    def train(self, env_id: str, budget: int, rng: jax.Array) -> TrainingCurve:
        env = make(env_id)
        hparams = DreamerHparams(budget=budget)
        # Unlike PPO's ActorCritic (which needs a pre-flattened observation),
        # Dreamer's world model flattens whatever shape env.observation_fn
        # returns internally - obs_dim just needs to match the flattened
        # size of that raw shape.
        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = len(env.action_set)
        agent = Dreamer(
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
        _, logs = agent.train(rng)
        mask = jnp.asarray(logs["done_mask"], dtype=jnp.bool_)
        return TrainingCurve(
            episodic_returns=masked_mean(logs["returns"], mask, axis=(-2, -1)),
            lengths=masked_mean(logs["lengths"], mask, axis=(-2, -1)),
        )


if __name__ == "__main__":
    config = yaml.safe_load((HERE / "config.yml").read_text())

    entry = DreamerEntry(
        name=config["name"],
        author=config["author"],
        paper_url=config["paper_url"],
        navix_commit_url=config["navix_commit_url"],
        algorithm_commit_url=config["algorithm_commit_url"],
    )

    benchmark = Navix100K()
    raw = benchmark.run(entry)
    summary = benchmark.summary(raw)
    details = benchmark.details(raw)
    benchmark.submit_entry(entry, raw)
    print(f"{type(benchmark).__name__} / {entry.name} summary:")
    print(f"  episodic_returns:     {summary['episodic_returns']}")
    print(f"  flops:                {summary['flops']}")
    print(f"  memory_bytes:         {summary['memory_bytes']}")
    print(f"  compile_time_seconds: {summary['compile_time_seconds']}")
    print(f"  fps:                  {summary['fps']}")
    print(f"  wall_time:            {summary['wall_time']}")
    print(f"  returns_variance:            {summary['returns_variance']}")
    print(f"  returns_convergence_rate:    {summary['returns_convergence_rate']}")
    for i, env_id in enumerate(details["env_ids"]):
        print(f"  {env_id}: episodic_returns={details['episodic_returns'][i]} length={details['length'][i]}")
