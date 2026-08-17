"""Shared, algorithm-agnostic helpers for `benchmarks/<preset>/<entry>/
run.py` scripts. Not itself part of any entry's reproduction inputs
(it's generic infrastructure, not algorithm-specific config/code) - the
entry-specific pieces are each `run.py`, its sibling `config.yml`, and
`requirements.txt`, per issue #130's per-entry folder layout."""
import subprocess

import jax.numpy as jnp
import numpy as np

from navix.environments.environment import Environment


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def flatten_obs(env: Environment) -> Environment:
    """PPO's `ActorCritic` (unlike Dreamer's world model or PQN's
    `QNetwork`, which flatten internally) needs a pre-flattened
    observation."""
    flatten_obs_fn = lambda x: jnp.ravel(env.observation_fn(x))
    flatten_shape = (int(np.prod(env.observation_space.shape)),)
    return env.replace(
        observation_fn=flatten_obs_fn,
        observation_space=env.observation_space.replace(shape=flatten_shape),
    )
