"""Adapt a navix `Environment` to the `gymnax` API, so navix
environments can be dropped into agents and tooling written against
gymnax (`(obs, state)` from `reset`, `(obs, state, reward, done, info)`
from `step`, with an explicit `params`).
"""

from typing import Any, Dict, Tuple
import jax
from jax import Array
from flax import struct
from gymnax.environments.environment import (
    Environment as GymnaxEnv,
    EnvParams,
    EnvState,
)
from gymnax.environments.spaces import Discrete as GymnaxDiscrete, Box as GymnaxBox

from .environment import Environment, Timestep


@struct.dataclass
class GymnaxState(EnvState):
    """A gymnax `EnvState` that simply carries the wrapped navix
    `Timestep`. `ToGymnax.step` unwraps `timestep`, advances the navix
    environment, and rewraps.

    Attributes:
        timestep: the underlying navix `Timestep`.
        time: `timestep.t`, exposed at the top level because gymnax reads
            `state.time` for its own truncation bookkeeping.
    """

    timestep: Timestep
    time: Array


class ToGymnax(GymnaxEnv):
    """Wraps a navix `Environment` as a `gymnax.environments.Environment`.

    Prefer `ToGymnax.wrap(env)`, which returns the `(gymnax_env, params)`
    pair gymnax callers expect.

    ```python
    import navix as nx
    from navix.environments.wrappers import ToGymnax

    gymnax_env, params = ToGymnax.wrap(nx.make("Navix-Empty-5x5-v0"))
    obs, state = gymnax_env.reset(key, params)
    obs, state, reward, done, info = gymnax_env.step(key, state, action, params)
    ```

    `done` merges navix's truncation and termination into gymnax's single
    boolean; `StepType` is still available via `state.timestep.step_type`.
    Autoreset follows navix's rules (see `Environment.step`), not
    gymnax's.
    """

    def __init__(self, env: Environment):
        """Args:
        env (Environment): the navix environment to wrap."""
        self.env = env

    @property
    def default_params(self) -> EnvParams:
        """A gymnax `EnvParams` carrying the wrapped env's `max_steps` as
        `max_steps_in_episode`."""
        return EnvParams(max_steps_in_episode=self.env.max_steps)

    @classmethod
    def wrap(cls, env: Environment) -> Tuple[GymnaxEnv, EnvParams]:
        """Builds the wrapper and its params in one call.

        Args:
            env (Environment): the navix environment to wrap.

        Returns:
            tuple: `(ToGymnax(env), EnvParams(max_steps_in_episode=env.max_steps))`.
        """
        return cls(env=env), EnvParams(max_steps_in_episode=env.max_steps)

    def action_space(self, params: Any):
        """The gymnax `Discrete` action space, `len(env.action_set)`
        values. `params` is ignored (kept for API compatibility)."""
        return GymnaxDiscrete(len(self.env.action_set))

    def observation_space(self, params: Any):
        """The wrapped env's observation `Space` as a gymnax `Box`
        (`low`/`high`/`shape`/`dtype` copied across). `params` is
        ignored."""
        o_space = self.env.observation_space
        return GymnaxBox(
            low=o_space.minimum,
            high=o_space.maximum,
            shape=o_space.shape,
            dtype=o_space.dtype,
        )

    def reset(
        self, key: jax.Array, params: EnvParams | None = None
    ) -> Tuple[Array, EnvState]:
        """Resets the wrapped environment.

        Args:
            key (Array): PRNG key.
            params (EnvParams | None): ignored; `max_steps` comes from the
                wrapped env.

        Returns:
            tuple: `(observation, GymnaxState)`.
        """
        timestep = self.env.reset(key)
        return (
            timestep.observation,
            GymnaxState(time=timestep.t, timestep=timestep),
        )

    def step(
        self, key: Array, state: GymnaxState, action: jax.Array, params: EnvParams
    ) -> Tuple[Array, EnvState, Array, Array, Dict[str, Any]]:
        """Advances the wrapped environment one step.

        Args:
            key (Array): PRNG key (unused - navix carries its own key in
                `state.timestep.state.key`).
            state (GymnaxState): the state returned by the previous
                `reset`/`step`.
            action (Array): integer action, `[0, len(action_set))`.
            params (EnvParams): ignored.

        Returns:
            tuple: `(observation, GymnaxState, reward, done, info)`, where
            `done = timestep.is_done()` (truncation or termination).
        """
        timestep = self.env.step(state.timestep, action)
        return (
            timestep.observation,
            GymnaxState(time=timestep.t, timestep=timestep),
            timestep.reward,
            timestep.is_done(),
            timestep.info,
        )
