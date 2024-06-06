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
    timestep: Timestep
    time: Array


class ToGymnax(GymnaxEnv):
    def __init__(self, env: Environment):
        self.env = env

    @property
    def default_params(self) -> EnvParams:
        return EnvParams(max_steps_in_episode=self.env.max_steps)

    @classmethod
    def wrap(cls, env: Environment) -> Tuple[GymnaxEnv, EnvParams]:
        return cls(env=env), EnvParams(max_steps_in_episode=env.max_steps)

    def action_space(self, params: Any):
        return GymnaxDiscrete(len(self.env.action_set))

    def observation_space(self, params: Any):
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
        timestep = self.env.reset(key)
        return (
            timestep.observation,
            GymnaxState(time=timestep.t, timestep=timestep),
        )

    def step(
        self, key: Array, state: GymnaxState, action: jax.Array, params: EnvParams
    ) -> Tuple[Array, EnvState, Array, Array, Dict[str, Any]]:
        timestep = self.env.step(state.timestep, action)
        return (
            timestep.observation,
            GymnaxState(time=timestep.t, timestep=timestep),
            timestep.reward,
            timestep.is_done(),
            timestep.info,
        )
