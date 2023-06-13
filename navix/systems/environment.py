from __future__ import annotations


import abc
from typing import Callable
import jax
import jax.numpy as jnp
from jax.random import KeyArray
from jax.typing import ArrayLike
from chex import Array
from flax import struct

from ..tasks import navigation
from ..components import State, Timestep, StepType
from ..transitions import deterministic_transition
from ..termination import on_navigation_completion, check_truncation
from ..actions import ACTIONS



class Environment(struct.PyTreeNode):
    width: ArrayLike
    height: ArrayLike
    max_steps: ArrayLike
    gamma: ArrayLike = struct.field(default=1.0)
    observation_fn: Callable[[Array], Array] = struct.field(
        pytree_node=False, default=lambda x: None
    )
    reward_fn: Callable[[Array, ArrayLike], Array] = struct.field(
        pytree_node=False, default=navigation
    )
    state_transition_fn: Callable[[Environment, Timestep], Timestep] = struct.field(
        pytree_node=False, default=deterministic_transition
    )
    termination_fn: Callable[[State, int], StepType] = struct.field(
        pytree_node=False, default=on_navigation_completion
    )

    @abc.abstractmethod
    def _reset(self, key: KeyArray) -> State:
        raise NotImplementedError()

    def _step(self, state: State, action: Array, ACTIONS=ACTIONS) -> State:
        # apply actions
        state = jax.lax.switch(action, ACTIONS.values(), state)
        # apply environment transition
        state = deterministic_transition(state, action)
        return state

    def reset(self, key: KeyArray) -> Timestep:
        state = self._reset(key)
        return Timestep(
            t=0,
            observation=self.observation(state.grid),
            action=jnp.asarray(0),
            reward=jnp.asarray(0.0),
            step_type=StepType(jnp.asarray(0)),
            state=state
        )

    def step(
        self, timestep: Timestep, action: Array, ACTIONS=ACTIONS
    ) -> Timestep:
        # autoreset if necessary
        state = jax.lax.cond(
            timestep.step_type,
            lambda _: self._step(timestep.state, action, ACTIONS),
            lambda _: self._reset(key),
            operand=(),
        )
        # calculate time accordingly
        t = jax.lax.cond(
            timestep.step_type, lambda _: timestep.t + 1, lambda _: 0, ()
        )
        return Timestep(
            t=t,
            state=state,
            action=jnp.asarray(action),
            reward=self.reward(state),
            step_type=self.step_type(state, timestep.t + 1),
            observation=self.observation(state.grid),
        )

    def observation(self, state: State):
        return self.observation_fn(state)

    def reward(self, state: State):
        return self.reward_fn(state)

    def step_type(self, state: State, t: int) -> StepType:
        terminated = self.termination_fn(state)
        truncated = t <= self.max_steps
        return check_truncation(terminated, truncated)

    def discount(self, state: State, t: int) -> Array:
        return (self.gamma ** t) * self.termination_fn(state)