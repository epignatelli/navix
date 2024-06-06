# Copyright 2023 The Navix Authors.

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
from __future__ import annotations

import abc
from typing import Any, Callable, Dict, Tuple
import jax
import jax.numpy as jnp
from jax import Array
from flax import struct


from .. import rewards, terminations, observations, transitions
from ..rendering.cache import RenderingCache, TILE_SIZE
from ..states import State
from ..actions import DEFAULT_ACTION_SET
from ..spaces import Space, Discrete, Continuous


class StepType(struct.PyTreeNode):
    TRANSITION = jnp.asarray(0)
    """Standard timestep transition: the episode continues"""
    TRUNCATION = jnp.asarray(1)
    """The environment reached its maximum number of timesteps.
    The episode ended, but the agent could have still collected rewards.
    The value of the state is not 0"""
    TERMINATION = jnp.asarray(2)
    """The episode ended and the current state is an absorbing state."""


class Timestep(struct.PyTreeNode):
    t: Array
    """The number of timesteps elapsed from the last reset of the environment"""
    observation: Array
    """The observation corresponding to the current state (for POMDPs)"""
    action: Array
    """The action taken by the agent at the current timestep a_t = $\\pi(s_t)$, where $s_t$ is `state`"""
    reward: Array
    """The reward $r_{t=1}$ received by the agent after taking action $a_t$"""
    step_type: Array
    """The type of the current timestep, 0 for TRANSITION, 1 for TRUNCATION, 2 for TERMINATION"""
    state: State
    """The true state of the MDP, $s_t$ before taking action `action`"""
    info: Dict[str, Any] = struct.field(default_factory=dict)
    """Additional information about the environment. Useful for accumulations (e.g. returns)"""

    def is_truncation(self) -> Array:
        return self.step_type == StepType.TRUNCATION

    def is_termination(self) -> Array:
        return self.step_type == StepType.TERMINATION

    def is_transition(self) -> Array:
        return self.step_type == StepType.TRANSITION

    def is_done(self) -> Array:
        return jnp.logical_or(self.is_truncation(), self.is_termination())

    def is_start(self) -> Array:
        return self.t == 0


class Environment(struct.PyTreeNode):
    height: int = struct.field(pytree_node=False)
    width: int = struct.field(pytree_node=False)
    max_steps: int = struct.field(pytree_node=False)
    observation_space: Space = struct.field(pytree_node=False)
    action_space: Space = struct.field(pytree_node=False)
    reward_space: Space = struct.field(pytree_node=False)
    gamma: float = struct.field(pytree_node=False, default=0.99)
    penality_coeff: float = struct.field(pytree_node=False, default=0.0)
    observation_fn: Callable[[State], Array] = struct.field(
        pytree_node=False, default=observations.none
    )
    reward_fn: Callable[[State, Array, State], Array] = struct.field(
        pytree_node=False, default=rewards.DEFAULT_TASK
    )
    termination_fn: Callable[[State, Array, State], Array] = struct.field(
        pytree_node=False, default=terminations.DEFAULT_TERMINATION
    )
    transitions_fn: Callable[
        [State, Array, Tuple[Callable[[State], State], ...]], State
    ] = struct.field(pytree_node=False, default=transitions.DEFAULT_TRANSITION)
    action_set: Tuple[Callable[[State], State], ...] = struct.field(
        pytree_node=False, default=DEFAULT_ACTION_SET
    )

    @classmethod
    def create(
        cls,
        height: int,
        width: int,
        max_steps: int | None = None,
        observation_fn: Callable[[State], Array] = observations.symbolic,
        reward_fn: Callable[[State, Array, State], Array] = rewards.DEFAULT_TASK,
        termination_fn: Callable[
            [State, Array, State], Array
        ] = terminations.DEFAULT_TERMINATION,
        transitions_fn: Callable[
            [State, Array, Tuple[Callable[[State], State], ...]], State
        ] = transitions.DEFAULT_TRANSITION,
        action_set: Tuple[Callable[[State], State], ...] = DEFAULT_ACTION_SET,
        observation_space: Space | None = None,
        action_space: Space | None = None,
        reward_space: Space | None = None,
        **kwargs,
    ) -> Environment:

        if observation_space is None:
            observation_space = cls._get_obs_space_from_fn(
                width, height, observation_fn
            )
        if action_space is None:
            action_space = Discrete.create(len(action_set))
        if reward_space is None:
            reward_space = Continuous.create(
                shape=(), minimum=jnp.asarray(-1.0), maximum=jnp.asarray(1.0)
            )
        if max_steps is None:
            max_steps = 4 * height * width
        return cls(
            height=height,
            width=width,
            max_steps=max_steps,
            observation_fn=observation_fn,
            reward_fn=reward_fn,
            termination_fn=termination_fn,
            transitions_fn=transitions_fn,
            action_set=action_set,
            observation_space=observation_space,
            action_space=action_space,
            reward_space=reward_space,
            **kwargs,
        )

    @abc.abstractmethod
    def _reset(self, key: Array, cache: RenderingCache | None = None) -> Timestep:
        raise NotImplementedError()

    def reset(self, key: Array, cache: RenderingCache | None = None) -> Timestep:
        k1, k2 = jax.random.split(key)
        timestep = self._reset(k1, cache)
        timestep.info["return"] = jnp.asarray(0.0)
        return timestep.replace(state=timestep.state.replace(key=k2))

    def step(self, timestep: Timestep, action: Array) -> Timestep:
        # autoreset if necessary: 0 = transition, 1 = truncation, 2 = termination
        should_reset = timestep.step_type > 0
        return jax.lax.cond(
            should_reset,
            lambda timestep: self.reset(timestep.state.key, timestep.state.cache),
            lambda timestep: self._step(timestep, action),
            timestep,
        )

    def _step(self, timestep: Timestep, action: Array) -> Timestep:
        """
        Args:
            timestep (Timestep): The timestep at time $t$.
            action (Array): The action $a_t \\sim \\pi(A_t | s_t)$
        Returns:
            (Timestep): The timestep at time $t + 1$
        """
        # update agents
        state = self.transitions_fn(timestep.state, action, self.action_set)
        t = timestep.t + 1

        # calculate termination
        step_type = self.termination(timestep.state, action, state, timestep.t + 1)

        # calculate reward
        reward = self.reward_fn(timestep.state, action, state)
        reward = jax.lax.cond(
            step_type == StepType.TERMINATION,
            lambda reward: reward - self.penality_coeff * (t / self.max_steps),
            lambda reward: reward,
            reward,
        )

        new_timestep = Timestep(
            t=t,
            state=state,
            action=jnp.asarray(action),
            reward=reward,
            step_type=step_type,
            observation=self.observation_fn(state),
        )

        new_timestep.info["return"] = (
            timestep.info.get("return", jnp.asarray(0.0)) + reward
        )

        # build timestep
        return new_timestep

    def termination(
        self, prev_state: State, action: Array, state: State, t: Array
    ) -> Array:
        terminated = self.termination_fn(prev_state, action, state)
        truncated = t >= self.max_steps
        return terminations.check_truncation(terminated, truncated)

    @staticmethod
    def _get_obs_space_from_fn(
        width: int, height: int, observation_fn: Callable[[State], Array]
    ) -> Space:
        if observation_fn == observations.none:
            return Continuous.create(
                shape=(), minimum=jnp.asarray(0.0), maximum=jnp.asarray(0.0)
            )
        elif observation_fn == observations.categorical:
            return Discrete.create(n_elements=9, shape=(height, width))
        elif observation_fn == observations.categorical_first_person:
            radius = observations.RADIUS
            return Discrete.create(n_elements=9, shape=(radius + 1, radius * 2 + 1))
        elif observation_fn == observations.rgb:
            return Discrete.create(
                256,
                shape=(height * TILE_SIZE, width * TILE_SIZE, 3),
                dtype=jnp.uint8,
            )
        elif observation_fn == observations.rgb_first_person:
            radius = observations.RADIUS
            return Discrete.create(
                n_elements=256,
                shape=(radius * TILE_SIZE + 1, radius * TILE_SIZE * 2 + 1, 3),
                dtype=jnp.uint8,
            )
        elif observation_fn == observations.symbolic:
            return Discrete.create(
                n_elements=9,
                shape=(height, width, 3),
                dtype=jnp.uint8,
            )
        elif observation_fn == observations.symbolic_first_person:
            radius = observations.RADIUS
            return Discrete.create(
                n_elements=256,
                shape=(radius + 1, radius * 2 + 1, 3),
                dtype=jnp.uint8,
            )
        else:
            raise NotImplementedError(
                "Unknown observation space for observation function {}".format(
                    observation_fn
                )
            )
