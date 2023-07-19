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
from enum import IntEnum
from typing import Any, Callable, Dict
import jax
import jax.numpy as jnp
from jax.random import KeyArray
from jax import Array
from flax import struct


from .. import tasks, terminations, observations
from ..graphics import RenderingCache, TILE_SIZE
from ..entities import State
from ..actions import ACTIONS
from ..spaces import Space, Discrete, Continuous


class StepType(IntEnum):
    TRANSITION = 0
    """discount > 0, episode continues"""
    TRUNCATION = 1
    """discount > 0, episode ends"""
    TERMINATION = 2
    """discount == 0, episode ends"""


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
    """The type of the current timestep (see `StepType`)"""
    state: State
    """The true state of the MDP, $s_t$ before taking action `action`"""
    info: Dict[str, Any] = struct.field(default_factory=dict)
    """Additional information about the environment. Useful for accumulations (e.g. returns)"""


class Environment(struct.PyTreeNode):
    height: int = struct.field(pytree_node=False)
    width: int = struct.field(pytree_node=False)
    max_steps: int = struct.field(pytree_node=False)
    gamma: float = struct.field(pytree_node=False, default=1.0)
    observation_fn: Callable[[State], Array] = struct.field(
        pytree_node=False, default=observations.none
    )
    reward_fn: Callable[[State, Array, State], Array] = struct.field(
        pytree_node=False, default=tasks.navigation
    )
    termination_fn: Callable[[State, Array, State], Array] = struct.field(
        pytree_node=False, default=terminations.on_navigation_completion
    )

    @property
    def observation_space(self) -> Space:
        if self.observation_fn == observations.none:
            return Continuous(shape=())
        elif self.observation_fn == observations.categorical:
            return Discrete(shape=(self.height, self.width))
        elif self.observation_fn == observations.categorical_first_person:
            radius = observations.RADIUS
            return Discrete(shape=(radius + 1, radius * 2 + 1))
        elif self.observation_fn == observations.rgb:
            return Discrete(
                256,
                shape=(self.height * TILE_SIZE, self.width * TILE_SIZE, 3),
                dtype=jnp.uint8,
            )
        elif self.observation_fn == observations.rgb_first_person:
            radius = observations.RADIUS
            return Discrete(
                256,
                shape=(radius * TILE_SIZE * 2 + 1, radius * TILE_SIZE * 2 + 1, 3),
                dtype=jnp.uint8,
            )
        else:
            raise NotImplementedError(
                "Unknown observation space for observation function {}".format(
                    self.observation_fn
                )
            )

    @property
    def action_space(self) -> Space:
        return Discrete(len(ACTIONS))

    @abc.abstractmethod
    def reset(self, key: KeyArray, cache: RenderingCache | None = None) -> Timestep:
        raise NotImplementedError()

    def _step(self, timestep: Timestep, action: Array, actions_set=ACTIONS) -> Timestep:
        # update agents
        state = jax.lax.switch(action, actions_set.values(), timestep.state)

        # build timestep
        return Timestep(
            t=timestep.t + 1,
            state=state,
            action=jnp.asarray(action),
            reward=self.reward(timestep.state, action, state),
            step_type=self.termination(timestep.state, action, state, timestep.t + 1),
            observation=self.observation(state),
        )

    def step(self, timestep: Timestep, action: Array, actions_set=ACTIONS) -> Timestep:
        # autoreset if necessary: 0 = transition, 1 = truncation, 2 = termination
        should_reset = timestep.step_type > 0
        return jax.lax.cond(
            should_reset,
            lambda timestep: self.reset(timestep.state.key, timestep.state.cache),
            lambda timestep: self._step(timestep, action, actions_set),
            timestep,
        )

    def observation(self, state: State):
        return self.observation_fn(state)

    def reward(self, state: State, action: Array, new_state: State):
        return self.reward_fn(state, action, new_state)

    def termination(
        self, prev_state: State, action: Array, state: State, t: Array
    ) -> Array:
        terminated = self.termination_fn(prev_state, action, state)
        truncated = t >= self.max_steps
        return terminations.check_truncation(terminated, truncated)
