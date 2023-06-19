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

from .. import tasks
from ..components import State
from .. import terminations
from ..actions import ACTIONS
from .. import observations


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
        pytree_node=False, default=observations.categorical
    )
    reward_fn: Callable[[State, Array, State], Array] = struct.field(
        pytree_node=False, default=tasks.navigation
    )
    termination_fn: Callable[[State, Array, State], Array] = struct.field(
        pytree_node=False, default=terminations.on_navigation_completion
    )

    @abc.abstractmethod
    def reset(self, key: KeyArray) -> Timestep:
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
            lambda timestep: self.reset(timestep.state.key),
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
