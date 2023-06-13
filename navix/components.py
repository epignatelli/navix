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


from typing import Dict, Any, List
from enum import IntEnum
from jax.tree_util import register_pytree_node_class
from jax.random import KeyArray
from chex import Array
from flax import struct


class Component(struct.PyTreeNode):
    id: int
    walkable: bool = True
    direction: int = -1
    pocket: List[Any] = struct.field(default_factory=list)


class Timestep(struct.PyTreeNode):
    t: Array
    observation: Array
    action: Array
    reward: Array
    step_type: Array
    state: State
    info: Dict[str, Any] = struct.field(default_factory=dict)


class State(struct.PyTreeNode):
    """The Markovian state of the environment"""

    key: KeyArray
    """The random number generator state"""
    grid: Array
    """The mesh containing the positions the entities"""
    entities: Dict[str, Component]
    """The entities in the environment"""


class StepType(IntEnum):
    TRANSITION = 0
    """discount > 0, episode continues"""
    TRUNCATION = 1
    """discount > 0, episode ends"""
    TERMINATION = 2
    """discount == 0, episode ends"""
