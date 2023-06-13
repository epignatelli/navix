from __future__ import annotations


import abc
from typing import Dict, Any, List
from enum import IntEnum
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
