from typing import Callable
from jax import Array
from flax import struct

from . import rewards, terminations
from .states import State


class Task(struct.PyTreeNode):
    """Task formuation as described in https://arxiv.org/abs/1609.01995"""

    reward_fn: Callable[[State, Array, State], Array]
    termination_fn: Callable[[State, Array, State], Array]


NAVIGATION = Task(rewards.on_goal_reached, terminations.on_goal_reached)

GO_TO_DOOR = Task(terminations.on_door_done, terminations.on_door_done)
