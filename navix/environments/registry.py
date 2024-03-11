from functools import partial
from typing import Callable

from jax import Array

from ..entities import State


ENVS = {}


def register_env(name: str, ctor: Callable):
    ENVS[name] = ctor


def make(
    name: str,
    max_steps: int,
    observation_fn: Callable[[State], Array],
    reward_fn: Callable[[State, Array, State], Array],
    termination_fn: Callable[[State, Array, State], Array],
    **kwargs,
):
    if name in NotImplementedEnvs:
        raise NotImplementedError(
            f"Environment {name} not yet implemented. Please open a feature request!\
            \nhttps://github.com/epignatelli/naxiv"
        )
    ctor = ENVS[name]
    return ctor(
        max_steps=max_steps,
        observation_fn=observation_fn,
        reward_fn=reward_fn,
        termination_fn=termination_fn,
        **kwargs,
    )


NotImplementedEnvs = [
    "MiniGrid-BlockedUnlockPickup-v0",
    "MiniGrid-LavaCrossingS9N1-v0",
    "MiniGrid-LavaCrossingS9N2-v0",
    "MiniGrid-LavaCrossingS9N3-v0",
    "MiniGrid-LavaCrossingS11N5-v0",
    "MiniGrid-SimpleCrossingS9N1-v0",
    "MiniGrid-SimpleCrossingS9N2-v0",
    "MiniGrid-SimpleCrossingS9N3-v0",
    "MiniGrid-SimpleCrossingS11N5-v0",
    "MiniGrid-DistShift1-v0",
    "MiniGrid-DistShift2-v0",
    "MiniGrid-Dynamic-Obstacles-5x5-v0",
    "MiniGrid-Dynamic-Obstacles-Random-5x5-v0",
    "MiniGrid-Dynamic-Obstacles-6x6-v0",
    "MiniGrid-Dynamic-Obstacles-Random-6x6-v0",
    "MiniGrid-Dynamic-Obstacles-8x8-v0",
    "MiniGrid-Dynamic-Obstacles-16x16-v0",
    "MiniGrid-Fetch-5x5-N2-v0",
    "MiniGrid-Fetch-6x6-N2-v0",
    "MiniGrid-Fetch-8x8-N3-v0",
    "MiniGrid-FourRooms-v0",
    "MiniGrid-GoToDoor-5x5-v0",
    "MiniGrid-GoToDoor-6x6-v0",
    "MiniGrid-GoToDoor-8x8-v0",
    "MiniGrid-GoToObject-6x6-N2-v0",
    "MiniGrid-GoToObject-8x8-N2-v0",
    "MiniGrid-LavaGapS5-v0",
    "MiniGrid-LavaGapS6-v0",
    "MiniGrid-LavaGapS7-v0",
    "MiniGrid-LockedRoom-v0",
    "MiniGrid-MemoryS17Random-v0",
    "MiniGrid-MemoryS13Random-v0",
    "MiniGrid-MemoryS13-v0",
    "MiniGrid-MemoryS11-v0",
    "MiniGrid-MemoryS9-v0",
    "MiniGrid-MemoryS7-v0",
    "MiniGrid-MultiRoom-N2-S4-v0",
    "MiniGrid-MultiRoom-N4-S5-v0",
    "MiniGrid-MultiRoom-N6-v0",
    "MiniGrid-ObstructedMaze-1Dl-v0",
    "MiniGrid-ObstructedMaze-1Dlh-v0",
    "MiniGrid-ObstructedMaze-1Dlhb-v0",
    "MiniGrid-ObstructedMaze-2Dl-v0",
    "MiniGrid-ObstructedMaze-2Dlh-v0",
    "MiniGrid-ObstructedMaze-2Dlhb-v0",
    "MiniGrid-ObstructedMaze-1Q-v0",
    "MiniGrid-ObstructedMaze-2Q-v0",
    "MiniGrid-ObstructedMaze-Full-v0",
    "MiniGrid-ObstructedMaze-2Dlhb-v1",
    "MiniGrid-ObstructedMaze-1Q-v1",
    "MiniGrid-ObstructedMaze-2Q-v1",
    "MiniGrid-ObstructedMaze-Full-v1",
    "MiniGrid-Playground-v0",
    "MiniGrid-PutNear-6x6-N2-v0",
    "MiniGrid-PutNear-8x8-N3-v0",
    "MiniGrid-RedBlueDoors-6x6-v0",
    "MiniGrid-RedBlueDoors-8x8-v0",
    "MiniGrid-Unlock-v0",
    "MiniGrid-UnlockPickup-v0",
]
