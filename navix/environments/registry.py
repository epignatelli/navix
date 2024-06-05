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
from typing import Callable
import difflib


_ENVS_REGISTRY = {}


def registry():
    return _ENVS_REGISTRY


def register_env(name: str, ctor: Callable):
    _ENVS_REGISTRY[name] = ctor


def make(name: str, max_steps: int = 100, **kwargs):
    if name not in registry():
        closest = difflib.get_close_matches(name, registry().keys())
        msg = f"Environment {name} not yet implemented."
        if closest:
            msg += (
                f"Did you mean one of these? {closest}\n"
                + "If not, please open a feature request!"
                + "\nhttps://github.com/epignatelli/navix/issues/new?labels=enhancement"
            )
        raise NotImplementedError(msg)
    ctor = _ENVS_REGISTRY[name]
    return ctor(max_steps=max_steps, **kwargs)


NotImplementedEnvs = [
    "MiniGrid-BlockedUnlockPickup-v0",
    "MiniGrid-LavaCrossingS9N1-v0",
    "MiniGrid-LavaCrossingS9N2-v0",
    "MiniGrid-LavaCrossingS9N3-v0",
    "MiniGrid-LavaCrossingS11N5-v0",
    "MiniGrid-Fetch-5x5-N2-v0",
    "MiniGrid-Fetch-6x6-N2-v0",
    "MiniGrid-Fetch-8x8-N3-v0",
    "MiniGrid-GoToObject-6x6-N2-v0",
    "MiniGrid-GoToObject-8x8-N2-v0",
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
