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
import warnings


_ENVS_REGISTRY = {}


def registry():
    return _ENVS_REGISTRY


def register_env(name: str, ctor: Callable):
    _ENVS_REGISTRY[name] = ctor


# MiniGrid ids whose "-v1" behaviour navix ports under a plain "-v0" id
# instead of registering a distinct "-v1" of its own (per this
# project's own v1-becomes-v0 convention - see each target's module
# docstring for the actual behavioural difference this represents:
# `obstructed_maze.py`'s `ObstructedMazeFull`, `multi_room.py`'s
# registration comment). `make` transparently redirects these and
# warns, rather than raising, so a caller reaching for the exact id
# MiniGrid itself uses for that behaviour still gets it.
V1_ALIASES = {
    "MiniGrid-ObstructedMaze-2Dlhb-v1": "Navix-ObstructedMaze-2Dlhb-v0",
    "MiniGrid-ObstructedMaze-1Q-v1": "Navix-ObstructedMaze-1Q-v0",
    "MiniGrid-ObstructedMaze-2Q-v1": "Navix-ObstructedMaze-2Q-v0",
    "MiniGrid-ObstructedMaze-Full-v1": "Navix-ObstructedMaze-Full-v0",
    "MiniGrid-MultiRoom-N4-S5-v1": "Navix-MultiRoom-N4-S5-v0",
}


def make(name: str, **kwargs):
    if name in V1_ALIASES:
        target = V1_ALIASES[name]
        warnings.warn(
            f"{name} has no distinct navix registration - {target} already "
            f"implements MiniGrid's own `-v1` behaviour for this environment "
            f"family (see V1_ALIASES in registry.py). Instantiating {target} "
            f"instead.",
            stacklevel=2,
        )
        name = target
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
    return ctor(**kwargs)


# Every MiniGrid id without a navix equivalent used to be tracked here.
# As of this commit that list is empty - every remaining MiniGrid id
# either has a matching Navix-* registration, or (the "-v1" family
# above) resolves through V1_ALIASES instead. Kept as a named, exported
# list (rather than deleted) so a future gap has an obvious place to be
# added back to.
NotImplementedEnvs = []
