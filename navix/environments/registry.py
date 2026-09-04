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
"""The environment registry: `make` an environment by string id, and
`register_env` to add your own.

Every built-in environment registers itself at import time (importing
`navix` is enough), so `navix.make(id)` is the normal way to get an
`Environment`. The ids follow MiniGrid's naming with a `Navix-` prefix
and a `-v0` suffix, e.g. `Navix-Empty-5x5-v0`, `Navix-DoorKey-8x8-v0`,
`Navix-MultiRoom-N4-S5-v0`.
"""

from typing import Callable
import difflib
import warnings


_ENVS_REGISTRY = {}


def registry() -> dict:
    """The full id -> constructor mapping.

    Returns:
        dict: maps each registered environment id (e.g.
        `"Navix-Empty-5x5-v0"`) to a zero-argument-or-kwargs callable
        that builds the `Environment`. The live dict, not a copy -
        `sorted(navix.registry())` lists every available id."""
    return _ENVS_REGISTRY


def register_env(name: str, ctor: Callable):
    """Adds an environment id to the registry (or overwrites one).

    Args:
        name (str): the id `make` will accept, e.g.
            `"Navix-MyEnv-8x8-v0"`.
        ctor (Callable): builds and returns the `Environment`. `make`
            calls it as `ctor(**kwargs)`, forwarding whatever keyword
            arguments were passed to `make` (`observation_fn`, `gamma`,
            `max_steps`, ...), so `ctor` typically ends in
            `MyEnv.create(...)`."""
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
"""MiniGrid `-v1` ids that navix implements under a plain `-v0` id rather
than as a separate `-v1` registration: for these families navix ports
MiniGrid's `-v1` behaviour and calls it `-v0` (the per-environment
behavioural difference is documented in each target's module docstring).
`make` redirects a `-v1` key here to its `-v0` value and warns, so code
written against MiniGrid's own id still resolves."""


def make(name: str, **kwargs):
    """Builds a registered environment by id.

    Args:
        name (str): a registered id (see `sorted(navix.registry())` for
            the full list). A handful of MiniGrid `-v1` ids are accepted
            too and transparently redirected to their navix `-v0`
            equivalent with a warning (see `V1_ALIASES`).
        **kwargs: forwarded verbatim to the environment's constructor.
            Common ones: `observation_fn` (a function from `navix.observations`,
            default is environment-specific), `gamma` (discount, default
            `0.99`), `max_steps` (episode truncation horizon, default
            `4 * height * width`), `observation_space` (override the
            inferred `Space`). Anything the specific environment's
            `create` accepts also works.

    Returns:
        Environment: the constructed environment.

    Raises:
        NotImplementedError: if `name` is not registered. The message
            lists close matches and links the feature-request form.

    Example:
        ```python
        import navix as nx
        from navix import observations

        env = nx.make(
            "Navix-DoorKey-8x8-v0",
            observation_fn=observations.rgb_first_person,
            gamma=0.995,
        )
        ```
    """
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


NotImplementedEnvs = []
"""MiniGrid environment ids that navix does not yet cover. Currently
empty - every MiniGrid id either has a `Navix-*` registration or resolves
through `V1_ALIASES`. Kept as a named list so a future gap has an obvious
place to be recorded."""
