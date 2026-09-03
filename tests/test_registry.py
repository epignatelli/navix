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

"""Issue #202: `registry.make()` used to default `max_steps` to a flat
`100` and always forward it explicitly, which silently defeated every
downstream `kwargs.pop("max_steps", <formula>)` fallback (they never
saw `"max_steps"` missing from `kwargs`) as well as `Environment.
create`'s own generic `4 * height * width` fallback for environments
with no override of their own - every registered environment silently
got `max_steps=100` via `nx.make(...)` regardless of its own intended
default."""

import warnings

import navix as nx
from navix.environments.registry import V1_ALIASES


def test_v1_aliases_redirect_and_warn():
    """`V1_ALIASES` ids (MiniGrid ids whose `-v1` behaviour navix ports
    under a plain `-v0` navix id instead of its own `-v1` registration)
    must still resolve through `nx.make`, via the aliased `-v0` id,
    with a warning explaining the redirect - not raise
    `NotImplementedError` the way a genuinely unregistered id does."""
    for minigrid_id, navix_id in V1_ALIASES.items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            env = nx.make(minigrid_id)
        assert len(caught) == 1, f"{minigrid_id}: expected exactly one warning"
        assert navix_id in str(caught[0].message), (
            f"{minigrid_id}: warning should name the aliased id {navix_id}"
        )
        assert env == nx.make(navix_id), (
            f"{minigrid_id}: should construct the same environment as {navix_id}"
        )

        # the aliased id itself must not warn - only its MiniGrid alias does
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            nx.make(navix_id)
        assert len(caught) == 0, f"{navix_id}: unexpected warning on direct use"


def test_make_without_max_steps_uses_environment_default():
    """An environment with no registration-specific `max_steps`
    override (e.g. `Room`/`Empty`) should fall through to `Environment.
    create`'s own `4 * height * width` default, not a flat 100."""
    env = nx.make("Navix-Empty-8x8-v0")
    assert env.max_steps == 4 * 8 * 8, (
        f"expected Environment.create's own 4*height*width default (256), "
        f"got {env.max_steps}"
    )


def test_make_without_max_steps_uses_registration_specific_default():
    """Environments with their own `kwargs.pop("max_steps", <formula>)`
    registration default should get that formula's value, not the
    flat 100 `registry.make()` used to always force."""
    cases = {
        "Navix-MemoryS17Random-v0": 5 * 17**2,
        "Navix-ObstructedMaze-1Dl-v0": 4 * 2 * 6**2,
        "Navix-ObstructedMaze-Full-v0": 4 * 25 * 6**2,
        "Navix-MultiRoom-N6-v0": 6 * 20,
    }
    for env_id, expected in cases.items():
        env = nx.make(env_id)
        assert env.max_steps == expected, (
            f"{env_id}: expected registration-specific default {expected}, "
            f"got {env.max_steps}"
        )


def test_make_with_explicit_max_steps_still_overrides():
    """An explicit `max_steps` must still win over any default -
    registration-specific formula or `Environment.create`'s own -
    exactly as before this fix."""
    for env_id in ("Navix-Empty-8x8-v0", "Navix-MultiRoom-N6-v0"):
        env = nx.make(env_id, max_steps=7)
        assert env.max_steps == 7, f"{env_id}: explicit max_steps=7 was not honoured"
