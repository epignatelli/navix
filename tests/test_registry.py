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

import navix as nx


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
