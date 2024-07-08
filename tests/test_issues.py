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

import jax
import jax.numpy as jnp

import navix as nx
from navix import observations


def test_82():
    env = nx.make(
        "Navix-DoorKey-5x5-v0",
        max_steps=100,
        observation_fn=observations.rgb,
    )
    key = jax.random.PRNGKey(5)
    timestep = env.reset(key)
    # Seed 5 is:
    # # # # #
    # P # . #
    # . # . #
    # K D G #
    # # # # #

    # start agent direction = EAST
    prev_pos = timestep.state.entities["player"].position
    # action 2 is forward
    timestep = env.step(timestep, 2)  # should not walk into wall
    pos = timestep.state.entities["player"].position
    assert jnp.array_equal(prev_pos, pos)


if __name__ == "__main__":
    test_82()
