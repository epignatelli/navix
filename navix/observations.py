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

from jax import Array
import jax
import jax.numpy as jnp

from .components import State


def third_person_view(state: State, radius: int) -> Array:
    raise NotImplementedError()


def first_person_view(state: State, radius: int) -> Array:
    raise NotImplementedError()


def categorical(state: State) -> Array:
    # updates are in reverse order of display
    # place keys (keys are represented with opposite sign of the door they open)
    grid = jnp.max(jax.vmap(lambda key: state.grid.at[tuple(key.position)].set(-key.id))(state.keys), axis=0)
    # place doors
    grid = jnp.max(jax.vmap(lambda door: grid.at[tuple(door.position)].set(door.requires))(state.doors), axis=0)
    # place goals
    grid = jnp.max(jax.vmap(lambda goal: grid.at[tuple(goal.position)].set(2))(state.goals), axis=0)
    # place player last, always on top
    grid = grid.at[tuple(state.player.position)].set(1)
    return grid


def one_hot(state: State) -> Array:
    raise NotImplementedError()


def pixels(state: State) -> Array:
    raise NotImplementedError()
