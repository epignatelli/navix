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
from typing import Dict

from jax import Array
from flax import struct
from jax.random import KeyArray
import jax.numpy as jnp

from .graphics import RenderingCache


class Component(struct.PyTreeNode):
    """A component is a part of the state of the environment."""


class Player(Component):
    """Players are entities that can act around the environment"""

    position: Array = jnp.zeros((1, 2), dtype=jnp.int32) - 1  # IntArray['b 2']
    """The (row, column) position of the entity in the grid, defaults to the discard pile (-1, -1)"""
    # TODO(epignatelli): consider batching player over the number of players
    # to allow tranposing the entities pytree for faster computation
    # and to prepare the ground for multi-agent environments
    tag: Array = jnp.asarray(1)  # IntArray['2']
    """The tag of the component, used to identify the type of the component in `oobservations.categorical`"""
    # we mark direction as static because it is convenient for mapping observations (e.g. jnp.rot90(grid, k=direction)
    # however, this is feasible because we only have 4 direcitions
    # will it scale in multi-agent settings?
    direction: Array = jnp.asarray(0, dtype=jnp.int32)  # IntArray['2']
    """The direction the entity: 0 = east, 1 = south, 2 = west, 3 = north"""
    pocket: Array = jnp.asarray(0)  # IntArray['2']
    """The id of the item in the pocket (0 if empty)"""


class Goal(Component):
    """Goals are entities that can be reached by the player"""

    position: Array = jnp.zeros((1, 2), dtype=jnp.int32) - 1  # IntArray['b 2']
    """The (row, column) position of the entity in the grid, defaults to the discard pile (-1, -1)"""
    tag: Array = jnp.ones((1,), dtype=jnp.int32) + 1  # IntArray['b']
    """The tag of the component, used to identify the type of the component in `oobservations.categorical`"""
    probability: Array = jnp.ones((1,), dtype=jnp.float32)  # FloatArray['b']
    """The probability of receiving the reward, if reached."""


class Pickable(Component):
    """Pickable items are world objects that can be picked up by the player.
    Examples of pickable items are keys, coins, etc."""

    position: Array = jnp.zeros((1, 2), dtype=jnp.int32) - 1  # IntArray['b 2']
    """The (row, column) position of the entity in the grid, defaults to the discard pile (-1, -1)"""
    id: Array = jnp.zeros((1,), dtype=jnp.int32) - 1  # IntArray['b']
    """The id of the item. If set, it must be >= 1."""

    @property
    def tag(self):
        return -self.id


class Consumable(Component):
    """Consumable items are world objects that can be consumed by the player.
    Consuming an item requires a tool (e.g. a key to open a door).
    A tool is an id (int) of another item, specified in the `requires` field (-1 if no tool is required).
    After an item is consumed, it is both removed from the `state.entities` collection, and replaced in the grid
    by the item specified in the `replacement` field (0 = floor by default).
    Examples of consumables are doors (to open) food (to eat) and water (to drink), etc.
    """

    position: Array = jnp.zeros((1, 2), dtype=jnp.int32) - 1  # IntArray['b 2']
    """The (row, column) position of the entity in the grid, defaults to the discard pile (-1, -1)"""
    requires: Array = jnp.zeros((1,), dtype=jnp.int32) - 1  # IntArray['b']
    """The id of the item required to consume this item. If set, it must be >= 1."""
    replacement: Array = jnp.zeros((1,), dtype=jnp.float32)  # IntArray['b']
    """The grid signature to replace the item with, usually 0 (floor). If set, it must be >= 1."""

    @property
    def tag(self) -> Array:  # -> IntArray['b']
        return self.requires


class State(struct.PyTreeNode):
    """The Markovian state of the environment"""

    key: KeyArray
    """The random number generator state"""
    grid: Array
    """The base map of the environment that remains constant throughout the training"""
    cache: RenderingCache
    """The rendering cache to speed up rendering"""
    player: Player  # we can potentially extend this to multiple players easily
    """The player entity"""
    goals: Goal = Goal()
    """The goal entity, batched over the number of goals"""
    keys: Pickable = Pickable()
    """The key entity, batched over the number of keys"""
    doors: Consumable = Consumable()
    """The door entity, batched over the number of doors"""

    def get_positions(self, axis: int = -1) -> Array:
        return jnp.stack(
            [
                *self.keys.position,
                *self.doors.position,
                *self.goals.position,
                self.player.position,
            ],
            axis=axis,
        )

    def get_tags(self, axis: int = -1) -> Array:
        return jnp.stack(
            [
                *self.keys.tag,
                *self.doors.tag,
                *self.goals.tag,
                self.player.tag,
            ],
            axis=axis,
        )

    def get_tiles(self, tiles_registry: Dict[str, Array], axis: int = 0) -> Array:
        return jnp.stack(
            [
                *([tiles_registry["key"]] * len(self.keys.position)),
                *([tiles_registry["door"]] * len(self.doors.position)),
                *([tiles_registry["goal"]] * len(self.goals.position)),
                tiles_registry["player"],
            ],
            axis=axis,
        )
