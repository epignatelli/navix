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
from enum import IntEnum
from typing import Any, Dict, Tuple

from jax import Array
from flax import struct
from jax.random import KeyArray
import jax.numpy as jnp

from .graphics import RenderingCache, SpritesRegistry


DISCARD_PILE_COORDS = jnp.asarray((0, -1), dtype=jnp.int32)
DISCARD_PILE_IDX = jnp.asarray(-1, dtype=jnp.int32)
EMPTY_POCKET_ID = jnp.asarray(-1, dtype=jnp.int32)
UNSET_DIRECTION = jnp.asarray(-1, dtype=jnp.int32)
UNSET_CONSUMED = jnp.asarray(-1, dtype=jnp.int32)


# class Component(struct.PyTreeNode):
#     """A component is a part of the state of the environment."""


# class Player(Component):
#     """Players are entities that can act around the environment"""
#     # TODO(epignatelli): consider batching player over the number of players
#     # to allow tranposing the entities pytree for faster computation
#     # and to prepare the ground for multi-agent environments

#     position: Array = DISCARD_PILE_COORDS  # IntArray['b 2']
#     """The (row, column) position of the entity in the grid, defaults to the discard pile (-1, -1)"""
#     tag: Array = jnp.asarray(1)  # IntArray['2']
#     """The tag of the component, used to identify the type of the component in `oobservations.categorical`"""
#     direction: Array = jnp.asarray(0, dtype=jnp.int32)  # IntArray['2']
#     """The direction the entity: 0 = east, 1 = south, 2 = west, 3 = north"""
#     pocket: Array = EMPTY_POCKET_ID  # IntArray['2']
#     """The id of the item in the pocket (0 if empty)"""


# class Goal(Component):
#     """Goals are entities that can be reached by the player"""

#     position: Array = DISCARD_PILE_COORDS[None]  # IntArray['b 2']
#     """The (row, column) position of the entity in the grid, defaults to the discard pile (-1, -1)"""
#     tag: Array = jnp.ones((1,), dtype=jnp.int32) + 1  # IntArray['b']
#     """The tag of the component, used to identify the type of the component in `oobservations.categorical`"""
#     probability: Array = jnp.ones((1,), dtype=jnp.float32)  # FloatArray['b']
#     """The probability of receiving the reward, if reached."""


# class Key(Component):
#     """Pickable items are world objects that can be picked up by the player.
#     Examples of pickable items are keys, coins, etc."""

#     position: Array = DISCARD_PILE_COORDS[None]  # IntArray['b 2']
#     """The (row, column) position of the entity in the grid, defaults to the discard pile (-1, -1)"""
#     id: Array = jnp.ones((1,), dtype=jnp.int32)  # IntArray['b']
#     """The id of the item. If set, it must be >= 1."""

#     @property
#     def tag(self):
#         return -self.id


# class Door(Component):
#     """Consumable items are world objects that can be consumed by the player.
#     Consuming an item requires a tool (e.g. a key to open a door).
#     A tool is an id (int) of another item, specified in the `requires` field (-1 if no tool is required).
#     After an item is consumed, it is both removed from the `state.entities` collection, and replaced in the grid
#     by the item specified in the `replacement` field (0 = floor by default).
#     Examples of consumables are doors (to open) food (to eat) and water (to drink), etc.
#     """

#     position: Array = DISCARD_PILE_COORDS[None]  # IntArray['b 2']
#     """The (row, column) position of the entity in the grid, defaults to the discard pile (-1, -1)"""
#     direction: Array = jnp.zeros((1,), dtype=jnp.int32)  # IntArray['b']
#     """The direction the entity: 0 = east, 1 = south, 2 = west, 3 = north"""
#     requires: Array = EMPTY_POCKET_ID[None]  # IntArray['b']
#     """The id of the item required to consume this item. If set, it must be >= 1."""
#     replacement: Array = jnp.zeros((1,), dtype=jnp.float32)  # IntArray['b']
#     """The grid signature to replace the item with, usually 0 (floor). If set, it must be >= 1."""
#     open: Array = jnp.zeros((1,), dtype=jnp.bool_)  # BoolArray['b']
#     """Whether the door is open, closed or locked (0 = closed, 1 = open, 2 = locked)"""

#     @property
#     def tag(self) -> Array:  # -> IntArray['b']
#         return self.requires


# class State(struct.PyTreeNode):
#     """The Markovian state of the environment"""

#     key: KeyArray
#     """The random number generator state"""
#     grid: Array
#     """The base map of the environment that remains constant throughout the training"""
#     entities: Tuple[Entity, ...]
#     cache: RenderingCache
#     """The rendering cache to speed up rendering"""
#     player: Player  # we can potentially extend this to multiple players easily
#     """The player entity"""
#     goals: Goal = Goal()
#     """The goal entity, batched over the number of goals"""
#     keys: Key = Key()
#     """The key entity, batched over the number of keys"""
#     doors: Door = Door()
#     """The door entity, batched over the number of doors"""

#     def get_positions(self, axis: int = -1) -> Array:
#         return jnp.stack(
#             [
#                 *self.keys.position,
#                 *self.doors.position,
#                 *self.goals.position,
#                 self.player.position,
#             ],
#             axis=axis,
#         )

#     def get_tags(self, axis: int = -1) -> Array:
#         return jnp.stack(
#             [
#                 *self.keys.tag,
#                 *self.doors.tag,
#                 *self.goals.tag,
#                 self.player.tag,
#             ],
#             axis=axis,
#         )

#     def get_tiles(self, tiles_registry: Dict[str, Array], axis: int = 0) -> Array:
#         return jnp.stack(
#             [
#                 *([tiles_registry["key"]] * len(self.keys.position)),
#                 *([tiles_registry["door"]] * len(self.doors.position)),
#                 *([tiles_registry["goal"]] * len(self.goals.position)),
#                 tiles_registry["player"],
#             ],
#             axis=axis,
#         )


class EntityType(IntEnum):
    WALL = 0
    FLOOR = 1
    PLAYER = 2
    GOAL = 3
    KEY = 4
    DOOR = 5


class Component(struct.PyTreeNode):
    entity_type: Array = jnp.zeros((1,), dtype=jnp.int32)  # IntArray['b']
    """The type of the entity, 0 = player, 1 = goal, 2 = key, 3 = door"""


class Positionable(struct.PyTreeNode):
    position: Array = DISCARD_PILE_COORDS[None]  # IntArray['b 2']
    """The (row, column) position of the entity in the grid, defaults to the discard pile (-1, -1)"""


class Directional(struct.PyTreeNode):
    direction: Array = jnp.zeros((1,), dtype=jnp.int32)  # IntArray['b']
    """The direction the entity: 0 = east, 1 = south, 2 = west, 3 = north"""


class HasTag(struct.PyTreeNode):
    tag: Array = jnp.zeros((1,), dtype=jnp.int32)  # IntArray['b']
    """The tag of the component, used to identify the type of the component in `oobservations.categorical`"""


class Stochastic(struct.PyTreeNode):
    probability: Array = jnp.ones((1,), dtype=jnp.float32)  # FloatArray['b']
    """The probability of receiving the reward, if reached."""


class Consumable(struct.PyTreeNode):
    requires: Array = EMPTY_POCKET_ID[None]  # IntArray['b']
    """The id of the item required to consume this item. If set, it must be >= 1."""
    consumed: Array = jnp.zeros((1,), dtype=jnp.bool_)  # IntArray['b']
    """Whether the item has been consumed"""


class Pickable(struct.PyTreeNode):
    id: Array = jnp.ones((1,), dtype=jnp.int32)  # IntArray['b']
    """The id of the item. If set, it must be >= 1."""


class Holder(struct.PyTreeNode):
    pocket: Array = EMPTY_POCKET_ID  # IntArray['b']
    """The id of the item in the pocket (0 if empty)"""


class Entity(Component, Positionable, HasTag):
    """Entities are components that can be placed in the environment"""

    def get_sprite(self, registry: SpritesRegistry):
        raise NotImplementedError()


class Player(Entity, Directional, Holder):
    """Players are entities that can act around the environment"""

    @classmethod
    def create(cls, position: Array, direction: Array, tag: Array = jnp.asarray(1)) -> Player:
        return cls(
            entity_type=jnp.asarray(2),
            position=position,
            direction=direction,
            pocket=EMPTY_POCKET_ID,
            tag=tag,
        )

    def get_sprite(self, registry: SpritesRegistry):
        return registry[self.entity_type][self.direction]


class Goal(Entity, Stochastic):
    """Goals are entities that can be reached by the player"""

    @classmethod
    def create(cls, position: Array, tag: Array, probability: Array) -> Goal:
        return cls(
            entity_type=jnp.asarray(3),
            position=position,
            tag=tag,
            probability=probability,
        )

    def get_sprite(self, registry: SpritesRegistry):
        return registry[self.entity_type]

class Key(Entity, Pickable):
    """Pickable items are world objects that can be picked up by the player.
    Examples of pickable items are keys, coins, etc."""

    @classmethod
    def create(cls, position: Array, id: Array) -> Key:
        return cls(
            entity_type=jnp.asarray(4),
            position=position,
            tag=-id,
            id=id,
        )

    def get_sprite(self, registry: SpritesRegistry):
        return registry[self.entity_type]


class Door(Entity, Directional, Consumable):
    """Consumable items are world objects that can be consumed by the player.
    Consuming an item requires a tool (e.g. a key to open a door).
    A tool is an id (int) of another item, specified in the `requires` field (-1 if no tool is required).
    After an item is consumed, it is both removed from the `state.entities` collection, and replaced in the grid
    by the item specified in the `replacement` field (0 = floor by default).
    Examples of consumables are doors (to open) food (to eat) and water (to drink), etc.
    """

    @classmethod
    def create(
        cls,
        position: Array,
        direction: Array,
        requires: Array,
    ) -> Door:
        return cls(
            entity_type=jnp.asarray(5),
            position=position,
            direction=direction,
            requires=requires,
            tag=requires,
            consumed=jnp.zeros((1,), dtype=jnp.int32),
        )

    def get_sprite(self, registry: SpritesRegistry):
        return registry[self.entity_type][self.consumed][self.direction]


class State(struct.PyTreeNode):
    """The Markovian state of the environment"""

    key: KeyArray
    """The random number generator state"""
    grid: Array
    """The base map of the environment that remains constant throughout the training"""
    entities: Tuple[Entity, ...]
    """The entities in the environment as a tuple of Entity"""
    cache: RenderingCache
    """The rendering cache to speed up rendering"""

    def get_positions(self, axis: int = -1) -> Array:
        return jnp.stack([entity.position for entity in self.entities], axis=axis)

    def get_tags(self, axis: int = -1) -> Array:
        return jnp.stack([entity.tag for entity in self.entities], axis=axis)

    def get_sprites(self, sprites_registry: SpritesRegistry, axis: int = 0) -> Array:
        return jnp.stack([entity.get_sprite(sprites_registry) for entity in self.entities], axis=axis)
