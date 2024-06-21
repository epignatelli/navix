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
from typing import Tuple


from jax import Array
from flax import struct
import jax.numpy as jnp
import dataclasses


DISCARD_PILE_COORDS = jnp.asarray((0, -1), dtype=jnp.int32)
DISCARD_PILE_IDX = jnp.asarray(-1, dtype=jnp.int32)
EMPTY_POCKET_ID = jnp.asarray(-1, dtype=jnp.int32)
UNSET_DIRECTION = jnp.asarray(-1, dtype=jnp.int32)
UNSET_CONSUMED = jnp.asarray(-1, dtype=jnp.int32)


def field(shape: Tuple[int, ...], **kwargs):
    return dataclasses.field(metadata={"shape": shape}, **kwargs)


class Component(struct.PyTreeNode):
    """Base class for all components in the game.
    Components are used to store the data of the entities in the game."""

    def check_ndim(self, batched: bool = False) -> None:
        return


class Positionable(Component):
    """Flags an entity as positionable in the grid, and provides the `position` attribute"""

    position: Array = field(shape=(2,))
    """The (row, column) position of the entity in the grid as a JAX array, defaults to the discard pile (-1, -1)"""


class Directional(Component):
    """Flags an entity as directional, and provides the `direction` attribute"""

    direction: Array = field(shape=())
    """The direction the entity: 0 = east, 1 = south, 2 = west, 3 = north"""


class HasColour(Component):
    """Flags an entity as having a colour, and provides the `colour` attribute"""

    colour: Array = field(shape=())
    """The colour of the object for rendering. """


class Stochastic(Component):
    """Flags an entity as stochastic, and provides the `probability` attribute

    TODO:
        * consider replace probability (Array) with a distrax.Distribution

    """

    probability: Array = field(shape=())
    """The probability of receiving the reward, if reached."""


class Openable(Component):
    """Flags an entity as openable, and provides the `requires` and `open` attributes"""

    requires: Array = field(shape=())
    """The id of the item required to consume this item. If set, it must be > 0.
    If -1, the door is unlocked and does not require any key to open."""
    open: Array = field(shape=())
    """Open is jnp.asarray(0) if the entity is closed and 1 if open."""


class Pickable(Component):
    """Flags an entity as pickable, and provides the `id` attribute, which is used to identify the item in the inventory"""

    id: Array = field(shape=())
    """The id of the item. If set, it must be >= 1."""


class Holder(Component):
    """Flags an entity as a holder, and provides the `pocket` attribute. The pocket is used to store the id of the item in the pocket."""

    pocket: Array = field(shape=())
    """The id of the item in the pocket (0 if empty)"""


class HasTag(Component):
    """Flags an entity as having a tag, and provides the `tag` attribute. The tag is used to identify the type of the entity in the observations."""

    @property
    def tag(self) -> Array:
        """The tag of the component, used to identify the type of the component in `observations.categorical`"""
        raise NotImplementedError()


class HasSprite(Component):
    """Flags an entity as having a sprite, and provides the `sprite` attribute. The sprite is used to render the entity in the game."""

    @property
    def sprite(self) -> Array:
        raise NotImplementedError()
