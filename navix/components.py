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

from jax import Array
from flax import struct
import jax.numpy as jnp


DISCARD_PILE_COORDS = jnp.asarray((0, -1), dtype=jnp.int32)
DISCARD_PILE_IDX = jnp.asarray(-1, dtype=jnp.int32)
EMPTY_POCKET_ID = jnp.asarray(-1, dtype=jnp.int32)
UNSET_DIRECTION = jnp.asarray(-1, dtype=jnp.int32)
UNSET_CONSUMED = jnp.asarray(-1, dtype=jnp.int32)


class EntityType(IntEnum):
    WALL = 0
    FLOOR = 1
    PLAYER = 2
    GOAL = 3
    KEY = 4
    DOOR = 5


class Component(struct.PyTreeNode):
    entity_type: Array = jnp.asarray(0, dtype=jnp.int32)
    """The type of the entity, 0 = player, 1 = goal, 2 = key, 3 = door"""


class Positionable(struct.PyTreeNode):
    position: Array = DISCARD_PILE_COORDS
    """The (row, column) position of the entity in the grid, defaults to the discard pile (-1, -1)"""


class Directional(struct.PyTreeNode):
    direction: Array = jnp.asarray(0, dtype=jnp.int32)
    """The direction the entity: 0 = east, 1 = south, 2 = west, 3 = north"""


class HasTag(struct.PyTreeNode):
    tag: Array = jnp.asarray(0, dtype=jnp.int32)
    """The tag of the component, used to identify the type of the component in `oobservations.categorical`"""


class Stochastic(struct.PyTreeNode):
    probability: Array = jnp.asarray(1.0, dtype=jnp.float32)
    """The probability of receiving the reward, if reached."""


class Openable(struct.PyTreeNode):
    requires: Array = EMPTY_POCKET_ID
    """The id of the item required to consume this item. If set, it must be >= 1."""
    open: Array = jnp.asarray(False, dtype=jnp.bool_)
    """Whether the item is open or not."""


class Pickable(struct.PyTreeNode):
    id: Array = jnp.asarray(1, dtype=jnp.int32)
    """The id of the item. If set, it must be >= 1."""


class Holder(struct.PyTreeNode):
    pocket: Array = EMPTY_POCKET_ID
    """The id of the item in the pocket (0 if empty)"""


class HasSprite(struct.PyTreeNode):
    sprite: Array = jnp.asarray(0, dtype=jnp.int32)
    """The id of the sprite of the entity."""