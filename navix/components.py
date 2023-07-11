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
from flax import struct
import jax
import jax.numpy as jnp


DISCARD_PILE_COORDS = jnp.asarray((0, -1), dtype=jnp.int32)
DISCARD_PILE_IDX = jnp.asarray(-1, dtype=jnp.int32)
EMPTY_POCKET_ID = jnp.asarray(-1, dtype=jnp.int32)
UNSET_DIRECTION = jnp.asarray(-1, dtype=jnp.int32)
UNSET_CONSUMED = jnp.asarray(-1, dtype=jnp.int32)


class Component(struct.PyTreeNode):
    _disable_batching: bool = struct.field(pytree_node=False, default=False, repr=False)

    def __post_init__(self) -> None:
        # this stops the super().__post_init__() stream from reaching parent classes
        # and it's the canonical pattern for __post_init__ in composable dataclasses
        # see https://bugs.python.org/issue46757
        return


class Positionable(Component):
    position: Array = DISCARD_PILE_COORDS
    """The (row, column) position of the entity in the grid, defaults to the discard pile (-1, -1)"""

    def __post_init__(self) -> None:
        if self.position.ndim < 2 and not self._disable_batching:
            object.__setattr__(self, "position", self.position[None])
        return super().__post_init__()


class Directional(Component):
    direction: Array = jnp.asarray(0, dtype=jnp.int32)
    """The direction the entity: 0 = east, 1 = south, 2 = west, 3 = north"""

    def __post_init__(self) -> None:
        if self.direction.ndim < 1 and not self._disable_batching:
            object.__setattr__(self, "direction", self.direction[None])
        return super().__post_init__()


class HasTag(Component):
    tag: Array = jnp.asarray(0, dtype=jnp.int32)
    """The tag of the component, used to identify the type of the component in `oobservations.categorical`"""

    def __post_init__(self) -> None:
        if self.tag.ndim < 1 and not self._disable_batching:
            object.__setattr__(self, "tag", self.tag[None])
        return super().__post_init__()


class Stochastic(Component):
    probability: Array = jnp.asarray(1.0, dtype=jnp.float32)
    """The probability of receiving the reward, if reached."""

    def __post_init__(self) -> None:
        if self.probability.ndim < 1 and not self._disable_batching:
            object.__setattr__(self, "probability", self.probability[None])
        return super().__post_init__()


class Openable(Component):
    requires: Array = EMPTY_POCKET_ID
    """The id of the item required to consume this item. If set, it must be >= 1."""
    open: Array = jnp.asarray(False, dtype=jnp.bool_)
    """Whether the item is open or not."""

    def __post_init__(self) -> None:
        if self.requires.ndim < 1 and not self._disable_batching:
            object.__setattr__(self, "requires", self.requires[None])
        if self.open.ndim < 1 and not self._disable_batching:
            object.__setattr__(self, "open", self.open[None])
        return super().__post_init__()


class Pickable(Component):
    id: Array = jnp.asarray(1, dtype=jnp.int32)
    """The id of the item. If set, it must be >= 1."""

    def __post_init__(self) -> None:
        if self.id.ndim < 1 and not self._disable_batching:
            object.__setattr__(self, "id", self.id[None])
        return super().__post_init__()


class Holder(Component):
    pocket: Array = EMPTY_POCKET_ID
    """The id of the item in the pocket (0 if empty)"""

    def __post_init__(self) -> None:
        if self.pocket.ndim < 1 and not self._disable_batching:
            object.__setattr__(self, "pocket", self.pocket[None])
        return super().__post_init__()


class HasSprite(Component):
    @property
    def sprite(self) -> Array:
        raise NotImplementedError()
