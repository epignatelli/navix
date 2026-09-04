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


"""The reusable "capability" mixins that `navix.entities` compose into
concrete entities.

Each `Component` subclass adds one field (or one abstract property) plus
its semantics: `Positionable` -> `position`, `Directional` ->
`direction`, `Openable` -> `requires`/`open`, and so on. An entity like
`Door` is just `Positionable + Directional + Openable + HasColour + ...`.
Every field is a JAX array and the leading axis is the entity-instance
axis, so a whole batch of doors is one `Door` struct.
"""

from __future__ import annotations
from typing import Tuple


from jax import Array
from flax import struct
import jax.numpy as jnp
import dataclasses


DISCARD_PILE_COORDS = jnp.asarray((0, -1), dtype=jnp.int32)
"""The off-grid `(row, col)` an entity's `position` is set to once it has
been picked up / consumed. Column `-1` is outside every grid, so such an
entity renders nowhere and matches no real cell."""
DISCARD_PILE_IDX = jnp.asarray(-1, dtype=jnp.int32)
"""The flat patch index that `DISCARD_PILE_COORDS` maps to; rendering
slices it off (`patches[:DISCARD_PILE_IDX]`)."""
EMPTY_POCKET_ID = jnp.asarray(-1, dtype=jnp.int32)
"""`Holder.pocket` / `Player.pocket` value meaning "carrying nothing"."""
UNSET_DIRECTION = jnp.asarray(-1, dtype=jnp.int32)
"""`Directional.direction` sentinel for an entity whose facing is
irrelevant (it is never used as a rotation)."""
UNSET_CONSUMED = jnp.asarray(-1, dtype=jnp.int32)
"""Sentinel for a "consumed key" slot that has not been used yet."""


def field(shape: Tuple[int, ...], **kwargs):
    """A `dataclasses.field` that also records the per-instance `shape`
    of the array (excluding the leading instance axis) in its metadata,
    so `Component.check_ndim` can validate batched vs unbatched structs.

    Args:
        shape (tuple[int, ...]): the shape of one instance's value -
            `()` for a scalar field, `(2,)` for a `(row, col)` position.
        **kwargs: forwarded to `dataclasses.field` (e.g. `default_factory`).

    Returns:
        dataclasses.Field: the field descriptor."""
    return dataclasses.field(metadata={"shape": shape}, **kwargs)


class Component(struct.PyTreeNode):
    """Base of every capability mixin. A `Component` is a frozen
    `flax.struct` pytree; concrete entities inherit from several at once.
    Carries no data itself."""

    def check_ndim(self, batched: bool = False) -> None:
        """Hook for asserting each field has the expected rank (one extra
        axis when `batched`). The base implementation does nothing;
        subclasses may override. Not called on the hot path.

        Args:
            batched (bool): whether an extra leading instance axis is
                expected."""
        return


class Positionable(Component):
    """Entity has a location on the grid."""

    position: Array = field(shape=(2,))
    """`(row, col)` as `i32[2]` (or `i32[n_instances, 2]` batched).
    `DISCARD_PILE_COORDS` (`(0, -1)`) once the entity has been picked up
    or consumed."""


class Directional(Component):
    """Entity has a facing direction."""

    direction: Array = field(shape=())
    """`i32[]` in `0..3`: `0` east, `1` south, `2` west, `3` north
    (clockwise). Used directly by `grid.translate` / `grid.rotate`."""


class HasColour(Component):
    """Entity is drawn/encoded in one of the palette colours."""

    colour: Array = field(shape=())
    """`i32[]` index into `navix.rendering.registry.PALETTE` (`0` red,
    `1` green, `2` blue, `3` purple, `4` yellow, `5` grey). This is the
    `colour` channel of a `symbolic` observation and picks the sprite
    variant for `rgb`."""


class Stochastic(Component):
    """Entity's reward (e.g. a `Goal`) is granted only with some
    probability when reached."""

    probability: Array = field(shape=())
    """`f32[]` in `[0, 1]` - the chance the reward fires on contact.
    `1.0` for a deterministic goal."""


class Openable(Component):
    """Entity (a `Door`) can be opened, possibly after unlocking."""

    requires: Array = field(shape=())
    """`i32[]` - the `Pickable.id` of the key that unlocks this door, or
    `-1` if it needs no key. Set to `-1` once the door has been
    unlocked."""
    open: Array = field(shape=())
    """`0` closed, `1` open. navix doors do not re-close, so this only
    ever goes `0 -> 1`."""


class Pickable(Component):
    """Entity can be picked up into a pocket (`Key`, `Ball`, `Box`)."""

    id: Array = field(shape=())
    """`i32[] >= 1` - the identity written to `player.pocket` when this
    entity is picked up, and matched against `Openable.requires`."""


class Holder(Component):
    """Entity can carry one `Pickable` in a pocket (the `Player`)."""

    pocket: Array = field(shape=())
    """`i32[]` - the `Pickable.id` currently carried, or `EMPTY_POCKET_ID`
    (`-1`) when empty."""


class HasTag(Component):
    """Entity has an integer tag identifying its *type* in observations."""

    @property
    def tag(self) -> Array:
        """`i32[]` - the value this entity's cells take in a `categorical`
        observation and the first channel of a `symbolic` one. Constant
        per entity type (see `entities.EntityIds`).

        Raises:
            NotImplementedError: on the base class."""
        raise NotImplementedError()


class HasSprite(Component):
    """Entity has an RGB sprite for `rgb` rendering."""

    @property
    def sprite(self) -> Array:
        """`u8[TILE_SIZE, TILE_SIZE, 3]` (or with leading instance /
        direction axes) - the tile drawn for this entity.

        Raises:
            NotImplementedError: on the base class."""
        raise NotImplementedError()
