from __future__ import annotations
from typing import Tuple, TypeVar

import jax
from jax import Array
import jax.numpy as jnp
from flax import struct


from .components import (
    Positionable,
    Directional,
    HasColour,
    HasTag,
    Stochastic,
    Openable,
    Pickable,
    Holder,
    HasSprite,
)
from .rendering.registry import SPRITES_REGISTRY

T = TypeVar("T", bound="Entity")


class Entities(struct.PyTreeNode):
    WALL: str = struct.field(pytree_node=False, default="wall")
    FLOOR: str = struct.field(pytree_node=False, default="floor")
    PLAYER: str = struct.field(pytree_node=False, default="player")
    GOAL: str = struct.field(pytree_node=False, default="goal")
    KEY: str = struct.field(pytree_node=False, default="key")
    DOOR: str = struct.field(pytree_node=False, default="door")
    LAVA: str = struct.field(pytree_node=False, default="lava")
    BALL: str = struct.field(pytree_node=False, default="ball")
    BOX: str = struct.field(pytree_node=False, default="box")


class EntityIds:
    UNKNOWN: Array = jnp.asarray(0, dtype=jnp.uint8)
    FLOOR: Array = jnp.asarray(1, dtype=jnp.uint8)
    WALL: Array = jnp.asarray(2, dtype=jnp.uint8)
    DOOR: Array = jnp.asarray(4, dtype=jnp.uint8)
    KEY: Array = jnp.asarray(5, dtype=jnp.uint8)
    BALL: Array = jnp.asarray(6, dtype=jnp.uint8)
    BOX: Array = jnp.asarray(7, dtype=jnp.uint8)
    GOAL: Array = jnp.asarray(8, dtype=jnp.uint8)
    LAVA: Array = jnp.asarray(9, dtype=jnp.uint8)
    PLAYER: Array = jnp.asarray(10, dtype=jnp.uint8)


class Directions:
    EAST = jnp.asarray(0)
    SOUTH = jnp.asarray(1)
    WEST = jnp.asarray(2)
    NORTH = jnp.asarray(3)


class Entity(Positionable, HasTag, HasSprite):
    """Entities are components that can be placed in the environment"""

    def __getitem__(self: T, idx) -> T:
        return jax.tree_util.tree_map(lambda x: x[idx], self)

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def shape(self) -> Tuple[int, ...]:
        """The batch shape of the entity"""
        return self.position.shape[:-1]

    @property
    def ndim(self) -> int:
        return self.position.ndim - 1

    @property
    def walkable(self) -> Array:
        raise NotImplementedError()

    @property
    def transparent(self) -> Array:
        raise NotImplementedError()


class Wall(Entity, HasColour):
    """Walls are entities that cannot be walked through"""

    @classmethod
    def create(
        cls,
        position: Array,
    ) -> Wall:
        shape = position.shape[:-1]
        grey = jnp.ones(shape, dtype=jnp.uint8) * 5
        return cls(position=position, colour=grey)

    @property
    def walkable(self) -> Array:
        return jnp.broadcast_to(jnp.asarray(False), self.shape)

    @property
    def transparent(self) -> Array:
        return jnp.broadcast_to(jnp.asarray(False), self.shape)

    @property
    def sprite(self) -> Array:
        sprite = SPRITES_REGISTRY[Entities.WALL]
        return jnp.broadcast_to(sprite[None], (*self.shape, *sprite.shape))

    @property
    def tag(self) -> Array:
        return jnp.broadcast_to(EntityIds.WALL, self.shape)


class Player(Entity, Directional, Holder):
    """Players are entities that can act around the environment"""

    @classmethod
    def create(
        cls,
        position: Array,
        direction: Array,
        pocket: Array,
    ) -> Player:
        return cls(position=position, direction=direction, pocket=pocket)

    @property
    def walkable(self) -> Array:
        return jnp.broadcast_to(jnp.asarray(True), self.shape)

    @property
    def transparent(self) -> Array:
        return jnp.broadcast_to(jnp.asarray(True), self.shape)

    @property
    def sprite(self) -> Array:
        sprite = SPRITES_REGISTRY[Entities.PLAYER][self.direction]
        if sprite.ndim == 3:
            # batch it
            sprite = sprite[None]
        # broadcast to batch_size
        return jnp.broadcast_to(sprite, (*self.shape, *sprite.shape[1:]))

    @property
    def tag(self) -> Array:
        return jnp.broadcast_to(EntityIds.PLAYER, self.shape)


class Goal(Entity, HasColour, Stochastic):
    """Goals are entities that can be reached by the player"""

    @classmethod
    def create(
        cls,
        position: Array,
        probability: Array,
    ) -> Goal:
        shape = position.shape[:-1]
        green = jnp.ones(shape, dtype=jnp.uint8)
        return cls(position=position, probability=probability, colour=green)

    @property
    def walkable(self) -> Array:
        return jnp.broadcast_to(jnp.asarray(True), self.shape)

    @property
    def transparent(self) -> Array:
        return jnp.broadcast_to(jnp.asarray(True), self.shape)

    @property
    def sprite(self) -> Array:
        sprite = SPRITES_REGISTRY[Entities.GOAL]
        if sprite.ndim == 3:
            # batch it
            sprite = sprite[None]
        # ensure same batch size
        if sprite.shape[0] != self.position.shape[0]:
            sprite = jnp.broadcast_to(sprite, (*self.shape, *sprite.shape[1:]))
        return sprite

    @property
    def tag(self) -> Array:
        return jnp.broadcast_to(EntityIds.GOAL, self.shape)


class Key(Entity, Pickable, HasColour):
    """Pickable items are world objects that can be picked up by the player.
    Examples of pickable items are keys, coins, etc."""

    @classmethod
    def create(
        cls,
        position: Array,
        colour: Array,
        id: Array,
    ) -> Key:
        colour = jnp.asarray(colour, dtype=jnp.uint8)
        return cls(position=position, id=id, colour=colour)

    @property
    def walkable(self) -> Array:
        return jnp.broadcast_to(jnp.asarray(False), self.shape)

    @property
    def transparent(self) -> Array:
        return jnp.broadcast_to(jnp.asarray(True), self.shape)

    @property
    def sprite(self) -> Array:
        sprite = SPRITES_REGISTRY[Entities.KEY][self.colour]
        if sprite.ndim == 3:
            # batch it
            sprite = sprite[None]
        # ensure same batch size
        if sprite.shape[0] != self.position.shape[0]:
            sprite = jnp.broadcast_to(sprite, (*self.shape, *sprite.shape[1:]))
        return sprite

    @property
    def tag(self) -> Array:
        return jnp.broadcast_to(EntityIds.KEY, self.shape)


class Door(Entity, Openable, HasColour):
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
        requires: Array,
        colour: Array,
        open: Array,
    ) -> Door:
        colour = jnp.asarray(colour, dtype=jnp.uint8)
        return cls(
            position=position,
            requires=requires,
            open=open,
            colour=colour,
        )

    @property
    def walkable(self) -> Array:
        return self.open

    @property
    def transparent(self) -> Array:
        return self.open

    @property
    def sprite(self) -> Array:
        sprite = SPRITES_REGISTRY[Entities.DOOR][
            self.colour, jnp.asarray(self.open + 2 * self.locked, dtype=jnp.int32)
        ]
        if sprite.ndim == 3:
            # batch it
            sprite = sprite[None]
        # ensure same batch size
        if sprite.shape[0] != self.position.shape[0]:
            sprite = jnp.broadcast_to(sprite, (*self.shape, *sprite.shape[1:]))
        return sprite

    @property
    def tag(self) -> Array:
        return jnp.broadcast_to(EntityIds.DOOR, self.shape)

    @property
    def locked(self) -> Array:
        return self.requires != jnp.asarray(-1)


class Lava(Entity):
    """Goals are entities that can be reached by the player"""

    @classmethod
    def create(
        cls,
        position: Array,
    ) -> Lava:
        return cls(position=position)

    @property
    def walkable(self) -> Array:
        return jnp.broadcast_to(jnp.asarray(True), self.shape)

    @property
    def transparent(self) -> Array:
        return jnp.broadcast_to(jnp.asarray(True), self.shape)

    @property
    def sprite(self) -> Array:
        sprite = SPRITES_REGISTRY[Entities.LAVA]
        if sprite.ndim == 3:
            # batch it
            sprite = sprite[None]
        # ensure same batch size
        if sprite.shape[0] != self.position.shape[0]:
            sprite = jnp.broadcast_to(sprite, (*self.shape, *sprite.shape[1:]))
        return sprite

    @property
    def tag(self) -> Array:
        return jnp.broadcast_to(EntityIds.LAVA, self.shape)


class Ball(Entity, HasColour, Stochastic):
    """Goals are entities that can be reached by the player"""

    @classmethod
    def create(
        cls,
        position: Array,
        colour: Array,
        probability: Array,
    ) -> Ball:
        return cls(position=position, colour=colour, probability=probability)

    @property
    def walkable(self) -> Array:
        return jnp.broadcast_to(jnp.asarray(False), self.shape)

    @property
    def transparent(self) -> Array:
        return jnp.broadcast_to(jnp.asarray(True), self.shape)

    @property
    def sprite(self) -> Array:
        sprite = SPRITES_REGISTRY[Entities.BALL][self.colour]
        if sprite.ndim == 3:
            # batch it
            sprite = sprite[None]
        # ensure same batch size
        if sprite.shape[0] != self.position.shape[0]:
            sprite = jnp.broadcast_to(sprite, (*self.shape, *sprite.shape[1:]))
        return sprite

    @property
    def tag(self) -> Array:
        return jnp.broadcast_to(EntityIds.BALL, self.shape)


class Box(Entity, HasColour, Holder):
    """Goals are entities that can be reached by the player"""

    @classmethod
    def create(
        cls,
        position: Array,
        colour: Array,
        pocket: Array,
    ) -> Box:
        return cls(position=position, colour=colour, pocket=pocket)

    @property
    def walkable(self) -> Array:
        return jnp.broadcast_to(jnp.asarray(False), self.shape)

    @property
    def transparent(self) -> Array:
        return jnp.broadcast_to(jnp.asarray(True), self.shape)

    @property
    def sprite(self) -> Array:
        sprite = SPRITES_REGISTRY[Entities.BOX][self.colour]
        if sprite.ndim == 3:
            # batch it
            sprite = sprite[None]
        # ensure same batch size
        if sprite.shape[0] != self.position.shape[0]:
            sprite = jnp.broadcast_to(sprite, (*self.shape, *sprite.shape[1:]))
        return sprite

    @property
    def tag(self) -> Array:
        return jnp.broadcast_to(EntityIds.BOX, self.shape)
