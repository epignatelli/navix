from __future__ import annotations
import dataclasses
from typing import Dict, Tuple, Type, TypeVar

import jax
from jax import Array
import jax.numpy as jnp
from flax import struct
from jax.random import KeyArray
from jax_enums import Enumerable

from .components import (
    Positionable,
    Directional,
    HasTag,
    Stochastic,
    Openable,
    Pickable,
    Holder,
    HasSprite,
)
from .graphics import RenderingCache, SPRITES_REGISTRY
from .config import config

T = TypeVar("T", bound="Entity")


class Entities(Enumerable):
    WALL = "wall"
    FLOOR = "floor"
    PLAYER = "player"
    GOAL = "goal"
    KEY = "key"
    DOOR = "door"


class Entity(Positionable, HasTag, HasSprite):
    """Entities are components that can be placed in the environment"""

    def __post_init__(self) -> None:
        if not config.ARRAY_CHECKS_ENABLED:
            return
        # Check that all fields have the same batch size
        fields = self.__dataclass_fields__
        batch_size = self.shape[0:]
        for path, leaf in jax.tree_util.tree_leaves_with_path(self):
            name = path[0].name
            default_ndim = len(fields[name].metadata["shape"])
            prefix = int(default_ndim != leaf.ndim)
            leaf_batch_size = leaf.shape[:prefix]
            assert (
                leaf_batch_size == batch_size
            ), f"Expected {name} to have batch size {batch_size}, got {leaf_batch_size} instead"

    def check_ndim(self, batched: bool = False) -> None:
        if not config.ARRAY_CHECKS_ENABLED:
            return
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            default_ndim = len(field.metadata["shape"])
            assert (
                value.ndim == default_ndim + batched
            ), f"Expected {field.name} to have ndim {default_ndim - batched}, got {value.ndim} instead"

    def __getitem__(self: T, idx) -> T:
        return jax.tree_util.tree_map(lambda x: x[idx], self)

    @property
    def shape(self) -> Tuple[int, ...]:
        """The batch shape of the entity"""
        return self.position.shape[: self.position.ndim - 1]

    @property
    def walkable(self) -> Array:
        raise NotImplementedError()

    @property
    def transparent(self) -> Array:
        raise NotImplementedError()


class Wall(Entity):
    """Walls are entities that cannot be walked through"""

    @classmethod
    def create(
        cls,
        position: Array,
    ) -> Wall:
        return cls(position=position)

    @property
    def walkable(self) -> Array:
        return jnp.broadcast_to(jnp.asarray(False), self.shape)

    @property
    def transparent(self) -> Array:
        return jnp.broadcast_to(jnp.asarray(False), self.shape)

    @property
    def sprite(self) -> Array:
        sprite = SPRITES_REGISTRY[Entities.WALL.value]
        return jnp.broadcast_to(sprite[None], (*self.shape, *sprite.shape))

    @property
    def tag(self) -> Array:
        return jnp.broadcast_to(jnp.asarray(0), self.shape)


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
        sprite = SPRITES_REGISTRY[Entities.PLAYER.value][self.direction]
        if sprite.ndim == 3:
            # batch it
            sprite = sprite[None]
        # broadcast to batch_size
        return jnp.broadcast_to(sprite, (*self.shape, *sprite.shape[1:]))

    @property
    def tag(self) -> Array:
        return jnp.broadcast_to(jnp.asarray(2), self.shape)


class Goal(Entity, Stochastic):
    """Goals are entities that can be reached by the player"""

    @classmethod
    def create(
        cls,
        position: Array,
        probability: Array,
    ) -> Goal:
        return cls(position=position, probability=probability)

    @property
    def walkable(self) -> Array:
        return jnp.broadcast_to(jnp.asarray(True), self.shape)

    @property
    def transparent(self) -> Array:
        return jnp.broadcast_to(jnp.asarray(True), self.shape)

    @property
    def sprite(self) -> Array:
        sprite = SPRITES_REGISTRY[Entities.GOAL.value]
        if sprite.ndim == 3:
            # batch it
            sprite = sprite[None]
        # ensure same batch size
        if sprite.shape[0] != self.position.shape[0]:
            sprite = jnp.broadcast_to(sprite, (*self.shape, *sprite.shape[1:]))
        return sprite

    @property
    def tag(self) -> Array:
        return jnp.broadcast_to(jnp.asarray(3), self.shape)


class Key(Entity, Pickable):
    """Pickable items are world objects that can be picked up by the player.
    Examples of pickable items are keys, coins, etc."""

    @classmethod
    def create(
        cls,
        position: Array,
        id: Array,
    ) -> Key:
        return cls(position=position, id=id)

    @property
    def walkable(self) -> Array:
        return jnp.broadcast_to(jnp.asarray(False), self.shape)

    @property
    def transparent(self) -> Array:
        return jnp.broadcast_to(jnp.asarray(True), self.shape)

    @property
    def sprite(self) -> Array:
        sprite = SPRITES_REGISTRY[Entities.KEY.value]
        if sprite.ndim == 3:
            # batch it
            sprite = sprite[None]
        # ensure same batch size
        if sprite.shape[0] != self.position.shape[0]:
            sprite = jnp.broadcast_to(sprite, (*self.shape, *sprite.shape[1:]))
        return sprite

    @property
    def tag(self) -> Array:
        return jnp.broadcast_to(jnp.asarray(4), self.shape)


class Door(Entity, Directional, Openable):
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
        open: Array,
    ) -> Door:
        return cls(position=position, direction=direction, requires=requires, open=open)

    @property
    def walkable(self) -> Array:
        return self.open

    @property
    def transparent(self) -> Array:
        return self.open

    @property
    def sprite(self) -> Array:
        sprite = SPRITES_REGISTRY[Entities.DOOR.value][
            self.direction, jnp.asarray(self.open, dtype=jnp.int32)
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
        return jnp.broadcast_to(jnp.asarray(5), self.shape)


class State(struct.PyTreeNode):
    """The Markovian state of the environment"""

    key: KeyArray
    """The random number generator state"""
    grid: Array
    """The base map of the environment that remains constant throughout the training"""
    cache: RenderingCache
    """The rendering cache to speed up rendering"""
    entities: Dict[str, Entity] = struct.field(default_factory=dict)
    """The entities in the environment, indexed via entity type string representation.
    Batched over the number of entities for each type"""

    def get_entity(self, entity_enum: Entities) -> Entity:
        return self.entities[entity_enum.value]

    def set_entity(self, entity_enum: Entities, entity: Entity) -> State:
        self.entities[entity_enum.value] = entity
        return self

    def get_walls(self) -> Wall:
        return self.entities.get(Entities.WALL.value, Wall())  # type: ignore

    def set_walls(self, walls: Wall) -> State:
        self.entities[Entities.WALL.value] = walls
        return self

    def get_player(self, idx: int = 0) -> Player:
        return self.entities[Entities.PLAYER.value][idx]  # type: ignore

    def set_player(self, player: Player, idx: int = 0) -> State:
        # TODO(epignatelli): this is a hack and won't work in multi-agent settings
        self.entities[Entities.PLAYER.value] = player[None]
        return self

    def get_goals(self) -> Goal:
        return self.entities[Entities.GOAL.value]  # type: ignore

    def set_goals(self, goals: Goal) -> State:
        self.entities[Entities.GOAL.value] = goals
        return self

    def get_keys(self) -> Key:
        return self.entities[Entities.KEY.value]  # type: ignore

    def set_keys(self, keys: Key) -> State:
        self.entities[Entities.KEY.value] = keys
        return self

    def get_doors(self) -> Door:
        return self.entities[Entities.DOOR.value]  # type: ignore

    def set_doors(self, doors: Door) -> State:
        self.entities[Entities.DOOR.value] = doors
        return self

    def get_positions(self) -> Array:
        return jnp.concatenate([self.entities[k].position for k in self.entities])

    def get_tags(self) -> Array:
        return jnp.concatenate([self.entities[k].tag for k in self.entities])

    def get_sprites(self) -> Array:
        return jnp.concatenate([self.entities[k].sprite for k in self.entities])

    def get_transparency(self) -> Array:
        return jnp.concatenate([self.entities[k].transparent for k in self.entities])
