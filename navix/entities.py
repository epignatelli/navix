from __future__ import annotations
from typing import Dict, TypeVar

import dataclasses
import jax
from jax import Array
import jax.numpy as jnp
from flax import struct
from jax.random import KeyArray
from enum import Enum

from .components import Positionable, Directional, HasTag, Stochastic, Openable, Pickable, Holder, HasSprite, EMPTY_POCKET_ID, DISCARD_PILE_COORDS
from .graphics import RenderingCache, SPRITES_REGISTRY


T = TypeVar('T', bound='Entity')


class Entities(Enum):
    WALL = "wall"
    FLOOR = "floor"
    PLAYER = "player"
    GOAL = "goal"
    KEY = "key"
    DOOR = "door"


class Entity(Positionable, HasTag, HasSprite):
    """Entities are components that can be placed in the environment"""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self._disable_batching:
            return
        # make sure that all components have the same batch size
        attrs = jax.tree_util.tree_leaves(self)
        batch_size = max([field.shape[0] for field in attrs])
        for field in dataclasses.fields(self):
            if field.metadata.get("pytree_node", True):
                value = getattr(self, field.name)
                new_value = jnp.broadcast_to(value, (batch_size, *value.shape[1:]))
                object.__setattr__(self, field.name, new_value)

    def __getitem__(self: T, idx) -> T:
        # this will always return the object with properties of with at least rank = 1
        # entity = self.replace(_disable_batching=True)
        object.__setattr__(self, "_disable_batching", True)
        entity = jax.tree_util.tree_map(lambda attr: attr[idx], self)
        object.__setattr__(entity, "_disable_batching", False)
        return entity

    @property
    def walkable(self) -> Array:
        raise NotImplementedError()

    @property
    def transparent(self) -> Array:
        raise NotImplementedError()


class Wall(Entity):
    """Walls are entities that cannot be walked through"""

    @property
    def walkable(self) -> Array:
        return jnp.broadcast_to(jnp.asarray(False), self.position.shape)

    @property
    def transparent(self) -> Array:
        return jnp.broadcast_to(jnp.asarray(False), self.position.shape)

    @property
    def sprite(self) -> Array:
        sprite = SPRITES_REGISTRY[Entities.WALL.value]
        return jnp.broadcast_to(self.sprite[None], (self.position.shape[0], *sprite.shape))


class Player(Entity, Directional, Holder):
    """Players are entities that can act around the environment"""

    @property
    def walkable(self) -> Array:
        return jnp.broadcast_to(jnp.asarray(True), self.direction.shape)

    @property
    def transparent(self) -> Array:
        return jnp.broadcast_to(jnp.asarray(True), self.direction.shape)

    @property
    def sprite(self) -> Array:
        sprite = SPRITES_REGISTRY[Entities.PLAYER.value][self.direction]
        if sprite.ndim == 3:
            # batch it
            sprite = sprite[None]
        # broadcast to batch_size
        return jnp.broadcast_to(sprite, (self.position.shape[0], *sprite.shape[1:]))


class Goal(Entity, Stochastic):
    """Goals are entities that can be reached by the player"""

    @property
    def walkable(self) -> Array:
        return jnp.broadcast_to(jnp.asarray(True), self.probability.shape)

    @property
    def transparent(self) -> Array:
        return jnp.broadcast_to(jnp.asarray(True), self.probability.shape)

    @property
    def sprite(self) -> Array:
        sprite = SPRITES_REGISTRY[Entities.GOAL.value]
        if sprite.ndim == 3:
            # batch it
            sprite = sprite[None]
        # ensure same batch size
        if sprite.shape[0] != self.position.shape[0]:
            sprite = jnp.broadcast_to(sprite, (self.position.shape[0], *sprite.shape[1:]))
        return sprite


class Key(Entity, Pickable):
    """Pickable items are world objects that can be picked up by the player.
    Examples of pickable items are keys, coins, etc."""

    @property
    def walkable(self) -> Array:
        return jnp.broadcast_to(jnp.asarray(False), self.id.shape)

    @property
    def transparent(self) -> Array:
        return jnp.broadcast_to(jnp.asarray(True), self.id.shape)

    @property
    def sprite(self) -> Array:
        sprite = SPRITES_REGISTRY[Entities.KEY.value]
        if sprite.ndim == 3:
            # batch it
            sprite = sprite[None]
        # ensure same batch size
        if sprite.shape[0] != self.position.shape[0]:
            sprite = jnp.broadcast_to(sprite, (self.position.shape[0], *sprite.shape[1:]))
        return sprite


class Door(Entity, Directional, Openable):
    """Consumable items are world objects that can be consumed by the player.
    Consuming an item requires a tool (e.g. a key to open a door).
    A tool is an id (int) of another item, specified in the `requires` field (-1 if no tool is required).
    After an item is consumed, it is both removed from the `state.entities` collection, and replaced in the grid
    by the item specified in the `replacement` field (0 = floor by default).
    Examples of consumables are doors (to open) food (to eat) and water (to drink), etc.
    """

    @property
    def walkable(self) -> Array:
        return self.open

    @property
    def transparent(self) -> Array:
        return self.open

    @property
    def sprite(self) -> Array:
        sprite = SPRITES_REGISTRY[Entities.DOOR.value][self.direction, self.open]
        if sprite.ndim == 3:
            # batch it
            sprite = sprite[None]
        # ensure same batch size
        if sprite.shape[0] != self.position.shape[0]:
            sprite = jnp.broadcast_to(sprite, (self.position.shape[0], *sprite.shape[1:]))
        return sprite


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
        self.entities[Entities.PLAYER.value] = player
        return self

    def get_goals(self) -> Goal:
        return self.entities.get(Entities.GOAL.value, Goal())  # type: ignore

    def set_goals(self, goals: Goal) -> State:
        self.entities[Entities.GOAL.value] = goals
        return self

    def get_keys(self) -> Key:
        return self.entities.get(Entities.KEY.value, Key())  # type: ignore

    def set_keys(self, keys: Key) -> State:
        self.entities[Entities.KEY.value] = keys
        return self

    def get_doors(self) -> Door:
        return self.entities.get(Entities.DOOR.value, Door())  # type: ignore

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
