from __future__ import annotations
from typing import Dict, Tuple

import jax
from jax import Array
import jax.numpy as jnp
from flax import struct
from jax.random import KeyArray

from .components import Component, Positionable, Directional, HasTag, Stochastic, Openable, Pickable, Holder, EMPTY_POCKET_ID, DISCARD_PILE_COORDS
from .graphics import RenderingCache


def ensure_batched(x: Array, unbached_dims: int) -> Array:
    if x.ndim <= unbached_dims:
        return x[None]
    return x


class Entity(Component, Positionable, HasTag):
    """Entities are components that can be placed in the environment"""

    def __getitem__(self, idx) -> Entity:
        return self.__class__(jax.tree_util.tree_map(lambda attr: attr[idx], self))

    @property
    def walkable(self) -> Array:
        return jnp.broadcast_to(jnp.asarray(False), self.position.shape)

    @property
    def transparent(self) -> Array:
        return jnp.broadcast_to(jnp.asarray(False), self.position.shape)

    def get_sprite(self, registry: Array) -> Array:
        return registry[self.entity_type, 0, 0]


class Wall(Entity):
    """Walls are entities that cannot be walked through"""

    @classmethod
    def create(cls, position: Array = DISCARD_PILE_COORDS[None]) -> Wall:
        return cls(
            entity_type=jnp.asarray(-1, dtype=jnp.int32),
            position=position,
        )

    @property
    def walkable(self) -> Array:
        return jnp.broadcast_to(jnp.asarray(False), self.position.shape)

    @property
    def transparent(self) -> Array:
        return jnp.broadcast_to(jnp.asarray(False), self.position.shape)

    def get_sprite(self, registry: Array) -> Array:
        return registry[self.entity_type, 0, 0]


class Player(Entity, Directional, Holder):
    """Players are entities that can act around the environment"""

    @classmethod
    def create(cls, position: Array = DISCARD_PILE_COORDS[None], direction: Array = jnp.asarray(0)[None], tag: Array = jnp.asarray(1)[None]) -> Player:
        return cls(
            entity_type=jnp.broadcast_to(jnp.asarray(2), direction.shape),
            position=position,
            direction=direction,
            pocket=jnp.broadcast_to(EMPTY_POCKET_ID, direction.shape),
            tag=tag,
        )

    @property
    def walkable(self) -> Array:
        return jnp.broadcast_to(jnp.asarray(True), self.direction.shape)

    @property
    def transparent(self) -> Array:
        return jnp.broadcast_to(jnp.asarray(True), self.direction.shape)

    def get_sprite(self, registry: Array) -> Array:
        return registry[self.entity_type, self.direction, 0]


class Goal(Entity, Stochastic):
    """Goals are entities that can be reached by the player"""

    @classmethod
    def create(cls, position: Array = DISCARD_PILE_COORDS[None], probability: Array = jnp.asarray((1.0,)), tag: Array = jnp.asarray((2,))) -> Goal:
        # ensure that the inputs are batched
        position = ensure_batched(position, 1)
        probability = ensure_batched(probability, 0)
        tag = ensure_batched(tag, 0)

        # check that the batch sizes are the same
        assert len(position) == len(probability) == len(tag)

        return cls(
            entity_type=jnp.broadcast_to(jnp.asarray(3), probability.shape),
            position=ensure_batched(position, 1),
            tag=ensure_batched(tag, 0),
            probability=ensure_batched(probability, 0),
        )

    @property
    def walkable(self) -> Array:
        return jnp.broadcast_to(jnp.asarray(True), self.probability.shape)

    @property
    def transparent(self) -> Array:
        return jnp.broadcast_to(jnp.asarray(True), self.probability.shape)

    def get_sprite(self, registry: Array) -> Array:
        return registry[self.entity_type, 0, 0]


class Key(Entity, Pickable):
    """Pickable items are world objects that can be picked up by the player.
    Examples of pickable items are keys, coins, etc."""

    @classmethod
    def create(cls, position: Array =  DISCARD_PILE_COORDS[None], id: Array = jnp.asarray((3,))) -> Key:
        # ensure that the inputs are batched
        position = ensure_batched(position, 1)
        id = ensure_batched(id, 0)

        # check that the batch sizes are the same
        assert len(position) == len(id)

        return cls(
            entity_type=jnp.broadcast_to(jnp.asarray(4), id.shape),
            position=position,
            tag=-id,
            id=id,
        )

    @property
    def walkable(self) -> Array:
        return jnp.broadcast_to(jnp.asarray(False), self.id.shape)

    @property
    def transparent(self) -> Array:
        return jnp.broadcast_to(jnp.asarray(True), self.id.shape)

    def get_sprite(self, registry: Array) -> Array:
        return registry[self.entity_type, 0, 0]


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
        position: Array =  DISCARD_PILE_COORDS[None],
        direction: Array = jnp.asarray((0,)),
        requires: Array = jnp.asarray((3,)),
    ) -> Door:
        # ensure that the inputs are batched
        position = ensure_batched(position, 1)
        direction = ensure_batched(direction, 0)
        requires = ensure_batched(requires, 0)

        # check that the batch sizes are the same
        assert len(position) == len(direction) == len(requires)
        return cls(
            entity_type=jnp.broadcast_to(jnp.asarray(5), direction.shape),
            position=position,
            direction=direction,
            requires=requires,
            tag=requires,
            open=jnp.broadcast_to(jnp.asarray(False), direction.shape),
        )


    @property
    def walkable(self) -> Array:
        return self.open

    @property
    def transparent(self) -> Array:
        return self.open

    def get_sprite(self, registry: Array) -> Array:
        open = jnp.asarray(self.open, dtype=jnp.int32)
        return registry[self.entity_type, self.direction, open]


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

    def get_player(self, idx: int = 0) -> Player:
        return self.entities["player"]  # type: ignore
        # return jax.tree_util.tree_map(lambda attr: attr[idx], self.entities["player"])

    def get_goals(self) -> Goal:
        return self.entities.get("goal", Goal.create())  # type: ignore

    def get_keys(self) -> Key:
        return self.entities.get("key", Key.create())  # type: ignore

    def get_doors(self) -> Door:
        return self.entities.get("door", Door.create())  # type: ignore

    def get_positions(self) -> Array:
        return jnp.concatenate([self.entities[k].position for k in self.entities])

    def get_tags(self) -> Array:
        return jnp.concatenate([self.entities[k].tag for k in self.entities])

    def get_sprites(self, sprites_registry: Array) -> Array:
        return jnp.concatenate([self.entities[k].get_sprite(sprites_registry) for k in self.entities])

    def get_transparency(self) -> Array:
        return jnp.concatenate([self.entities[k].transparent for k in self.entities])
