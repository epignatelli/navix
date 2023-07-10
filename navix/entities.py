from __future__ import annotations
from typing import Dict

import jax
from jax import Array
import jax.numpy as jnp
from flax import struct
from jax.random import KeyArray
from enum import Enum

from .components import Component, Positionable, Directional, HasTag, Stochastic, Openable, Pickable, Holder, HasSprite, EMPTY_POCKET_ID, DISCARD_PILE_COORDS
from .graphics import RenderingCache, SPRITES_REGISTRY


class Entities(Enum):
    WALL = "wall"
    FLOOR = "floor"
    PLAYER = "player"
    GOAL = "goal"
    KEY = "key"
    DOOR = "door"


def ensure_batched(x: Array, ndim_as_unbatched: int) -> Array:
    if x.ndim <= ndim_as_unbatched:
        return x[None]
    return x


class Entity(Component, Positionable, HasTag, HasSprite):
    """Entities are components that can be placed in the environment"""

    def __getitem__(self, idx) -> Entity:
        return jax.tree_util.tree_map(lambda attr: attr[idx], self)

    def batch_size(self) -> int:
        return self.position.shape[0]

    @property
    def walkable(self) -> Array:
        raise NotImplementedError()

    @property
    def transparent(self) -> Array:
        raise NotImplementedError()

    def get_sprite(self) -> Array:
        raise NotImplementedError()


class Wall(Entity):
    """Walls are entities that cannot be walked through"""

    @classmethod
    def create(cls, position: Array = DISCARD_PILE_COORDS[None]) -> Wall:
        assert position.ndim == 2
        batch_size = position.shape[0]

        entity_type = jnp.broadcast_to(jnp.asarray(-1), (batch_size,))
        sprite = SPRITES_REGISTRY["wall"]
        sprite = jnp.broadcast_to(sprite, (batch_size, *sprite.shape))
        return cls(
            entity_type=entity_type,
            position=position,
            sprite=sprite
        )

    @property
    def walkable(self) -> Array:
        return jnp.broadcast_to(jnp.asarray(False), self.position.shape)

    @property
    def transparent(self) -> Array:
        return jnp.broadcast_to(jnp.asarray(False), self.position.shape)

    def get_sprite(self) -> Array:
        sprite = SPRITES_REGISTRY["wall"]
        return jnp.broadcast_to(sprite, (self.batch_size, *sprite.shape))


class Player(Entity, Directional, Holder):
    """Players are entities that can act around the environment"""

    @classmethod
    def create(cls, position: Array = DISCARD_PILE_COORDS[None], direction: Array = jnp.asarray(0)[None], tag: Array = jnp.asarray(1)[None]) -> Player:
        # chech that all inputs are batched
        position = ensure_batched(position, 1)
        direction = ensure_batched(direction, 0)
        tag = ensure_batched(tag, 0)

        # ensure that the inputs are batched and have the same batch size
        assert len(position) == len(direction) == len(tag)

        batch_size = position.shape[0]
        entity_type = jnp.broadcast_to(jnp.asarray(2), (batch_size,))
        pocket = jnp.broadcast_to(EMPTY_POCKET_ID, (batch_size,))
        sprite = SPRITES_REGISTRY["player"][direction]
        return cls(
            entity_type=entity_type,
            position=position,
            direction=direction,
            pocket=pocket,
            tag=tag,
            sprite=sprite,
        )

    @property
    def walkable(self) -> Array:
        return jnp.broadcast_to(jnp.asarray(True), self.direction.shape)

    @property
    def transparent(self) -> Array:
        return jnp.broadcast_to(jnp.asarray(True), self.direction.shape)

    def get_sprite(self) -> Array:
        return SPRITES_REGISTRY["player"][self.direction]

    # this is a patch to type annotation issues
    # If we do not override this, the type checker will complain that
    # the return type is not a `Player``, but an `Entity`
    def __getitem__(self, idx) -> Player:
        return jax.tree_util.tree_map(lambda attr: attr[idx], self)


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

        batch_size = position.shape[0]
        entity_type = jnp.broadcast_to(jnp.asarray(3), (batch_size,))

        sprite = SPRITES_REGISTRY["goal"]
        sprite = jnp.broadcast_to(sprite, (batch_size, *sprite.shape))

        return cls(
            entity_type=entity_type,
            position=ensure_batched(position, 1),
            tag=ensure_batched(tag, 0),
            probability=ensure_batched(probability, 0),
            sprite=sprite
        )

    @property
    def walkable(self) -> Array:
        return jnp.broadcast_to(jnp.asarray(True), self.probability.shape)

    @property
    def transparent(self) -> Array:
        return jnp.broadcast_to(jnp.asarray(True), self.probability.shape)

    def get_sprite(self) -> Array:
        sprite = SPRITES_REGISTRY["goal"]
        return jnp.broadcast_to(sprite, (self.batch_size, *sprite.shape))


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

        batch_size = position.shape[0]

        entity_type = jnp.broadcast_to(jnp.asarray(4), (batch_size,))
        batched_idx = jnp.broadcast_to(0, (batch_size,))
        sprite = SPRITES_REGISTRY["key"]
        sprite = jnp.broadcast_to(sprite, (batch_size, *sprite.shape))
        return cls(
            entity_type=entity_type,
            position=position,
            tag=-id,
            id=id,
            sprite=sprite
        )

    @property
    def walkable(self) -> Array:
        return jnp.broadcast_to(jnp.asarray(False), self.id.shape)

    @property
    def transparent(self) -> Array:
        return jnp.broadcast_to(jnp.asarray(True), self.id.shape)

    def get_sprite(self) -> Array:
        sprite = SPRITES_REGISTRY["key"]
        return jnp.broadcast_to(sprite, (self.batch_size, *sprite.shape))


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

        # init
        batch_size = position.shape[0]
        entity_type = jnp.broadcast_to(jnp.asarray(5), (batch_size,))
        is_open = jnp.broadcast_to(jnp.asarray(False), (batch_size,))

        is_open_as_idx = jnp.asarray(is_open, dtype=jnp.int32)
        sprite = SPRITES_REGISTRY["door"][direction, is_open_as_idx]

        return cls(
            entity_type=entity_type,
            position=position,
            direction=direction,
            requires=requires,
            tag=requires,
            open=is_open,
            sprite=sprite
        )


    @property
    def walkable(self) -> Array:
        return self.open

    @property
    def transparent(self) -> Array:
        return self.open

    def get_sprite(self) -> Array:
        is_open_as_idx = jnp.asarray(self.open, dtype=jnp.int32)
        return SPRITES_REGISTRY["door"][self.direction, is_open_as_idx]


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
        return self.entities.get(Entities.WALL.value, Wall.create())  # type: ignore

    def set_walls(self, walls: Wall) -> State:
        self.entities[Entities.WALL.value] = walls
        return self

    def get_player(self, idx: int = 0) -> Player:
        return self.entities[Entities.PLAYER.value][idx]  # type: ignore

    def set_player(self, player: Player, idx: int = 0) -> State:
        player = self.entities[Entities.PLAYER.value] = player[None]
        return self

    def get_goals(self) -> Goal:
        return self.entities.get(Entities.GOAL.value, Goal.create())  # type: ignore

    def set_goals(self, goals: Goal) -> State:
        self.entities[Entities.GOAL.value] = goals
        return self

    def get_keys(self) -> Key:
        return self.entities.get(Entities.KEY.value, Key.create())  # type: ignore

    def set_keys(self, keys: Key) -> State:
        self.entities[Entities.KEY.value] = keys
        return self

    def get_doors(self) -> Door:
        return self.entities.get(Entities.DOOR.value, Door.create())  # type: ignore

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
