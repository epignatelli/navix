"""The concrete things that live on a grid: `Player`, `Goal`, `Wall`,
`Key`, `Door`, `Lava`, `Ball`, `Box`.

Each is a frozen `flax.struct` pytree built by composing the mixins in
`navix.components` (`Positionable`, `Directional`, `Openable`, ...). All
fields are batched: `state.entities["key"]` is a single `Key` struct
holding *every* key in the environment, with the instance count as the
leading axis (`key.position` is `i32[n_keys, 2]`). Index into a batch
with `entity[i]`.

Entities also expose derived, per-instance properties the engine reads:
`walkable`, `transparent`, `tag`, `sprite`, `symbolic_state`.
"""

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
    """The string keys used in `state.entities` (a `dict[str, Entity]`).
    Use `Entities.KEY` etc. rather than the bare literal `"key"` so a
    rename is caught statically."""

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
    """The integer tag each entity type takes in a `categorical`
    observation and in the first channel of a `symbolic` one. `uint8`
    scalars. The values are not contiguous (there is no `3`) - treat them
    as opaque ids, and `MAX_CATEGORICAL_VALUE` (in
    `environments.environment`) as the count the observation `Space` uses.
    `UNKNOWN` (`0`) is also the value of a cell that has not been seen in
    a first-person observation."""

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
    """Named values for `Directional.direction`: `EAST=0`, `SOUTH=1`,
    `WEST=2`, `NORTH=3` (clockwise). `rotate_cw` adds 1 (mod 4)."""

    EAST = jnp.asarray(0)
    SOUTH = jnp.asarray(1)
    WEST = jnp.asarray(2)
    NORTH = jnp.asarray(3)


class Entity(Positionable, HasTag, HasSprite):
    """Base of every concrete entity: has a `position`, a `tag` and a
    `sprite`. Subclasses add more mixins and fill in the derived
    properties below. Build one with the subclass's `create`."""

    def __getitem__(self: T, idx) -> T:
        """Selects instance(s) from the batch - `key[0]` is the first key,
        `key[mask]` the masked subset - by indexing every field's leading
        axis."""
        return jax.tree.map(lambda x: x[idx], self)

    @property
    def name(self) -> str:
        """The class name (`"Key"`, `"Door"`, ...)."""
        return self.__class__.__name__

    @property
    def shape(self) -> Tuple[int, ...]:
        """The batch shape - the entity's axes *excluding* each field's
        own trailing axes. `()` for a single instance, `(n,)` for a batch
        of `n` (`position` is then `(n, 2)`)."""
        return self.position.shape[:-1]

    @property
    def ndim(self) -> int:
        """Number of batch dimensions (`len(self.shape)`) - `0` for a
        single instance, `1` for a flat batch."""
        return self.position.ndim - 1

    @property
    def walkable(self) -> Array:
        """`bool[*shape]` - can the player step onto this entity's cell?
        (`False` for walls and closed doors, `True` for goal/lava/floor.)

        Raises:
            NotImplementedError: on the base class."""
        raise NotImplementedError()

    @property
    def transparent(self) -> Array:
        """`bool[*shape]` - does line of sight pass through this cell?
        Feeds the first-person view cone (`False` blocks vision).

        Raises:
            NotImplementedError: on the base class."""
        raise NotImplementedError()

    @property
    def symbolic_state(self) -> Array:
        """`i32[*shape]` - the third channel of a `symbolic` observation
        for this entity (e.g. a door's open/closed/locked; `0` for
        entities with no internal state).

        Raises:
            NotImplementedError: on the base class."""
        raise NotImplementedError()


class Wall(Entity, HasColour):
    """An impassable, opaque cell. Not walkable, not transparent. The
    grid border is walls; interior walls form rooms and corridors."""

    @classmethod
    def create(
        cls,
        position: Array,
    ) -> Wall:
        """Args:
            position (Array): `(row, col)` of each wall, `i32[n, 2]`.
                Colour is fixed to grey.

        Returns:
            Wall: the batch of walls."""
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

    @property
    def symbolic_state(self) -> Array:
        return jnp.broadcast_to(0, self.shape)


class Player(Entity, Directional, Holder):
    """The agent. Has a `direction` it faces and a `pocket` holding at
    most one `Pickable`. navix is single-agent, so `state.entities["player"]`
    is a batch of one."""

    @classmethod
    def create(
        cls,
        position: Array,
        direction: Array,
        pocket: Array,
    ) -> Player:
        """Args:
            position (Array): `(row, col)`, `i32[2]` (or batched).
            direction (Array): facing, `i32[]` in `0..3` (see
                `Directions`).
            pocket (Array): carried item id, or `EMPTY_POCKET_ID` (`-1`).

        Returns:
            Player: the entity."""
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

    @property
    def symbolic_state(self) -> Array:
        return jnp.broadcast_to(self.direction, self.shape)


class Goal(Entity, HasColour, Stochastic):
    """The target cell. Walkable and transparent. Reaching it fires a
    `(GOAL, REACH)` event with probability `probability` (`1.0` in the
    standard tasks) - `rewards.on_goal_reached` /
    `terminations.on_goal_reached` react to it."""

    @classmethod
    def create(
        cls,
        position: Array,
        probability: Array,
    ) -> Goal:
        """Args:
            position (Array): `(row, col)`, `i32[2]` (or batched).
            probability (Array): `f32[]` in `[0, 1]` - chance the reward
                fires on contact (`1.0` for the standard tasks). Colour
                is fixed to green.

        Returns:
            Goal: the entity."""
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

    @property
    def symbolic_state(self) -> Array:
        return jnp.broadcast_to(0, self.shape)


class Key(Entity, Pickable, HasColour):
    """A pickable key. Not walkable, transparent. `pickup` puts its `id`
    in the player's pocket; a `Door` whose `requires` equals that `id`
    can then be opened, consuming the key. Its `colour` matches the door
    it opens."""

    @classmethod
    def create(
        cls,
        position: Array,
        colour: Array,
        id: Array,
    ) -> Key:
        """Args:
            position (Array): `(row, col)`, `i32[2]` (or batched).
            colour (Array): palette index (see `HasColour`).
            id (Array): `i32[] >= 1`, matched against `Door.requires`.

        Returns:
            Key: the entity."""
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

    @property
    def symbolic_state(self) -> Array:
        return jnp.broadcast_to(0, self.shape)


class Door(Entity, Openable, HasColour):
    """A door in a wall. While closed it is not walkable and not
    transparent; once open it is both. Opening needs the `open` action
    while facing it and, if `requires != -1`, the matching `Key` in the
    pocket (which is then consumed and `requires` set to `-1`). navix
    doors do not re-close."""

    @classmethod
    def create(
        cls,
        position: Array,
        requires: Array,
        colour: Array,
        open: Array,
    ) -> Door:
        """Args:
            position (Array): `(row, col)`, `i32[2]` (or batched).
            requires (Array): `Key.id` needed to unlock, or `-1` for an
                unlocked door.
            colour (Array): palette index (see `HasColour`).
            open (Array): `0` closed, `1` open.

        Returns:
            Door: the entity."""
        colour = jnp.asarray(colour, dtype=jnp.uint8)
        return cls(
            position=position,
            requires=requires,
            open=open,
            colour=colour,
        )

    @property
    def walkable(self) -> Array:
        return jnp.asarray(self.open, dtype=jnp.bool_)

    @property
    def transparent(self) -> Array:
        return jnp.asarray(self.open, dtype=jnp.bool_)

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
        """`bool[*shape]` - `True` while the door still needs a key
        (`requires != -1`). Becomes `False` once unlocked."""
        return self.requires != jnp.asarray(-1)

    @property
    def symbolic_state(self) -> Array:
        """
        Returns an integer array encoding the symbolic state of the door:

        - 0: Door is open
        - 1: Door is closed but not locked
        - 2: Door is closed and locked (requires a key or tool)

        Examples:
            - If open = 1 and locked = 0: symbolic_state = 0 (open)
            - If open = 0 and locked = 0: symbolic_state = 1 (closed, not locked)
            - If open = 0 and locked = 1: symbolic_state = 2 (closed and locked)
        """
        closed = 1 - self.open
        return closed + closed * self.locked


class Lava(Entity):
    """A hazard cell. Walkable and transparent (the player *can* step
    onto it), but doing so fires a `(LAVA, FALL)` event that
    `terminations.on_lava_fall` turns into a `TERMINATION` - stepping in
    ends the episode with no reward."""

    @classmethod
    def create(
        cls,
        position: Array,
    ) -> Lava:
        """Args:
            position (Array): `(row, col)`, `i32[2]` (or batched).

        Returns:
            Lava: the entity."""
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

    @property
    def symbolic_state(self) -> Array:
        return jnp.broadcast_to(0, self.shape)


class Ball(Entity, Pickable, HasColour, Stochastic):
    """A blocking obstacle that is also pickable. Not walkable,
    transparent. Colliding with the player fires a `(BALL, HIT)` event
    (`terminations.on_ball_hit`).

    Under the default `transitions.stochastic_transition`, every ball
    moves one random step each timestep (`transitions.update_balls`), so
    it acts as a wandering hazard. An environment that wants a *static*
    ball - a pickup target or decoy - must use
    `transitions_fn=transitions.deterministic_transition` instead (as
    `GoToObject` / `Fetch` / `PutNear` / `BlockedUnlockPickup` do).

    `id` (from `Pickable`) is the pickup identity, matched against
    `player.pocket`; `probability` (from `Stochastic`) is unused by
    `Ball`."""

    @classmethod
    def create(
        cls,
        position: Array,
        colour: Array,
        probability: Array,
        id: Array,
    ) -> Ball:
        """Args:
            position (Array): `(row, col)`, `i32[2]` (or batched).
            colour (Array): palette index (see `HasColour`).
            probability (Array): unused by `Ball`; pass `1.0`.
            id (Array): `i32[] >= 1` pickup identity.

        Returns:
            Ball: the entity."""
        colour = jnp.asarray(colour, dtype=jnp.uint8)
        return cls(position=position, colour=colour, probability=probability, id=id)

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

    @property
    def symbolic_state(self) -> Array:
        return jnp.broadcast_to(0, self.shape)


class Box(Entity, Pickable, HasColour, Holder):
    """A pickable container. Not walkable, transparent. `open` (the
    `toggle` action) while facing it removes the box and, if its `pocket`
    holds a `Key`'s id, reveals that key at the box's former cell
    (matching MiniGrid's `Box.toggle`). Used by `ObstructedMaze` to hide
    keys.

    Two separate id fields: `id` (from `Pickable`) is the box's own
    pickup identity, matched against `player.pocket`; `pocket` (from
    `Holder`) is the id of the item hidden *inside*."""

    @classmethod
    def create(
        cls,
        position: Array,
        colour: Array,
        id: Array,
        pocket: Array,
    ) -> Box:
        """Args:
            position (Array): `(row, col)`, `i32[2]` (or batched).
            colour (Array): palette index (see `HasColour`).
            id (Array): `i32[] >= 1`, the box's own pickup identity.
            pocket (Array): id of the contained item, or
                `EMPTY_POCKET_ID` (`-1`) for an empty box.

        Returns:
            Box: the entity."""
        colour = jnp.asarray(colour, dtype=jnp.uint8)
        return cls(position=position, colour=colour, id=id, pocket=pocket)

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

    @property
    def symbolic_state(self) -> Array:
        return jnp.broadcast_to(0, self.shape)
