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
"""`State` - the full, true world state - and the per-step `Event` /
`EventsManager` record it carries.

`State` bundles the static `grid`, the batched `entities` dict, a
rendering `cache` and a PRNG `key`; an `observation_fn` derives what the
agent sees from it. `EventsManager` is a small fixed set of `Event`
slots that the transition pipeline writes ("goal reached", "ball hit",
...) and that `navix.events` / `rewards` / `terminations` read.
"""

from __future__ import annotations
from typing import Dict, Tuple

from jax import Array
import jax.numpy as jnp
from flax import struct
from dataclasses import field


from .components import (
    Positionable,
    HasColour,
)
from .grid import positions_equal
from .rendering.cache import RenderingCache
from .rendering.registry import PALETTE, SPRITES_REGISTRY
from .entities import Entity, Entities, Goal, Wall, Ball, Lava, Key, Door, Box, Player


class EventType:
    """Enumeration of the different types of events that can happen in the environment.

    Plain strings, not jax arrays - `EventsManager.events` keys off
    `(entity_type, event_type)` tuples, and a jax pytree's dict keys
    must be static, hashable Python values, not traced arrays."""

    NONE: str = "none"
    REACH: str = "reach"
    HIT: str = "hit"
    FALL: str = "fall"
    PICKUP: str = "pickup"
    OPEN: str = "open"
    UNLOCK: str = "unlock"


GRID = "grid"
"""Pseudo entity-type key for a wall-hit against the grid boundary or a
non-walkable empty cell, as opposed to hitting an actual `Wall` entity
- `EventsManager.record_grid_hit`/`record_wall_hit` write to separate
`(GRID, EventType.HIT)`/`(Entities.WALL, EventType.HIT)` slots so they
never fight over the same one; `navix.events.on_wall_hit` ORs both
together, so the distinction is invisible to callers that only care
whether a wall was hit at all."""


class Event(Positionable, HasColour):
    """A struct representing an event that happened in the environment. It contains the
    position of the event, the colour of the entity involved in the event, and whether the
    event happened.

    !!! note
        Notice that we need the `happened` property, which flags if an event has
        happened or not, because JAX does not support variable size arrays.
        This means that we cannot add an event to the list in the middle of training.
        Instead, we initialise all events, and mask them out as not happened.

    Every field of an `Event` stored in `EventsManager.events` is batched to match
    the entity type it tracks (see `empty_like`) - one slot per instance, not one
    scalar per event type - so e.g. two balls hitting the player the same step are
    two independent `True` entries, not collapsed into one (see issue #139).

    Attributes:
        position (Array): The (row, column) position of the event in the grid.
        colour (Array): The colour of the entity involved in the event.
        happened (Array): A boolean flag indicating whether the event happened.
    """

    position: Array = field(
        default_factory=lambda: jnp.asarray([-1, -1], dtype=jnp.int32)
    )
    colour: Array = field(default_factory=lambda: PALETTE.UNSET)
    happened: Array = field(default_factory=lambda: jnp.asarray(False, dtype=jnp.bool_))

    @classmethod
    def empty_like(cls, entity: Entity) -> Event:
        """An all-`happened=False` `Event`, batched to match `entity`'s own
        shape (one slot per instance of `entity`) rather than a single
        scalar slot.

        Args:
            entity (Entity): The entity type this event tracks - e.g.
                `state.entities[Entities.BALL]`.

        Returns:
            Event: `position`/`colour`/`happened`, each broadcast to
            `entity.shape` (plus `position`'s own trailing `(2,)`)."""
        shape = entity.shape
        return cls(
            position=jnp.broadcast_to(
                jnp.asarray([-1, -1], dtype=jnp.int32), (*shape, 2)
            ),
            colour=jnp.broadcast_to(PALETTE.UNSET, shape),
            happened=jnp.broadcast_to(jnp.asarray(False, dtype=jnp.bool_), shape),
        )

    def __eq__(self, other: Event) -> Array:
        return jnp.logical_and(
            jnp.array_equal(self.position, other.position),
            jnp.array_equal(self.colour, other.colour),
        )

    def __ne__(self, other: Event) -> Array:
        return jnp.logical_not(self == other)


class EventsManager(struct.PyTreeNode):
    """A struct that manages the events that happened in the environment this
    step, such as the goal being reached, the player being hit by a ball, etc.

    Keyed by `(entity_type, event_type)` rather than one named field per
    event type, and each slot's `Event` is batched to match that entity
    type's own instance count (see `Event.empty_like`) - this is what lets
    two independent occurrences of the same event type (e.g. two balls
    both hitting the player) coexist within one step, rather than one
    replacing the other (see issue #139). `record_*` merges new hits into
    a slot rather than replacing it wholesale, so a call earlier in the
    same step's transition pipeline (e.g. the player walking into a ball)
    isn't discarded by a later one (e.g. a different ball moving onto the
    player) writing the same slot.

    Attributes:
        events (Dict[Tuple[str, str], Event]): One `Event` slot per
            `(entity_type, event_type)` this environment's entities can
            actually produce - see `create`. Not meant to be constructed
            directly; `State.__post_init__` calls `create` automatically
            from `State.entities`.
    """

    events: Dict[Tuple[str, str], Event] = struct.field(default_factory=dict)

    @classmethod
    def create(cls, entities: Dict[str, Entity]) -> EventsManager:
        """Builds one all-`happened=False` `Event` slot per `(entity_type,
        event_type)` this environment's `entities` can actually produce -
        e.g. `(Entities.BALL, EventType.HIT)` only exists if
        `Entities.BALL in entities`. Called automatically by
        `State.__post_init__`; not meant to be called directly.

        Args:
            entities (Dict[str, Entity]): `State.entities`.

        Returns:
            EventsManager: A fresh, all-`happened=False` manager."""
        events: Dict[Tuple[str, str], Event] = {(GRID, EventType.HIT): Event()}
        if Entities.GOAL in entities:
            events[Entities.GOAL, EventType.REACH] = Event.empty_like(
                entities[Entities.GOAL]
            )
        if Entities.WALL in entities:
            events[Entities.WALL, EventType.HIT] = Event.empty_like(
                entities[Entities.WALL]
            )
        if Entities.LAVA in entities:
            events[Entities.LAVA, EventType.FALL] = Event.empty_like(
                entities[Entities.LAVA]
            )
        if Entities.KEY in entities:
            events[Entities.KEY, EventType.PICKUP] = Event.empty_like(
                entities[Entities.KEY]
            )
        if Entities.DOOR in entities:
            events[Entities.DOOR, EventType.OPEN] = Event.empty_like(
                entities[Entities.DOOR]
            )
            events[Entities.DOOR, EventType.UNLOCK] = Event.empty_like(
                entities[Entities.DOOR]
            )
        if Entities.BALL in entities:
            events[Entities.BALL, EventType.HIT] = Event.empty_like(
                entities[Entities.BALL]
            )
            events[Entities.BALL, EventType.PICKUP] = Event.empty_like(
                entities[Entities.BALL]
            )
        if Entities.BOX in entities:
            events[Entities.BOX, EventType.PICKUP] = Event.empty_like(
                entities[Entities.BOX]
            )
        return cls(events=events)

    def happened(self, key: Tuple[str, str]) -> Array:
        """Whether any instance of `key`'s `(entity_type, event_type)` slot
        fired this step - `False` (not a `KeyError`) if this environment's
        `entities` never included that entity type at all (see `create`).

        Args:
            key (Tuple[str, str]): The `(entity_type, event_type)` slot to
                check.

        Returns:
            Array: A boolean scalar."""
        event = self.events.get(key)
        if event is None:
            return jnp.asarray(False)
        return jnp.any(event.happened)

    def happened_at(self, key: Tuple[str, str], position: Array) -> Array:
        """Whether `key`'s slot fired this step for the specific instance
        that was at `position` when recorded - `Event.position` keeps the
        firing instance's own position even after game logic later moves
        the entity itself (e.g. to the discard pile on pickup), so this
        can identify *which* instance fired, not just whether any did
        (see `happened`). `False` (not a `KeyError`) if this
        environment's `entities` never included that entity type at all.

        Args:
            key (Tuple[str, str]): The `(entity_type, event_type)` slot to
                check.
            position (Array): The position to match against.

        Returns:
            Array: A boolean scalar."""
        event = self.events.get(key)
        if event is None:
            return jnp.asarray(False)
        at_position = jnp.all(event.position == position, axis=-1)
        return jnp.any(jnp.logical_and(event.happened, at_position))

    def merge_event(
        self, key: Tuple[str, str], hit: Array, position: Array, colour: Array
    ) -> EventsManager:
        """Records `hit` into `key`'s `Event` slot, OR-merging onto
        whatever already happened this step rather than replacing it
        wholesale - so a call earlier in the same step's transition
        pipeline isn't silently discarded by a later one writing the same
        slot (see issue #139). `position`/`colour` are written only where
        `hit` is newly `True`; already-`True` entries keep their existing
        stored `position`/`colour`.

        Args:
            key (Tuple[str, str]): The `(entity_type, event_type)` slot to
                update - must already exist (see `create`).
            hit (Array): Boolean - per-instance for a batched slot (e.g.
                `Entities.BALL, EventType.HIT`), a bare scalar for a
                single-instance slot (e.g. `GRID, EventType.HIT`).
            position (Array): This instance's own position - written
                where `hit` is `True`.
            colour (Array): This instance's own colour - written where
                `hit` is `True`.

        Returns:
            EventsManager: The updated events manager."""
        existing = self.events[key]
        hit = jnp.asarray(hit, dtype=jnp.bool_)
        happened = jnp.logical_or(existing.happened, hit)
        new_position = jnp.where(hit[..., None], position, existing.position)
        new_colour = jnp.where(hit, colour, existing.colour)
        events = dict(self.events)
        events[key] = existing.replace(
            position=new_position, colour=new_colour, happened=happened
        )
        return self.replace(events=events)

    def record_walk_into(self, entity: Entity, position: Array) -> EventsManager:
        """Flags an event when the player walks into an entity as happened and returns the
        updated events manager.

        Args:
            entity (Entity): The entity the player walked into.
            position (Array): The position of the entity in the grid.

        Returns:
            EventsManager: The updated events manager."""
        hit = positions_equal(entity.position, position)
        if isinstance(entity, Goal):
            return self.record_goal_reached(entity, hit)
        elif isinstance(entity, Wall):
            return self.record_wall_hit(entity, hit)
        elif isinstance(entity, Lava):
            return self.record_lava_fall(entity, hit)
        elif isinstance(entity, Ball):
            return self.record_ball_hit(entity, hit)
        return self

    def record_pickup(self, entity: Entity, position: Array) -> EventsManager:
        """Flags an event when the player picks up an entity as happened and returns the
        updated events manager.

        Args:
            entity (Entity): The entity the player picked up.
            position (Array): The position of the entity in the grid.

        Returns:
            EventsManager: The updated events manager."""
        hit = positions_equal(entity.position, position)
        if isinstance(entity, Key):
            return self.record_key_pickup(entity, hit)
        elif isinstance(entity, Ball):
            return self.record_ball_pickup(entity, hit)
        elif isinstance(entity, Box):
            return self.record_box_pickup(entity, hit)
        return self

    def record_goal_reached(self, goal: Goal, hit: Array) -> EventsManager:
        """Flags an event when the player reaches the goal as happened and returns the
        updated events manager.

        Args:
            goal (Goal): Every `Goal` instance in the environment.
            hit (Array): Boolean, one entry per `goal` instance.

        Returns:
            EventsManager: The updated events manager."""
        return self.merge_event(
            (Entities.GOAL, EventType.REACH), hit, goal.position, PALETTE.UNSET
        )

    def record_ball_hit(self, ball: Ball, hit: Array) -> EventsManager:
        """Flags an event when the player is hit by a ball as happened and returns the
        updated events manager.

        Args:
            ball (Ball): Every `Ball` instance in the environment.
            hit (Array): Boolean, one entry per `ball` instance.

        Returns:
            EventsManager: The updated events manager."""
        return self.merge_event(
            (Entities.BALL, EventType.HIT), hit, ball.position, ball.colour
        )

    def record_wall_hit(self, wall: Wall, hit: Array) -> EventsManager:
        """Flags an event when the player hits a wall as happened and returns the
        updated events manager.

        Args:
            wall (Wall): Every `Wall` instance in the environment.
            hit (Array): Boolean, one entry per `wall` instance.

        Returns:
            EventsManager: The updated events manager."""
        return self.merge_event(
            (Entities.WALL, EventType.HIT), hit, wall.position, PALETTE.UNSET
        )

    def record_grid_hit(self, position: Array) -> EventsManager:
        """Flags an event when the player hits the grid boundary or a
        non-walkable empty cell (no `Wall` entity there) as happened and
        returns the updated events manager. Kept in a separate `GRID` slot
        from `record_wall_hit` (see `GRID`'s docstring).

        Args:
            position (Array): The position hit.

        Returns:
            EventsManager: The updated events manager."""
        return self.merge_event(
            (GRID, EventType.HIT), jnp.asarray(True), position, PALETTE.UNSET
        )

    def record_lava_fall(self, lava: Lava, hit: Array) -> EventsManager:
        """Flags an event when the lava falls as happened and returns the
        updated events manager.

        Args:
            lava (Lava): Every `Lava` instance in the environment.
            hit (Array): Boolean, one entry per `lava` instance.

        Returns:
            EventsManager: The updated events manager."""
        return self.merge_event(
            (Entities.LAVA, EventType.FALL), hit, lava.position, PALETTE.UNSET
        )

    def record_key_pickup(self, key: Key, hit: Array) -> EventsManager:
        """Flags an event when the player picks up a key as happened and returns the
        updated events manager.

        Args:
            key (Key): Every `Key` instance in the environment.
            hit (Array): Boolean, one entry per `key` instance.

        Returns:
            EventsManager: The updated events manager."""
        return self.merge_event(
            (Entities.KEY, EventType.PICKUP), hit, key.position, key.colour
        )

    def record_box_pickup(self, box: Box, hit: Array) -> EventsManager:
        """Flags an event when the player picks up a box as happened and returns the
        updated events manager.

        Args:
            box (Box): Every `Box` instance in the environment.
            hit (Array): Boolean, one entry per `box` instance.

        Returns:
            EventsManager: The updated events manager."""
        return self.merge_event(
            (Entities.BOX, EventType.PICKUP), hit, box.position, box.colour
        )

    def record_door_opening(self, door: Door, hit: Array) -> EventsManager:
        """Flags an event when the player opens a door as happened and returns the
        updated events manager.

        Args:
            door (Door): Every `Door` instance in the environment.
            hit (Array): Boolean, one entry per `door` instance.

        Returns:
            EventsManager: The updated events manager."""
        return self.merge_event(
            (Entities.DOOR, EventType.OPEN), hit, door.position, door.colour
        )

    def record_door_unlock(self, door: Door, hit: Array) -> EventsManager:
        """Flags an event when the player unlocks a door as happened and returns the
        updated events manager.

        Args:
            door (Door): Every `Door` instance in the environment.
            hit (Array): Boolean, one entry per `door` instance.

        Returns:
            EventsManager: The updated events manager."""
        return self.merge_event(
            (Entities.DOOR, EventType.UNLOCK), hit, door.position, door.colour
        )

    def record_ball_pickup(self, ball: Ball, hit: Array) -> EventsManager:
        """Flags an event when the player picks up a ball as happened and returns the
        updated events manager.

        Args:
            ball (Ball): Every `Ball` instance in the environment.
            hit (Array): Boolean, one entry per `ball` instance.

        Returns:
            EventsManager: The updated events manager."""
        return self.merge_event(
            (Entities.BALL, EventType.PICKUP), hit, ball.position, ball.colour
        )


class State(struct.PyTreeNode):
    """The full, true state of the world - everything `step` needs to
    compute the next state, and everything an `observation_fn` reads.
    A frozen `flax.struct` pytree (`vmap`s over a batch of environments).
    `Timestep.state` is one of these.

    The `get_*` / `set_*` helpers below are the convenient way to reach
    into `entities`; the raw `entities` dict is also fine to use
    directly."""

    key: Array
    """PRNG key for this environment's own stochasticity (ball motion,
    stochastic goals). Advanced by the functions that consume it."""
    grid: Array
    """`i32[H, W]` static base map: `0` is floor, `-1` marks a wall cell.
    Fixed for the lifetime of an episode; moving entities live in
    `entities`, not here."""
    cache: RenderingCache
    """Pre-rendered tile patches, so `rgb` observations only re-draw the
    cells that changed. Carried in the state so it survives `jax.jit`."""
    entities: Dict[str, Entity] = struct.field(default_factory=dict)
    """Maps an `Entities` key (`"player"`, `"key"`, ...) to a single
    batched `Entity` holding every instance of that type. A type with no
    instances in this environment is simply absent from the dict."""
    events: EventsManager = EventsManager()
    """A struct indicating which events happened this timestep. For example, the
    goal is reached, or the player is hit by a ball. Left at its default (empty)
    here - `__post_init__` populates it from `entities` via `EventsManager.create`,
    since a `struct.PyTreeNode` field's default can't see its sibling fields."""
    mission: Tuple[Event, ...] = ()
    """The environment's mission target(s), if any - e.g. `(door,)` in
    `GoToDoor`, `(target,)` in `GoToObject`/`Fetch`, `(carry, drop_near)`
    in `PutNear` (index 0 is always the "primary"/carry target; index 1,
    where present, is a second, independently-tracked target - only
    `PutNear` needs two today). Empty for an environment with no mission
    at all. A tuple, not a fixed number of separate fields, so a future
    environment needing a third simultaneous target is just a longer
    tuple, not another numbered field."""

    def __post_init__(self) -> None:
        # events.events is only ever empty right after construction with no
        # explicit `events=...` (the default `EventsManager()` above) - once
        # populated, it's never emptied again (record_* only ever adds/merges
        # entries into existing slots, see EventsManager.merge_event), so this
        # correctly runs exactly once per episode (at _reset time), not on
        # every subsequent .replace() call within an episode.
        if not self.events.events:
            object.__setattr__(self, "events", EventsManager.create(self.entities))

    def get_entity(self, entity_enum: str) -> Entity:
        """The batched `Entity` for one type. The typed helpers
        (`get_keys`, `get_doors`, ...) are thin wrappers around this.

        Args:
            entity_enum (str): an `Entities` key, e.g. `Entities.KEY`.

        Returns:
            Entity: every instance of that type, batched (leading axis =
            instance count).

        Raises:
            KeyError: if this environment has no entity of that type -
            check `entity_enum in state.entities` first if unsure."""
        return self.entities[entity_enum]

    def set_entity(self, entity_enum: str, entity: Entity) -> State:
        """Replaces one entity type's batch. Mutates `self.entities` in
        place *and* returns `self` (it does not build a new `State`), so
        `state.set_entity(...)` and `state = state.set_entity(...)` are
        equivalent.

        Args:
            entity_enum (str): an `Entities` key.
            entity (Entity): the new batched entity (its shape may differ
                from the old one - e.g. after a pickup moves an instance
                off-grid).

        Returns:
            State: `self`."""
        self.entities[entity_enum] = entity
        return self

    def get_walls(self) -> Wall:
        """Gets all the `WALL` entities from the state."""
        return self.entities[Entities.WALL]  # type: ignore

    def set_walls(self, walls: Wall) -> State:
        """Sets the `WALL` entities in the state."""
        self.entities[Entities.WALL] = walls
        return self

    def get_player(self, idx: int = 0) -> Player:
        """The player, *unbatched* (navix is single-agent). Unlike the
        other `get_*` helpers this indexes into the batch and returns one
        `Player`.

        Args:
            idx (int): which player - only `0` is meaningful today.

        Returns:
            Player: the player entity, no leading instance axis."""
        return self.entities[Entities.PLAYER][idx]  # type: ignore

    def set_player(self, player: Player, idx: int = 0) -> State:
        """Sets the player entity in the state. Notice that we only support one player in the
        environment for now, but this can easily be extended to multiple players."""
        # TODO(epignatelli): this is a hack and won't work in multi-agent settings
        self.entities[Entities.PLAYER] = player[None]
        return self

    def get_goals(self) -> Goal:
        """Gets the goal entity from the state."""
        return self.entities[Entities.GOAL]  # type: ignore

    def set_goals(self, goals: Goal) -> State:
        """Sets the goal entity in the state."""
        self.entities[Entities.GOAL] = goals
        return self

    def get_keys(self) -> Key:
        """Gets the key entity from the state."""
        return self.entities[Entities.KEY]  # type: ignore

    def set_keys(self, keys: Key) -> State:
        """Sets the key entity in the state."""
        self.entities[Entities.KEY] = keys
        return self

    def get_doors(self) -> Door:
        """Gets the door entity from the state."""
        return self.entities[Entities.DOOR]  # type: ignore

    def set_doors(self, doors: Door) -> State:
        """Sets the door entity in the state."""
        self.entities[Entities.DOOR] = doors
        return self

    def get_lavas(self) -> Lava:
        """Gets the lava entity from the state."""
        return self.entities[Entities.LAVA]  # type: ignore

    def get_balls(self) -> Ball:
        """Gets the ball entity from the state."""
        return self.entities[Entities.BALL]  # type: ignore

    def get_boxes(self) -> Box:
        """Gets the box entity from the state."""
        return self.entities[Entities.BOX]  # type: ignore

    def set_balls(self, balls: Ball) -> State:
        """Sets the ball entity in the state."""
        self.entities[Entities.BALL] = balls
        return self

    def set_boxes(self, boxes: Box) -> State:
        """Sets the box entity in the state."""
        self.entities[Entities.BOX] = boxes
        return self

    def set_events(self, events: EventsManager) -> State:
        """Sets the events in the state."""
        return self.replace(events=events)

    def get_positions(self) -> Array:
        """Every entity instance's `(row, col)`, concatenated across all
        types into one `i32[N, 2]` (`N` = total instance count). The
        ordering matches `get_tags` / `get_sprites` / `get_transparency`,
        so they can be zipped - this is what the observation functions
        do to paint entities onto the grid."""
        return jnp.concatenate([self.entities[k].position for k in self.entities])

    def get_tags(self) -> Array:
        """Every entity instance's `tag`, concatenated into `i32[N]`, in
        the same order as `get_positions`. Used to build a `categorical`
        observation."""
        return jnp.concatenate([self.entities[k].tag for k in self.entities])

    def get_sprites(self) -> Array:
        """Every entity instance's RGB sprite, concatenated into
        `u8[N, TILE_SIZE, TILE_SIZE, 3]`, in the same order as
        `get_positions`. Used to build an `rgb` observation."""
        return jnp.concatenate([self.entities[k].sprite for k in self.entities])

    def get_sprites_first_person(self) -> Array:
        """Like `get_sprites`, but the player's sprite is forced to its
        north-facing variant. First-person observations rotate the whole
        view so the player always points up, so its sprite must not also
        carry a rotation."""
        player_sprite = SPRITES_REGISTRY[Entities.PLAYER][-1][None]  # -1 is north
        sprites = []
        for k, v in self.entities.items():
            if k is not Entities.PLAYER:
                sprites.append(v.sprite)
            else:
                sprites.append(player_sprite)
        return jnp.concatenate(sprites)

    def get_transparency(self) -> Array:
        """Every entity instance's `transparent` flag, concatenated into
        `bool[N]`, in the same order as `get_positions`. Feeds the
        first-person view cone."""
        return jnp.concatenate([self.entities[k].transparent for k in self.entities])
