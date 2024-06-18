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
from typing import Dict, Tuple

from jax import Array
import jax.numpy as jnp
from flax import struct


from .components import (
    Positionable,
    HasColour,
)
from .rendering.cache import RenderingCache
from .entities import Entity, Entities, Goal, Wall, Ball, Lava, Key, Door, Box, Player


POSITION_UNSET = jnp.asarray([-1, -1], dtype=jnp.int32)
COLOUR_UNSET = jnp.asarray(-1, dtype=jnp.uint8)


# class EventType:
#     NONE: Array = jnp.asarray(-1, dtype=jnp.int32)
#     REACH: Array = jnp.asarray(0, dtype=jnp.int32)
#     HIT: Array = jnp.asarray(1, dtype=jnp.int32)
#     FALL: Array = jnp.asarray(2, dtype=jnp.int32)
#     PICKUP: Array = jnp.asarray(3, dtype=jnp.int32)
#     OPEN: Array = jnp.asarray(4, dtype=jnp.int32)
#     UNLOCK: Array = jnp.asarray(5, dtype=jnp.int32)


class EventType:
    NONE: str = "NONE"
    REACH: str = "REACH"
    HIT: str = "HIT"
    FALL: str = "FALL"
    PICKUP: str = "PICKUP"
    OPEN: str = "OPEN"
    UNLOCK: str = "UNLOCK"


class Event(Positionable, HasColour):
    position: Array = POSITION_UNSET
    colour: Array = COLOUR_UNSET
    happened: Array = jnp.asarray(False, dtype=jnp.bool_)
    # event_type: Array = EventType.NONE

    @classmethod
    def empty_like(cls, entity: Entity) -> Event:
        return cls(
            position=jnp.broadcast_to(POSITION_UNSET, entity.shape),
            colour=jnp.broadcast_to(COLOUR_UNSET, entity.shape),
            happened=jnp.broadcast_to(False, entity.shape),
            # event_type=event_type,
        )

    def __eq__(self, other: Event) -> Array:
        return jnp.logical_and(
            jnp.array_equal(self.position, other.position),
            jnp.array_equal(self.colour, other.colour),
        )

    def __ne__(self, other: Event) -> Array:
        return jnp.logical_not(self == other)


class EventsManager(struct.PyTreeNode):
    events: Dict[Tuple[str, str], Event] = struct.field(default_factory=dict)
    # goal_reached: Event = Event()
    # ball_hit: Event = Event()
    # wall_hit: Event = Event()
    # lava_fall: Event = Event()
    # key_pickup: Event = Event()
    # door_opening: Event = Event()
    # door_unlock: Event = Event()
    # ball_pickup: Event = Event()

    @classmethod
    def create(cls, entities: Dict[str, Entity]) -> EventsManager:
        events = {}

        if Entities.GOAL in entities:
            goal_reached = Event.empty_like(entities[Entities.GOAL])
            events[Entities.GOAL, EventType.REACH] = goal_reached

        if Entities.BALL in entities:
            ball_hit = Event.empty_like(entities[Entities.BALL])
            ball_pickup = Event.empty_like(entities[Entities.BALL])
            events[Entities.BALL, EventType.HIT] = ball_hit
            events[Entities.BALL, EventType.PICKUP] = ball_pickup

        if Entities.WALL in entities:
            wall_hit = Event.empty_like(entities[Entities.WALL])
            events[Entities.WALL, EventType.HIT] = wall_hit

        if Entities.LAVA in entities:
            lava_fall = Event.empty_like(entities[Entities.LAVA])
            events[Entities.LAVA, EventType.FALL] = lava_fall

        if Entities.KEY in entities:
            key_pickup = Event.empty_like(entities[Entities.KEY])
            events[Entities.KEY, EventType.PICKUP] = key_pickup

        if Entities.DOOR in entities:
            door_opening = Event.empty_like(entities[Entities.DOOR])
            door_unlock = Event.empty_like(entities[Entities.DOOR])
            events[Entities.DOOR, EventType.OPEN] = door_opening
            events[Entities.DOOR, EventType.UNLOCK] = door_unlock

        # wall hit due to grid
        grid_hit = Event(
            position=POSITION_UNSET,
            colour=COLOUR_UNSET,
            happened=jnp.asarray(False, dtype=jnp.bool_),
        )
        events["grid", EventType.HIT] = grid_hit

        return cls(events)

    def record_goal_reached(self, goals: Goal, position: Array) -> EventsManager:
        assert (
            Entities.GOAL,
            EventType.REACH,
        ) in self.events, f"No subscription to event ({Entities.GOAL, EventType.REACH})"

        goal_reached = self.events[Entities.GOAL, EventType.REACH]
        cond = jnp.array_equal(goals.position, position)
        new_pos = jnp.where(cond, goals.position, goal_reached.position)
        new_col = jnp.where(cond, goals.colour, goal_reached.colour)
        new_hap = jnp.where(
            cond, jnp.asarray(True, dtype=jnp.bool_), goal_reached.happened
        )
        self.events[Entities.GOAL, EventType.REACH] = Event(
            position=new_pos,
            colour=new_col,
            happened=new_hap,
        )
        return self

    def record_ball_hit(self, ball: Ball, position: Array) -> EventsManager:
        assert (
            Entities.BALL,
            EventType.HIT,
        ) in self.events, f"No subscription to event ({Entities.BALL, EventType.HIT})"

        ball_hit = self.events[Entities.BALL, EventType.HIT]
        cond = jnp.array_equal(ball.position, position)
        new_pos = jnp.where(cond, ball.position, ball_hit.position)
        new_col = jnp.where(cond, ball.colour, ball_hit.colour)
        new_hap = jnp.where(cond, jnp.asarray(True, dtype=jnp.bool_), ball_hit.happened)
        self.events[Entities.BALL, EventType.HIT] = Event(
            position=new_pos,
            colour=new_col,
            happened=new_hap,
        )
        return self

    def record_wall_hit(self, wall: Wall, position: Array) -> EventsManager:
        assert (
            Entities.WALL,
            EventType.HIT,
        ) in self.events, f"No subscription to event ({Entities.WALL, EventType.HIT})"

        wall_hit = self.events[Entities.WALL, EventType.HIT]
        cond = jnp.array_equal(wall.position, position)
        new_pos = jnp.where(cond, wall.position, wall_hit.position)
        new_col = jnp.where(cond, wall.colour, wall_hit.colour)
        new_hap = jnp.where(cond, jnp.asarray(True, dtype=jnp.bool_), wall_hit.happened)
        self.events[Entities.WALL, EventType.HIT] = Event(
            position=new_pos,
            colour=new_col,
            happened=new_hap,
        )
        return self

    def record_lava_fall(self, lava: Lava, position: Array) -> EventsManager:
        assert (
            Entities.LAVA,
            EventType.FALL,
        ) in self.events, f"No subscription to event ({Entities.LAVA, EventType.FALL})"

        lava_fall = self.events[Entities.LAVA, EventType.FALL]
        cond = jnp.array_equal(lava.position, position)
        new_pos = jnp.where(cond, lava.position, lava_fall.position)
        new_hap = jnp.where(
            cond, jnp.asarray(True, dtype=jnp.bool_), lava_fall.happened
        )
        self.events[Entities.LAVA, EventType.FALL] = Event(
            position=new_pos,
            colour=lava_fall.colour,
            happened=new_hap,
        )
        return self

    def record_key_pickup(self, key: Key, position: Array) -> EventsManager:
        assert (
            Entities.KEY,
            EventType.PICKUP,
        ) in self.events, f"No subscription to event ({Entities.KEY, EventType.PICKUP})"

        key_pickup = self.events[Entities.KEY, EventType.PICKUP]
        cond = jnp.array_equal(key.position, position)
        new_pos = jnp.where(cond, key.position, key_pickup.position)
        new_col = jnp.where(cond, key.colour, key_pickup.colour)
        new_hap = jnp.where(
            cond, jnp.asarray(True, dtype=jnp.bool_), key_pickup.happened
        )
        self.events[Entities.KEY, EventType.PICKUP] = Event(
            position=new_pos,
            colour=new_col,
            happened=new_hap,
        )
        return self

    def record_door_opening(self, door: Door, position: Array) -> EventsManager:
        assert (
            Entities.DOOR,
            EventType.OPEN,
        ) in self.events, f"No subscription to event ({Entities.DOOR, EventType.OPEN})"

        door_opening = self.events[Entities.DOOR, EventType.OPEN]
        cond = jnp.array_equal(door.position, position)
        new_pos = jnp.where(cond, door.position, door_opening.position)
        new_col = jnp.where(cond, door.colour, door_opening.colour)
        new_hap = jnp.where(
            cond, jnp.asarray(True, dtype=jnp.bool_), door_opening.happened
        )
        self.events[Entities.DOOR, EventType.OPEN] = Event(
            position=new_pos,
            colour=new_col,
            happened=new_hap,
        )
        return self

    def record_door_unlock(self, door: Door, position: Array) -> EventsManager:
        assert (
            Entities.DOOR,
            EventType.UNLOCK,
        ) in self.events, (
            f"No subscription to event ({Entities.DOOR, EventType.UNLOCK})"
        )

        door_unlock = self.events[Entities.DOOR, EventType.UNLOCK]
        cond = jnp.array_equal(door.position, position)
        new_pos = jnp.where(cond, door.position, door_unlock.position)
        new_col = jnp.where(cond, door.colour, door_unlock.colour)
        new_hap = jnp.where(
            cond, jnp.asarray(True, dtype=jnp.bool_), door_unlock.happened
        )
        self.events[Entities.DOOR, EventType.UNLOCK] = Event(
            position=new_pos,
            colour=new_col,
            happened=new_hap,
        )
        return self

    def record_ball_pickup(self, ball: Ball, position: Array) -> EventsManager:
        assert (
            Entities.BALL,
            EventType.PICKUP,
        ) in self.events, (
            f"No subscription to event ({Entities.BALL, EventType.PICKUP})"
        )

        ball_pickup = self.events[Entities.BALL, EventType.PICKUP]
        cond = jnp.array_equal(ball.position, position)
        new_pos = jnp.where(cond, ball.position, ball_pickup.position)
        new_col = jnp.where(cond, ball.colour, ball_pickup.colour)
        new_hap = jnp.where(
            cond, jnp.asarray(True, dtype=jnp.bool_), ball_pickup.happened
        )
        self.events[Entities.BALL, EventType.PICKUP] = Event(
            position=new_pos,
            colour=new_col,
            happened=new_hap,
        )
        return self

    def record_grid_hit(self, position: Array) -> EventsManager:
        assert (
            "grid",
            EventType.HIT,
        ) in self.events, f"No subscription to event (grid, EventType.HIT)"
        self.events["grid", EventType.HIT] = Event(
            position=position,
            colour=COLOUR_UNSET,
            happened=jnp.asarray(True, dtype=jnp.bool_),
        )
        return self


class State(struct.PyTreeNode):
    """The Markovian state of the environment"""

    key: Array
    """The random number generator state"""
    grid: Array
    """The base map of the environment that remains constant throughout the training"""
    cache: RenderingCache
    """The rendering cache to speed up rendering"""
    entities: Dict[str, Entity] = struct.field(default_factory=dict)
    """The entities in the environment, indexed via entity type string representation.
    Batched over the number of entities for each type"""
    events: EventsManager = EventsManager()
    """A struct indicating which events happened this timestep. For example, the
    goal is reached, or the player is hit by a ball."""
    mission: Event | None = None

    def get_entity(self, entity_enum: str) -> Entity:
        return self.entities[entity_enum]

    def set_entity(self, entity_enum: str, entity: Entity) -> State:
        self.entities[entity_enum] = entity
        return self

    def get_walls(self) -> Wall:
        return self.entities.get(Entities.WALL, Wall())  # type: ignore

    def set_walls(self, walls: Wall) -> State:
        self.entities[Entities.WALL] = walls
        return self

    def get_player(self, idx: int = 0) -> Player:
        return self.entities[Entities.PLAYER][idx]  # type: ignore

    def set_player(self, player: Player, idx: int = 0) -> State:
        # TODO(epignatelli): this is a hack and won't work in multi-agent settings
        self.entities[Entities.PLAYER] = player[None]
        return self

    def get_goals(self) -> Goal:
        return self.entities[Entities.GOAL]  # type: ignore

    def set_goals(self, goals: Goal) -> State:
        self.entities[Entities.GOAL] = goals
        return self

    def get_keys(self) -> Key:
        return self.entities[Entities.KEY]  # type: ignore

    def set_keys(self, keys: Key) -> State:
        self.entities[Entities.KEY] = keys
        return self

    def get_doors(self) -> Door:
        return self.entities[Entities.DOOR]  # type: ignore

    def set_doors(self, doors: Door) -> State:
        self.entities[Entities.DOOR] = doors
        return self

    def get_lavas(self) -> Lava:
        return self.entities[Entities.LAVA]  # type: ignore

    def get_balls(self) -> Ball:
        return self.entities[Entities.BALL]  # type: ignore

    def get_boxes(self) -> Ball:
        return self.entities[Entities.BOX]  # type: ignore

    def set_balls(self, balls: Ball) -> State:
        self.entities[Entities.BALL] = balls
        return self

    def set_boxes(self, boxes: Box) -> State:
        self.entities[Entities.BOX] = boxes
        return self

    def set_events(self, events: EventsManager) -> State:
        return self.replace(events=events)

    def get_positions(self) -> Array:
        return jnp.concatenate([self.entities[k].position for k in self.entities])

    def get_tags(self) -> Array:
        return jnp.concatenate([self.entities[k].tag for k in self.entities])

    def get_sprites(self) -> Array:
        return jnp.concatenate([self.entities[k].sprite for k in self.entities])

    def get_transparency(self) -> Array:
        return jnp.concatenate([self.entities[k].transparent for k in self.entities])
