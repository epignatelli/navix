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
from typing import Dict

from jax import Array
import jax.numpy as jnp
from flax import struct


from .components import (
    Positionable,
    HasColour,
)
from .rendering.cache import RenderingCache
from .rendering.registry import PALETTE
from .entities import Entity, Entities, Goal, Wall, Ball, Lava, Key, Door, Box, Player


class EventType:
    """Enumeration of the different types of events that can happen in the environment."""

    NONE: Array = jnp.asarray(-1, dtype=jnp.int32)
    REACH: Array = jnp.asarray(0, dtype=jnp.int32)
    HIT: Array = jnp.asarray(1, dtype=jnp.int32)
    FALL: Array = jnp.asarray(2, dtype=jnp.int32)
    PICKUP: Array = jnp.asarray(3, dtype=jnp.int32)
    OPEN: Array = jnp.asarray(4, dtype=jnp.int32)
    UNLOCK: Array = jnp.asarray(5, dtype=jnp.int32)


class Event(Positionable, HasColour):
    """A struct representing an event that happened in the environment. It contains the
    position of the event, the colour of the entity involved in the event, whether the event
    happened, and the type of event that happened.

    !!! note
        Notice that we need the `happened` property, which flags if an event has
        happened or not, because JAX does not support variable size arrays.
        This means that we cannot add an event to the list in the middle of training.
        Instead, we initialise all events, and mask them out as not happened.


    Attributes:
        position (Array): The (row, column) position of the event in the grid.
        colour (Array): The colour of the entity involved in the event.
        happened (Array): A boolean flag indicating whether the event happened.
        event_type (Array): The type of event that happened."""

    position: Array = jnp.asarray([-1, -1], dtype=jnp.int32)
    colour: Array = PALETTE.UNSET
    happened: Array = jnp.asarray(False, dtype=jnp.bool_)
    event_type: Array = EventType.NONE

    def __eq__(self, other: Event) -> Array:
        return jnp.logical_and(
            jnp.array_equal(self.position, other.position),
            jnp.array_equal(self.colour, other.colour),
        )

    def __ne__(self, other: Event) -> Array:
        return jnp.logical_not(self == other)


class EventsManager(struct.PyTreeNode):
    """A struct that manages the events. It contains the different events that can happen
    in the environment, such as the goal being reached, the player being hit by a ball, etc.

    Attributes:
        goal_reached (Event): An event indicating that the goal has been reached.
        ball_hit (Event): An event indicating that the player has been hit by a ball.
        wall_hit (Event): An event indicating that the player has hit a wall.
        lava_fall (Event): An event indicating that the lava has fallen.
        key_pickup (Event): An event indicating that the player has picked up a key.
        door_opening (Event): An event indicating that the player has opened a door.
        door_unlock (Event): An event indicating that the player has unlocked a door.
        ball_pickup (Event): An event indicating that the player has picked up a ball.
    """

    goal_reached: Event = Event()
    ball_hit: Event = Event()
    wall_hit: Event = Event()
    lava_fall: Event = Event()
    key_pickup: Event = Event()
    door_opening: Event = Event()
    door_unlock: Event = Event()
    ball_pickup: Event = Event()

    def record_walk_into(self, entity: Entity, position: Array) -> EventsManager:
        """Flags an event when the player walks into an entity as happened and returns the
        updated events manager.

        Args:
            entity (Entity): The entity the player walked into.
            position (Array): The position of the entity in the grid.

        Returns:
            EventsManager: The updated events manager."""
        if isinstance(entity, Goal):
            return self.record_goal_reached(entity, position)
        elif isinstance(entity, Wall):
            return self.record_wall_hit(entity, position)
        elif isinstance(entity, Lava):
            return self.record_lava_fall(entity, position)
        return self

    def record_pickup(self, entity: Entity, position: Array) -> EventsManager:
        """Flags an event when the player picks up an entity as happened and returns the
        updated events manager.

        Args:
            entity (Entity): The entity the player picked up.
            position (Array): The position of the entity in the grid.

        Returns:
            EventsManager: The updated events manager."""
        if isinstance(entity, Key):
            return self.record_key_pickup(entity, position)
        elif isinstance(entity, Ball):
            return self.record_ball_pickup(entity, position)
        return self

    def record_goal_reached(self, goal: Goal, position: Array) -> EventsManager:
        """Flags an event when the player reaches the goal as happened and returns the
        updated events manager.

        Args:
            goal (Goal): The goal the player reached.
            position (Array): The position of the goal in the grid.

        Returns:
            EventsManager: The updated events manager."""
        idx = jnp.where(goal.position == position, size=1)[0][0]
        goal = goal[idx]
        return self.replace(
            goal_reached=Event(
                position=position,
                colour=PALETTE.UNSET,
                happened=jnp.asarray(True, dtype=jnp.bool_),
                event_type=EventType.REACH,
            )
        )

    def record_ball_hit(self, ball: Ball) -> EventsManager:
        """Flags an event when the player is hit by a ball as happened and returns the
        updated events manager.
        
        Args:
            ball (Ball): The ball that hit the player.
        
        Returns:
            EventsManager: The updated events manager."""
        return self.replace(
            ball_hit=Event(
                position=ball.position,
                colour=ball.colour,
                happened=jnp.asarray(True, dtype=jnp.bool_),
                event_type=EventType.HIT,
            )
        )

    def record_wall_hit(self, wall: Wall, position: Array) -> EventsManager:
        """Flags an event when the player hits a wall as happened and returns the
        updated events manager.
        
        Args:
            wall (Wall): The wall the player hit.
            position (Array): The position of the wall in the grid.
        
        Returns:
            EventsManager: The updated events manager."""
        idx = jnp.where(wall.position == position, size=1)[0][0]
        wall = wall[idx]
        return self.replace(
            wall_hit=Event(
                position=wall.position,
                colour=PALETTE.UNSET,
                happened=jnp.asarray(True, dtype=jnp.bool_),
                event_type=EventType.HIT,
            )
        )

    def record_grid_hit(self, position: Array) -> EventsManager:
        """Flags an event when the player hits a wall as happened and returns the
        updated events manager.
        
        Args:
            position (Array): The position of the wall in the grid.
        
        Returns:
            EventsManager: The updated events manager."""
        return self.replace(
            wall_hit=Event(
                position=position,
                colour=PALETTE.UNSET,
                happened=jnp.asarray(True, dtype=jnp.bool_),
                event_type=EventType.HIT,
            )
        )

    def record_lava_fall(self, lava: Lava, position: Array) -> EventsManager:
        """Flags an event when the lava falls as happened and returns the
        updated events manager.
        
        Args:
            lava (Lava): The lava that fell.
            position (Array): The position of the lava in the grid.
        
        Returns:
            EventsManager: The updated events manager."""
        idx = jnp.where(lava.position == position, size=1)[0][0]
        lava = lava[idx]
        return self.replace(
            lava_fall=Event(
                position=lava.position,
                colour=PALETTE.UNSET,
                happened=jnp.asarray(True, dtype=jnp.bool_),
                event_type=EventType.FALL,
            )
        )

    def record_key_pickup(self, key: Key, position: Array) -> EventsManager:
        """Flags an event when the player picks up a key as happened and returns the
        updated events manager.
        
        Args:
            key (Key): The key the player picked up.
            position (Array): The position of the key in the grid.
        
        Returns:
            EventsManager: The updated events manager."""
        idx = jnp.where(key.position == position, size=1)[0][0]
        key = key[idx]
        return self.replace(
            key_pickup=Event(
                position=key.position,
                colour=key.colour,
                happened=jnp.asarray(True, dtype=jnp.bool_),
                event_type=EventType.PICKUP,
            )
        )

    def record_door_opening(self, door: Door, position: Array) -> EventsManager:
        """Flags an event when the player opens a door as happened and returns the
        updated events manager.
        
        Args:
            door (Door): The door the player opened.
            position (Array): The position of the door in the grid.
        
        Returns:
            EventsManager: The updated events manager."""
        idx = jnp.where(door.position == position, size=1)[0][0]
        door = door[idx]
        return self.replace(
            door_opening=Event(
                position=door.position,
                colour=door.colour,
                happened=jnp.asarray(True, dtype=jnp.bool_),
                event_type=EventType.OPEN,
            )
        )

    def record_door_unlock(self, door: Door, position: Array) -> EventsManager:
        """Flags an event when the player unlocks a door as happened and returns the
        updated events manager.

        Args:
            door (Door): The door the player unlocked.
            position (Array): The position of the door in the grid.
        
        Returns:
            EventsManager: The updated events manager."""
        idx = jnp.where(door.position == position, size=1)[0][0]
        door = door[idx]
        return self.replace(
            door_opening=Event(
                position=door.position,
                colour=door.colour,
                happened=jnp.asarray(True, dtype=jnp.bool_),
                event_type=EventType.UNLOCK,
            )
        )

    def record_ball_pickup(self, ball: Ball, position: Array) -> EventsManager:
        """Flags an event when the player picks up a ball as happened and returns the
        updated events manager.
        
        Args:
            ball (Ball): The ball the player picked up.
            position (Array): The position of the ball in the grid.
        
        Returns:
            EventsManager: The updated events manager."""
        idx = jnp.where(ball.position == position, size=1)[0][0]
        ball = ball[idx]
        return self.replace(
            ball_pickup=Event(
                position=ball.position,
                colour=ball.colour,
                happened=jnp.asarray(True, dtype=jnp.bool_),
                event_type=EventType.PICKUP,
            )
        )


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
        """Get an entity from the state by its enum.
        
        Args:
            entity_enum (str): The enum of the entity to get.
        
        Returns:
            Entity: The entity from the state."""
        return self.entities[entity_enum]

    def set_entity(self, entity_enum: str, entity: Entity) -> State:
        """Set an entity in the state by its enum.
        
        Args:
            entity_enum (str): The enum of the entity to set.
            entity (Entity): The entity to set.
        
        Returns:
            State: The updated state."""
        self.entities[entity_enum] = entity
        return self

    def get_walls(self) -> Wall:
        """Gets all the `WALL` entities from the state."""
        return self.entities.get(Entities.WALL, Wall())  # type: ignore

    def set_walls(self, walls: Wall) -> State:
        """Sets the `WALL` entities in the state."""
        self.entities[Entities.WALL] = walls
        return self

    def get_player(self, idx: int = 0) -> Player:
        """Gets the player entity from the state."""
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

    def get_boxes(self) -> Ball:
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
        """Get the positions of all the entities in the state."""
        return jnp.concatenate([self.entities[k].position for k in self.entities])

    def get_tags(self) -> Array:
        """Get the tags of all the entities in the state."""
        return jnp.concatenate([self.entities[k].tag for k in self.entities])

    def get_sprites(self) -> Array:
        """Get the sprites of all the entities in the state."""
        return jnp.concatenate([self.entities[k].sprite for k in self.entities])

    def get_transparency(self) -> Array:
        """Get the transparency of all the entities in the state."""
        return jnp.concatenate([self.entities[k].transparent for k in self.entities])
